from typing import List
import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import percentile_filter
import torch
from torch.nested import nested_tensor
from torchquad import Simpson
from torchaudio.functional import fftconvolve
from .utils import ensure_tensor, j_nu, thin_plate_spline, pad_array, unpad_array, vmacro_kernel
from .misc.priors import GaussianLogPrior, UniformLogPrior, DeltaLogPrior
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def mse_loss(pred, target, pred_errs, target_errs):
    total_errs = torch.sqrt(pred_errs ** 2 + target_errs ** 2)
    return torch.mean(((pred - target) / total_errs) ** 2, axis=[1, 2])


def gaussian_log_likelihood(pred, target, pred_errs, target_errs):  # , mask):
    tot_vars = pred_errs ** 2 + target_errs ** 2
    loglike = -0.5 * (
            torch.log(2 * np.pi * tot_vars) + (target - pred) ** 2 / (tot_vars)
    )
    mask = torch.isfinite(loglike)
    loglike[..., ~mask] = 0  # Cludgy masking
    return torch.sum(loglike, axis=[1, 2])


def gaussian_log_likelihood_multi(pred, target, pred_errs, target_errs):
    tot_vars = pred_errs ** 2 + target_errs ** 2
    loglike = -0.5 * (
        torch.log(2 * np.pi * tot_vars) + (target - pred) ** 2 / (tot_vars)
    )
    return torch.sum(loglike, axis=[1, 2, 3])


class PayneEmulator(torch.nn.Module):
    def __init__(
        self,
        model,
        cont_deg,
        rv_scale=100,
        cont_wave_norm_range=(-10,10),
        obs_wave=None,
        obs_blaz=None,
        model_res=None,
        include_model_errs=True,
        vmacro_method='iso',
    ):
        super(PayneEmulator, self).__init__()
        self.model = model
        self.include_model_errs = include_model_errs
        self.mod_wave = ensure_tensor(self.model.wavelength, precision=torch.float64)
        self.mod_errs = ensure_tensor(self.model.mod_errs) if self.include_model_errs else None
        self.model_res = model_res
        self.labels = model.labels
        self.stellar_labels_min = ensure_tensor(list(model.x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(model.x_max.values()))
        self.n_stellar_labels = self.model.input_dim

        if vmacro_method == 'rt_fft':
            self.vmacro_broaden = self.vmacro_rt_broaden_fft
        elif vmacro_method == 'iso_fft':
            self.vmacro_broaden = self.vmacro_iso_broaden_fft
        elif vmacro_method == 'iso':
            self.vmacro_broaden = self.vmacro_iso_broaden
        else:
            print("vmacro_method not recognized, defaulting to 'iso'")
            self.vmacro_broaden = self.vmacro_iso_broaden

        self.rv_scale = rv_scale
        self.cont_deg = cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = cont_wave_norm_range

        if obs_wave is not None:
            self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        else:
            self.obs_wave = ensure_tensor(self.mod_wave.view(1, -1), precision=torch.float64)
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

        if obs_blaz is not None:
            self.obs_blaz = ensure_tensor(obs_blaz)
        else:
            self.obs_blaz = torch.ones_like(self.obs_wave)

        #self.n_mod_pix = self.mod_wave.shape[0]
        self.n_obs_ord = self.obs_wave.shape[0]
        self.n_obs_pix_per_ord = self.obs_wave.shape[1]

    def scale_wave(self, wave):
        old_len = wave[:, -1] - wave[:, 0]
        new_len = self.cont_wave_norm_range[1] - self.cont_wave_norm_range[0]
        offset = (wave[:, -1] * self.cont_wave_norm_range[0] - wave[:, 0] * self.cont_wave_norm_range[1]) / old_len
        scale = new_len / old_len
        new_wave = offset[:, np.newaxis] + scale[:, np.newaxis] * wave
        return new_wave, offset, scale

    @staticmethod
    def calc_cont(coeffs, wave_):
        cont_flux = torch.einsum('ij, ijk -> jk', coeffs, wave_)
        return cont_flux

    @staticmethod
    def interp(x, y, x_new, fill):
        y_ = y.unsqueeze(0) if y.ndim == 1 else y
        out_of_bounds = (x_new < x[0]) | (x_new > x[-1])
        x_new_indices = torch.searchsorted(x, x_new)
        x_new_indices = x_new_indices.clamp(1, x.shape[0] - 1)
        lo = x_new_indices - 1
        hi = x_new_indices
        x_lo = x[lo]
        x_hi = x[hi]
        if (y_.shape[0] == lo.shape[0] == hi.shape[0]) and (y_.shape[0] > 1):
            # Separate interpolation for each spectrum
            y_lo = torch.vstack([y_[i, lo[i]] for i in range(lo.shape[0])])
            y_hi = torch.vstack([y_[i, hi[i]] for i in range(hi.shape[0])])
        else:
            y_lo = y_[..., lo]
            y_hi = y_[..., hi]
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        y_new = slope * (x_new - x_lo) + y_lo
        y_new[..., out_of_bounds] = fill
        return y_new

    @staticmethod
    def inst_broaden(wave, flux, errs, inst_res, model_res=None):
        sigma_out = (inst_res * 2.355) ** -1
        if model_res is None:
            sigma_in = 0.0
        else:
            sigma_in = (model_res * 2.355) ** -1
        sigma = torch.sqrt(sigma_out ** 2 - sigma_in ** 2)
        inv_res_grid = torch.diff(torch.log(wave))
        dx = torch.median(inv_res_grid)
        ss = torch.fft.rfftfreq(wave.shape[-1], d=dx)
        sigma_ = sigma.repeat(ss.shape[0]).view(ss.shape[0], -1).T
        ss_ = ss.repeat(1, sigma.shape[0]).view(sigma.shape[0], -1)
        kernel = torch.exp(-2 * (np.pi ** 2) * (sigma_ ** 2) * (ss_ ** 2))
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= kernel.unsqueeze(1)
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel.unsqueeze(1)
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def rot_broaden(wave, flux, errs, vsini):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        ub = 2.0 * np.pi * vsini.unsqueeze(-1) * freq[1:]
        j1_term = j_nu(ub, 1) / ub
        cos_term = 3.0 * torch.cos(ub) / (2 * ub ** 2)
        sin_term = 3.0 * torch.sin(ub) / (2 * ub ** 3)
        sb = j1_term - cos_term + sin_term
        # Clean up rounding errors at low frequency; Should be safe for vsini > 0.1 km/s
        low_freq_idx = (freq[1:].repeat(vsini.shape[0], 1).T < freq[1:][torch.argmax(sb, dim=-1)]).T
        sb[low_freq_idx] = 1.0
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= torch.hstack([torch.ones(vsini.shape[0], 1), sb])
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).repeat(vsini.shape[0], 1)
            errs_ff *= torch.hstack([torch.ones(vsini.shape[0], 1), sb])
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv, errs_conv

    def doppler_shift(self, wave, flux, errs, rv):
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((1 - rv / c) / (1 + rv / c))
        new_wave = wave.unsqueeze(0) * doppler_factor.unsqueeze(-1)
        shifted_flux = self.interp(wave, flux, new_wave, fill=1.0).squeeze()
        if errs is not None:
            shifted_errs = self.interp(wave, errs, new_wave, fill=np.inf).squeeze()
        else:
            shifted_errs = None
        return shifted_flux, shifted_errs

    @staticmethod
    def vmacro_iso_broaden(wave, flux, errs, vmacro, ks: int = 21):
        wave = wave.to(torch.float32)
        n_spec = flux.shape[0]
        d_wave = wave[1] - wave[0]
        eff_wave = torch.median(wave)
        loc = (torch.arange(ks) - (ks - 1) // 2) * d_wave
        scale = vmacro / 3e5 * eff_wave
        norm = torch.distributions.normal.Normal(
            loc=torch.zeros(ks, 1),
            scale=scale.view(1, -1).repeat(ks, 1)
        )
        kernel = norm.log_prob(loc.view(-1, 1).repeat(1, n_spec)).exp()
        kernel = kernel / kernel.sum(axis=0)
        conv_spec = torch.nn.functional.conv1d(
            input=flux.view(1, n_spec, -1),
            weight=kernel.T.view(n_spec, 1, -1),
            padding=ks // 2,
            groups=n_spec,
        ).squeeze()
        if errs is not None:
            conv_errs = torch.nn.functional.conv1d(
                input=errs.repeat(1, n_spec, 1),
                weight=kernel.T.view(n_spec, 1, -1),
                padding=ks // 2,
                groups=n_spec,
            ).squeeze()
        else:
            conv_errs = None
        return conv_spec, conv_errs

    @staticmethod
    def vmacro_iso_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        kernel = torch.exp(-2 * (np.pi * vmacro * freq) ** 2)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv.squeeze(), errs_conv.squeeze()

    @staticmethod
    def vmacro_rt_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        kernel = (1 - torch.exp(-1*(np.pi*vmacro*freq)**2))/(np.pi*vmacro*freq)**2
        kernel[0] = 1.0
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv.squeeze(), errs_conv.squeeze()

    def scale_stellar_labels(self, unscaled_labels):
        x_min = self.stellar_labels_min
        x_max = self.stellar_labels_max
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = np.array(list(self.model.x_min.values()))
        x_max = np.array(list(self.model.x_max.values()))
        return (scaled_labels + 0.5) * (x_max - x_min) + x_min

    def forward_model_spec(self, norm_flux, norm_errs, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        # Macroturbulent Broadening
        if vmacro is not None:
            conv_flux, conv_errs = self.vmacro_broaden(
                wave=self.mod_wave,
                flux=norm_flux,
                errs=norm_errs,
                vmacro=vmacro,
                ks=21,
            )
        else:
            conv_flux = norm_flux
            conv_errs = self.mod_errs
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
                wave=self.mod_wave,
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # RV Shift
        shifted_flux, shifted_errs = self.doppler_shift(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            rv=rv * self.rv_scale,
        )
        # Interpolate to Observed Wavelength
        intp_flux = self.interp(
            x=self.mod_wave,
            y=shifted_flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        if self.include_model_errs:
            intp_errs = self.interp(
                x=self.mod_wave,
                y=shifted_errs,
                x_new=self.obs_wave,
                fill=1.0,
            )
        else:
            intp_errs = None
        # Instrumental Broadening
        if inst_res is not None:
            intp_flux, intp_errs = self.inst_broaden(
                wave=self.obs_wave,
                flux=intp_flux,
                errs=intp_errs,
                inst_res=inst_res,
                model_res=self.model_res,
            )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        if self.include_model_errs:
            return intp_flux * cont_flux * self.obs_blaz, intp_errs * cont_flux * self.obs_blaz
        else:
            return intp_flux * cont_flux * self.obs_blaz, torch.zeros_like(intp_flux)

    def numpy(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        flux, errs = self.forward(stellar_labels, rv, vmacro, cont_coeffs, inst_res, vsini)
        return flux.detach().numpy(), errs.detach().numpy()

    def forward(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        # Model Spectrum
        norm_flux = self.model(stellar_labels)
        # Macroturbulent Broadening
        if vmacro is not None:
            conv_flux, conv_errs = self.vmacro_broaden(
                wave=self.mod_wave,
                flux=norm_flux,
                errs=self.mod_errs,
                vmacro=vmacro,
                ks=21,
            )
        else:
            conv_flux = norm_flux
            conv_errs = self.mod_errs
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
                wave=self.mod_wave,
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # RV Shift
        shifted_flux, shifted_errs = self.doppler_shift(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            rv=rv * self.rv_scale,
        )
        # Interpolate to Observed Wavelength
        intp_flux = self.interp(
            x=self.mod_wave,
            y=shifted_flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        if self.include_model_errs:
            intp_errs = self.interp(
                x=self.mod_wave,
                y=shifted_errs,
                x_new=self.obs_wave,
                fill=np.inf,
            )
        else:
            intp_errs = None
        # Instrumental Broadening
        if inst_res is not None:
            intp_flux, intp_errs = self.inst_broaden(
                wave=self.obs_wave,
                flux=intp_flux,
                errs=intp_errs,
                inst_res=inst_res,
                model_res=self.model_res,
            )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        if self.include_model_errs:
            return intp_flux * cont_flux * self.obs_blaz, intp_errs * cont_flux * self.obs_blaz
        else:
            return intp_flux * cont_flux * self.obs_blaz, torch.zeros_like(intp_flux)


class CompositePayneEmulator(torch.nn.Module):
    def __init__(
            self,
            models: List[torch.nn.Module],
            model_bounds,
            cont_deg,
            rv_scale=100,
            cont_wave_norm_range=(-10, 10),
            obs_wave=None,
            obs_blaz=None,
            model_res=None,
            include_model_errs=True,
            vmacro_method='iso',
    ):
        super(CompositePayneEmulator, self).__init__()
        self.models = models
        self.n_models = len(self.models)
        self.include_model_errs = include_model_errs
        self.mod_wave = [ensure_tensor(model.wavelength, precision=torch.float64) for model in self.models]
        if self.include_model_errs:
            self.mod_errs = [ensure_tensor(model.mod_errs) for model in self.models]
        else:
            self.mod_errs = [None for model in self.models]
        self.mod_bounds = [ensure_tensor(model_bound) for model_bound in model_bounds]
        self.model_res = model_res
        self.labels = self.models[0].labels
        self.stellar_labels_min = ensure_tensor(list(self.models[0].x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(self.models[0].x_max.values()))
        self.n_stellar_labels = self.models[0].input_dim

        if vmacro_method == 'rt_fft':
            self.vmacro_broaden = self.vmacro_rt_broaden_fft
        elif vmacro_method == 'iso_fft':
            self.vmacro_broaden = self.vmacro_iso_broaden_fft
        elif vmacro_method == 'iso':
            self.vmacro_broaden = self.vmacro_iso_broaden
        else:
            print("vmacro_method not recognized, defaulting to 'iso'")
            self.vmacro_broaden = self.vmacro_iso_broaden

        self.rv_scale = rv_scale
        self.cont_deg = cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = cont_wave_norm_range

        if obs_wave is not None:
            self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        else:
            self.obs_wave = ensure_tensor(self.mod_wave.view(1, -1), precision=torch.float64)
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

        if obs_blaz is not None:
            self.obs_blaz = ensure_tensor(obs_blaz)
        else:
            self.obs_blaz = torch.ones_like(self.obs_wave)

        self.n_mod_pix = self.mod_wave.shape[1]
        self.n_obs_ord = self.obs_wave.shape[0]
        self.n_obs_pix_per_ord = self.obs_wave.shape[1]


    def scale_wave(self, wave):
        old_len = wave[:, -1] - wave[:, 0]
        new_len = self.cont_wave_norm_range[1] - self.cont_wave_norm_range[0]
        offset = (wave[:, -1] * self.cont_wave_norm_range[0] - wave[:, 0] * self.cont_wave_norm_range[1]) / old_len
        scale = new_len / old_len
        new_wave = offset[:, np.newaxis] + scale[:, np.newaxis] * wave
        return new_wave, offset, scale

    @staticmethod
    def calc_cont(coeffs, wave_):
        cont_flux = torch.einsum('ij, ijk -> jk', coeffs, wave_)
        return cont_flux

    @staticmethod
    def interp(x, y, x_new, fill):
        y_ = y.unsqueeze(0) if y.ndim == 1 else y
        out_of_bounds = (x_new < x[0]) | (x_new > x[-1])
        x_new_indices = torch.searchsorted(x, x_new)
        x_new_indices = x_new_indices.clamp(1, x.shape[0] - 1)
        lo = x_new_indices - 1
        hi = x_new_indices
        x_lo = x[lo]
        x_hi = x[hi]
        if (y_.shape[0] == lo.shape[0] == hi.shape[0]) and (y_.shape[0] > 1):
            # Separate interpolation for each spectrum
            y_lo = torch.vstack([y_[i, lo[i]] for i in range(lo.shape[0])])
            y_hi = torch.vstack([y_[i, hi[i]] for i in range(hi.shape[0])])
        else:
            y_lo = y_[..., lo]
            y_hi = y_[..., hi]
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        y_new = slope * (x_new - x_lo) + y_lo
        y_new[..., out_of_bounds] = fill
        return y_new

    @staticmethod
    def inst_broaden(wave, flux, errs, inst_res, model_res=None):
        sigma_out = (inst_res * 2.355) ** -1
        if model_res is None:
            sigma_in = 0.0
        else:
            sigma_in = (model_res * 2.355) ** -1
        sigma = torch.sqrt(sigma_out ** 2 - sigma_in ** 2)
        inv_res_grid = torch.diff(torch.log(wave))
        dx = torch.median(inv_res_grid)
        ss = torch.fft.rfftfreq(wave.shape[-1], d=dx)
        sigma_ = sigma.repeat(ss.shape[0]).view(ss.shape[0], -1).T
        ss_ = ss.repeat(1, sigma.shape[0]).view(sigma.shape[0], -1)
        kernel = torch.exp(-2 * (np.pi ** 2) * (sigma_ ** 2) * (ss_ ** 2))
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= kernel.unsqueeze(1)
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel.unsqueeze(1)
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def rot_broaden(wave, flux, errs, vsini):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        ub = 2.0 * np.pi * vsini.unsqueeze(-1) * freq[1:]
        j1_term = j_nu(ub, 1) / ub
        cos_term = 3.0 * torch.cos(ub) / (2 * ub ** 2)
        sin_term = 3.0 * torch.sin(ub) / (2 * ub ** 3)
        sb = j1_term - cos_term + sin_term
        # Clean up rounding errors at low frequency; Should be safe for vsini > 0.1 km/s
        low_freq_idx = (freq[1:].repeat(vsini.shape[0], 1).T < freq[1:][torch.argmax(sb, dim=-1)]).T
        sb[low_freq_idx] = 1.0
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= torch.hstack([torch.ones(vsini.shape[0], 1), sb])
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).repeat(vsini.shape[0], 1)
            errs_ff *= torch.hstack([torch.ones(vsini.shape[0], 1), sb])
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv, errs_conv

    def doppler_shift(self, wave, flux, errs, rv):
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((1 - rv / c) / (1 + rv / c))
        new_wave = wave.unsqueeze(0) * doppler_factor.unsqueeze(-1)
        shifted_flux = self.interp(wave, flux, new_wave, fill=1.0).squeeze()
        if errs is not None:
            shifted_errs = self.interp(wave, errs, new_wave, fill=np.inf).squeeze()
        else:
            shifted_errs = None
        return shifted_flux, shifted_errs

    @staticmethod
    def vmacro_iso_broaden(wave, flux, errs, vmacro, ks: int = 21):
        wave = wave.to(torch.float32)
        n_spec = flux.shape[0]
        d_wave = wave[1] - wave[0]
        eff_wave = torch.median(wave)
        loc = (torch.arange(ks) - (ks - 1) // 2) * d_wave
        scale = vmacro / 3e5 * eff_wave
        norm = torch.distributions.normal.Normal(
            loc=torch.zeros(ks, 1),
            scale=scale.view(1, -1).repeat(ks, 1)
        )
        kernel = norm.log_prob(loc.view(-1, 1).repeat(1, n_spec)).exp()
        kernel = kernel / kernel.sum(axis=0)
        conv_spec = torch.nn.functional.conv1d(
            input=flux.view(1, n_spec, -1),
            weight=kernel.T.view(n_spec, 1, -1),
            padding=ks // 2,
            groups=n_spec,
        ).squeeze()
        if errs is not None:
            conv_errs = torch.nn.functional.conv1d(
                input=errs.repeat(1, n_spec, 1),
                weight=kernel.T.view(n_spec, 1, -1),
                padding=ks // 2,
                groups=n_spec,
            ).squeeze()
        else:
            conv_errs = None
        return conv_spec, conv_errs

    @staticmethod
    def vmacro_iso_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        kernel = torch.exp(-2 * (np.pi * vmacro * freq) ** 2)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv.squeeze(), errs_conv.squeeze()

    @staticmethod
    def vmacro_rt_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        kernel = (1 - torch.exp(-1*(np.pi*vmacro*freq)**2))/(np.pi*vmacro*freq)**2
        kernel[0] = 1.0
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv.squeeze(), errs_conv.squeeze()

    def scale_stellar_labels(self, unscaled_labels):
        x_min = np.array(list(self.models[0].x_min.values()))
        x_max = np.array(list(self.models[0].x_max.values()))
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = np.array(list(self.models[0].x_min.values()))
        x_max = np.array(list(self.models[0].x_max.values()))
        return (scaled_labels + 0.5) * (x_max - x_min) + x_min

    def forward_model_spec(self, norm_flux, norm_errs, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        flux_list = []
        errs_list = []
        wave_list = []
        for i, model in enumerate(self.models):
            # Macroturbulent Broadening
            if vmacro is not None:
                conv_flux, conv_errs = self.vmacro_broaden(
                    wave=self.mod_wave[i],
                    flux=norm_flux[i],
                    errs=norm_errs[i],
                    vmacro=vmacro,
                    ks=21,
                )
            else:
                conv_flux = norm_flux[i]
                conv_errs = self.mod_errs[i]
            # Rotational Broadening
            if vsini is not None:
                conv_flux, conv_errs = self.rot_broaden(
                    wave=self.mod_wave[i],
                    flux=conv_flux,
                    errs=conv_errs,
                    vsini=vsini,
                )
            # RV Shift
            shifted_flux, shifted_errs = self.doppler_shift(
                wave=self.mod_wave[i],
                flux=conv_flux,
                errs=conv_errs,
                rv=rv * self.rv_scale,
            )
            cut = torch.where((self.mod_wave[i] > self.mod_bounds[i][0]) & (self.mod_wave[i] < self.mod_bounds[i][1]))[
                0]
            wave_list.append(self.mod_wave[i][cut])
            flux_list.append(shifted_flux[..., cut])
            if self.include_model_errs:
                errs_list.append(shifted_errs[..., cut])
        wave = torch.cat(wave_list, axis=-1)
        flux = torch.cat(flux_list, axis=-1)
        if self.include_model_errs:
            errs = torch.cat(errs_list, axis=-1)
        else:
            errs = None
        # Interpolate to Observed Wavelength
        intp_flux = self.interp(
            x=self.mod_wave[i],
            y=shifted_flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        if self.include_model_errs:
            intp_errs = self.interp(
                x=self.mod_wave[i],
                y=shifted_errs,
                x_new=self.obs_wave,
                fill=np.inf,
            )
        else:
            intp_errs = None
        # Instrumental Broadening
        if inst_res is not None:
            intp_flux, intp_errs = self.inst_broaden(
                wave=self.obs_wave,
                flux=intp_flux,
                errs=intp_errs,
                inst_res=inst_res,
                model_res=self.model_res,
            )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        if self.include_model_errs:
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)

    def forward(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        flux_list = []
        errs_list = []
        wave_list = []
        for i, model in enumerate(self.models):
            # Model Spectrum
            norm_flux = model(stellar_labels)
            # Macroturbulent Broadening
            if vmacro is not None:
                conv_flux, conv_errs = self.vmacro_broaden(
                    wave=self.mod_wave[i],
                    flux=norm_flux,
                    errs=self.mod_errs[i],
                    vmacro=vmacro,
                    ks=21,
                )
            else:
                conv_flux = norm_flux
                conv_errs = self.mod_errs[i]
            # Rotational Broadening
            if vsini is not None:
                conv_flux, conv_errs = self.rot_broaden(
                    wave=self.mod_wave[i],
                    flux=conv_flux,
                    errs=conv_errs,
                    vsini=vsini,
                )
            # RV Shift
            shifted_flux, shifted_errs = self.doppler_shift(
                wave=self.mod_wave[i],
                flux=conv_flux,
                errs=conv_errs,
                rv=rv * self.rv_scale,
            )
            cut = torch.where((self.mod_wave[i] > self.mod_bounds[i][0]) & (self.mod_wave[i] < self.mod_bounds[i][1]))[
                0]
            wave_list.append(self.mod_wave[i][cut])
            flux_list.append(shifted_flux[..., cut])
            if self.include_model_errs:
                errs_list.append(shifted_errs[..., cut])
        wave = torch.cat(wave_list, axis=-1)
        flux = torch.cat(flux_list, axis=-1)
        if self.include_model_errs:
            errs = torch.cat(errs_list, axis=-1)
        else:
            errs = None
        # Interpolate to Observed Wavelength
        intp_flux = self.interp(
            x=wave,
            y=flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        if self.include_model_errs:
            intp_errs = self.interp(
                x=wave,
                y=errs,
                x_new=self.obs_wave,
                fill=np.inf,
            )
        else:
            intp_errs = None
        # Instrumental Broadening
        if inst_res is not None:
            intp_flux, intp_errs = self.inst_broaden(
                wave=self.obs_wave,
                flux=intp_flux,
                errs=intp_errs,
                inst_res=inst_res,
                model_res=self.model_res,
            )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        if self.include_model_errs:
            return intp_flux * cont_flux * self.obs_blaz, intp_errs * cont_flux * self.obs_blaz
        else:
            return intp_flux * cont_flux * self.obs_blaz, torch.zeros_like(intp_flux)


class PayneOrderEmulator(PayneEmulator):
    def __init__(
            self,
            models: List[torch.nn.Module],
            cont_deg,
            rv_scale=100,
            cont_wave_norm_range=(-10, 10),
            obs_wave=None,
            obs_blaz=None,
            model_res=None,
            include_model_errs=True,
    ):
        super(PayneEmulator, self).__init__()
        self.models = models
        self.n_models = self.n_mod_ord = len(self.models)
        self.include_model_errs = include_model_errs

        # Define Model Inputs & Scalings
        self.model_res = model_res
        self.labels = self.models[0].labels
        self.stellar_labels_min = ensure_tensor(list(self.models[0].x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(self.models[0].x_max.values()))
        self.n_stellar_labels = self.models[0].input_dim
        self.rv_scale = rv_scale
        self.cont_deg = cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = cont_wave_norm_range

        # Set Model Broadening Methods
        self.vmacro_integrator = Simpson().get_jit_compiled_integrate(
            dim=1, N=1001, backend="torch"
        )
        self.vmacro_broaden = self.VmacroRTBroad
        self.rot_broaden = self.RotBroadApprox
        self.inst_broaden = None

        # Stitch Model Orders Together
        self.ragged_mods = len(np.unique([model.wavelength.shape[0] for model in models])) > 1
        if self.ragged_mods:
            self.mod_wave_ragged = nested_tensor(
                [ensure_tensor(model.wavelength, precision=torch.float64) for model in self.models]
            )
            self.n_mod_pix = ensure_tensor([self.mod_wave_ragged[i].shape[0] for i in range(self.n_mod_ord)],
                                           precision=int)
            self.n_mod_pad = torch.max(self.n_mod_pix) - self.n_mod_pix
            print(
                f"Model orders do not have the same lengths ({', '.join([str(n_pix.item()) for n_pix in self.n_mod_pix])}); "
                f"padding all to n_pix = {torch.max(self.n_mod_pix)}")
            self.mod_wave = self.mod_wave_ragged.to_padded_tensor(padding=0)
            for i in range(self.n_mod_ord):
                if self.n_mod_pad[i] > 0:
                    dx = torch.mean(torch.diff(self.mod_wave_ragged[i]))
                    self.mod_wave[i, -self.n_mod_pad[i]:] = torch.linspace(
                        self.mod_wave[i, -self.n_mod_pad[i] - 1],
                        (self.mod_wave[i, -self.n_mod_pad[i] - 1]) + (self.n_mod_pad[i] - 1) * dx,
                        self.n_mod_pad[i]
                    )
            if self.include_model_errs:
                self.mod_errs_ragged = nested_tensor(
                    [ensure_tensor(model.mod_errs) for model in self.models]
                )
                self.mod_errs = self.mod_errs_ragged.to_padded_tensor(padding=1)
            else:
                self.mod_errs = None
        else:
            self.mod_wave = torch.vstack(
                [ensure_tensor(model.wavelength, precision=torch.float64) for model in self.models]
            )
            if self.include_model_errs:
                self.mod_errs = torch.vstack(
                    [ensure_tensor(model.mod_errs) for model in self.models]
                )
            else:
                self.mod_errs = None

        # Stitch Observation Orders Together
        if obs_wave is not None:
            self.n_obs_ord = len(obs_wave)
            self.ragged_obs = len(np.unique([wave.shape[0] for wave in obs_wave])) > 1
            if self.ragged_obs:
                self.obs_wave_ragged = nested_tensor(
                    [ensure_tensor(wave, precision=torch.float64) for wave in obs_wave]
                )
                self.n_obs_pix = ensure_tensor([self.obs_wave_ragged[i].shape[0] for i in range(self.n_obs_ord)],
                                               precision=int)
                self.n_obs_pad = torch.max(self.n_obs_pix) - self.n_obs_pix
                print(
                    f"Observed orders do not have the same lengths ({', '.join([str(n_pix.item()) for n_pix in self.n_obs_pix])}); "
                    f"padding all to n_pix = {torch.max(self.n_obs_pix)}")
                self.obs_wave = self.obs_wave_ragged.to_padded_tensor(padding=0)
                for i in range(self.n_mod_ord):
                    if self.n_obs_pad[i] > 0:
                        dx = torch.mean(torch.diff(self.obs_wave_ragged[i]))
                        self.obs_wave[i, -self.n_obs_pad[i]:] = torch.linspace(
                            self.obs_wave[i, -self.n_obs_pad[i] - 1],
                            (self.obs_wave[i, -self.n_obs_pad[i] - 1]) + (self.n_obs_pad[i] - 1) * dx,
                            self.n_obs_pad[i]
                        )
                if obs_blaz is not None:
                    self.obs_blaz_ragged = nested_tensor([ensure_tensor(blaz) for blaz in obs_blaz])
                    self.obs_blaz = self.obs_blaz_ragged.to_padded_tensor(padding=0)
                else:
                    self.obs_blaz = torch.ones_like(self.obs_wave)
            else:
                self.obs_wave = torch.vstack(
                    [ensure_tensor(wave, precision=torch.float64) for wave in obs_wave]
                )
                if obs_blaz is not None:
                    self.obs_blaz = torch.vstack(
                        [ensure_tensor(blaz) for blaz in obs_wave]
                    )
                else:
                    self.obs_blaz = torch.ones_like(self.obs_wave)
        else:
            self.obs_wave = self.mod_wave
            self.obs_blaz = torch.ones_like(self.obs_wave)

        # Prepare Continuum Wavelength Grid
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

        # Continuum coefficients for a flat continuum
        self.c_flat = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        self.c_flat[0] = 1.0

    def set_obs_wave(self, obs_wave):
        self.n_obs_ord = len(obs_wave)
        self.ragged_obs = len(np.unique([wave.shape[0] for wave in obs_wave])) > 1
        if self.ragged_obs:
            self.obs_wave_ragged = nested_tensor(
                [ensure_tensor(wave, precision=torch.float64) for wave in obs_wave]
            )
            self.n_obs_pix = ensure_tensor([self.obs_wave_ragged[i].shape[0] for i in range(self.n_obs_ord)],
                                           precision=int)
            self.n_obs_pad = torch.max(self.n_obs_pix) - self.n_obs_pix
            print(
                f"Observed orders do not have the same lengths ({', '.join([str(n_pix.item()) for n_pix in self.n_obs_pix])}); "
                f"padding all to n_pix = {torch.max(self.n_obs_pix)}")
            self.obs_wave = self.obs_wave_ragged.to_padded_tensor(padding=0)
            for i in range(self.n_mod_ord):
                if self.n_obs_pad[i] > 0:
                    dx = torch.mean(torch.diff(self.obs_wave_ragged[i]))
                    self.obs_wave[i, -self.n_obs_pad[i]:] = torch.linspace(
                        self.obs_wave[i, -self.n_obs_pad[i] - 1],
                        (self.obs_wave[i, -self.n_obs_pad[i] - 1]) + (self.n_obs_pad[i] - 1) * dx,
                        self.n_obs_pad[i]
                    )
        else:
            self.obs_wave = torch.vstack(
                [ensure_tensor(wave, precision=torch.float64) for wave in obs_wave]
            )
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

    def set_obs_blaz(self, obs_blaz):
        if self.ragged_obs:
            self.obs_blaz_ragged = nested_tensor([ensure_tensor(blaz) for blaz in obs_blaz])
            self.obs_blaz = self.obs_blaz_ragged.to_padded_tensor(padding=0)
        else:
            self.obs_blaz = torch.vstack([ensure_tensor(blaz) for blaz in obs_blaz])

    def scale_stellar_labels(self, unscaled_labels):
        x_min = self.stellar_labels_min
        x_max = self.stellar_labels_max
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = self.stellar_labels_min
        x_max = self.stellar_labels_max
        return (scaled_labels + 0.5) * (x_max - x_min) + x_min

    def VmacroRTBroad(self, wave, flux, errs, vmacro, vmacro_rad=None, vmacro_tan=None, Ar=1, At=1):
        """
        Adopted from iSpec/FASMA
        """
        if vmacro_rad is not None and vmacro_tan is not None:
            vmacro_rad_c = vmacro_rad / 2.99792458e5
            vmacro_tan_c = vmacro_tan / 2.99792458e5
        else:
            vmacro_rad = vmacro_tan = vmacro
            vmacro_rad_c = vmacro_tan_c = vmacro_c = vmacro / 2.99792458e5
        n_spec = vmacro.shape[0]
        n_ord = wave.shape[0]
        dlambda_perpix = wave.diff()[:, 0]  # wave must be linearly spaced
        eff_wave = torch.mean(wave, axis=1)

        # Make Kernel
        n_pix_kern = torch.floor(torch.max(vmacro_tan_c, vmacro_rad_c) * eff_wave / dlambda_perpix) + 1
        n_pix_kern_max = torch.max(n_pix_kern)
        no_conv_necessary = True if n_pix_kern_max == 1 else False
        if no_conv_necessary:
            return flux, errs, no_conv_necessary
        dlambda = (torch.arange(14 * n_pix_kern_max) - 7 * n_pix_kern_max).unsqueeze(0) \
                  * dlambda_perpix.unsqueeze(1)
        kern_wave = dlambda + eff_wave.unsqueeze(1)
        Zr = vmacro_rad_c * eff_wave
        Zt = vmacro_tan_c * eff_wave
        kernel = vmacro_kernel(dlambda, Zr, Zt, integrator=self.vmacro_integrator, Ar=Ar, At=At)

        # Convolve Spectra
        flux_ = pad_array(flux, n_pix_kern_max, 1)
        conv_flux_ = dlambda_perpix.unsqueeze(0).unsqueeze(2) * fftconvolve(flux_, kernel, mode='same')
        conv_flux = unpad_array(conv_flux_, n_pix_kern_max)
        if errs is not None:
            errs_ = pad_array(errs, n_pix_kern_max, 1)
            conv_errs_ = dlambda_perpix.unsqueeze(0).unsqueeze(2) * fftconvolve(errs_, kernel, mode='same')
            conv_errs = unpad_array(conv_errs_, n_pix_kern_max)
        else:
            conv_errs = None
        return conv_flux, conv_errs, no_conv_necessary

    @staticmethod
    def RotBroadApprox(wave, flux, errs, vsini, epsilon=0.6):
        """
        Adopted from PyAstronomy.pyasl.fastRotBroad
        """
        n_spec = vsini.shape[0]
        n_ord = wave.shape[0]
        vsini_c = vsini / 2.99792458e5
        dlambda_perpix = wave.diff()[:, 0]  # wave must be linearly spaced
        eff_wave = torch.mean(wave, axis=1)

        # Make Kernel
        n_pix_kern = torch.floor(vsini_c * eff_wave / dlambda_perpix) + 1
        n_pix_kern_max = torch.max(n_pix_kern)
        no_conv_necessary = True if n_pix_kern_max == 1 else False
        if no_conv_necessary:
            return flux, errs, no_conv_necessary
        dlambda = (torch.arange(4 * n_pix_kern_max) - 2 * n_pix_kern_max).unsqueeze(0) \
                  * dlambda_perpix.unsqueeze(1)
        kern_wave = dlambda + eff_wave.unsqueeze(1)
        dlambda_max = vsini_c * eff_wave
        c1 = 2 * (1 - epsilon) / (torch.pi * dlambda_max * (1 - epsilon / 3))
        c2 = epsilon / (2 * dlambda_max * (1 - epsilon / 3))
        x = dlambda.unsqueeze(0) / dlambda_max.unsqueeze(2)
        kernel = c1.unsqueeze(2) * torch.sqrt(1. - x ** 2) + c2.unsqueeze(2) * (1. - x ** 2)
        kernel[torch.abs(x) > 1] = 0
        kernel /= torch.trapz(kernel, dlambda.unsqueeze(0)).unsqueeze(2)
        kernel = kernel[:, :, torch.sum(kernel > 0, axis=(0, 1)) > 0]

        # Convolve Spectra
        flux_ = pad_array(flux, n_pix_kern_max, 1)
        conv_flux_ = dlambda_perpix.unsqueeze(0).unsqueeze(2) * fftconvolve(flux_, kernel, mode='same')
        conv_flux = unpad_array(conv_flux_, n_pix_kern_max)
        if errs is not None:
            errs_ = pad_array(errs, n_pix_kern_max, 1)
            conv_errs_ = dlambda_perpix.unsqueeze(0).unsqueeze(2) * fftconvolve(errs_, kernel, mode='same')
            conv_errs = unpad_array(conv_errs_, n_pix_kern_max)
        else:
            conv_errs = None
        return conv_flux, conv_errs, no_conv_necessary

    @staticmethod
    def RotBroadFull(wave, flux, errs, vsini, epsilon=0.6):
        """
        Adopted from PyAstronomy.pyasl.RotBroad
        Should probably add in padding of the flux/errs here
        """
        n_spec = vsini.shape[0]
        n_ord = wave.shape[0]
        vsini_c = vsini / 2.99792458e5
        dlambda_perpix = wave.diff()[:, 0]  # wave must be linearly spaced

        # Make Kernels
        n_pix_kern = torch.floor(
            vsini_c.unsqueeze(1) * wave.unsqueeze(0) / dlambda_perpix.unsqueeze(0).unsqueeze(2) + 1
        )
        n_pix_kern_max = torch.max(n_pix_kern)
        no_conv_necessary = True if n_pix_kern_max == 1 else False
        if no_conv_necessary:
            return flux, errs, no_conv_necessary
        dlambda = (torch.arange(4 * n_pix_kern_max) - 2 * n_pix_kern_max).unsqueeze(0) \
                  * dlambda_perpix.unsqueeze(1)
        kern_wave = dlambda.unsqueeze(1) + wave.unsqueeze(2)
        dlambda_max = vsini_c.unsqueeze(2) * wave.unsqueeze(0)
        c1 = 2 * (1 - epsilon) / (torch.pi * dlambda_max * (1 - epsilon / 3))
        c2 = epsilon / (2 * dlambda_max * (1 - epsilon / 3))
        x = dlambda.unsqueeze(0).unsqueeze(2) / dlambda_max.unsqueeze(3)
        kernel = c1.unsqueeze(3) * torch.sqrt(1. - x ** 2) + c2.unsqueeze(3) * (1. - x ** 2)
        kernel[torch.abs(x) > 1] = 0
        kernel /= torch.trapz(kernel, dlambda.unsqueeze(0).unsqueeze(2)).unsqueeze(3)
        kernel = kernel[:, :, :, torch.sum(kernel > 0, axis=(0, 1, 2)) > 0]

        # Convolve Spectra
        offsets = (torch.arange(kernel.shape[-1]) - torch.floor(torch.tensor(kernel.shape[-1] / 2))).to(int)
        conv_flux = torch.zeros(n_spec, n_ord, wave.shape[1])
        conv_errs = torch.zeros(n_spec, n_ord, wave.shape[1])
        for i in range(n_spec):
            for j in range(n_ord):
                kern_mat = torch.sparse.spdiags(
                    kernel[i, j].T,
                    offsets=offsets,
                    shape=(wave.shape[1], wave.shape[1])
                )
                conv_flux[i, j] = dlambda_perpix[j] * (kern_mat.to(torch.float32) @ flux[i, j])
            if errs is not None:
                conv_errs[i, j] = dlambda_perpix[j] * (kern_mat @ errs[i, j])
            else:
                conv_errs = None
        return conv_flux, conv_errs, no_conv_necessary

    @staticmethod
    def get_doppler_wave(wave, rv):
        n_spec = rv.shape[0]
        n_ords = wave.shape[0]
        n_pix = wave.shape[1]
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((c - rv) / (c + rv))
        shifted_wave = doppler_factor.repeat_interleave(n_ords * n_pix).view(n_spec * n_ords, -1) \
                       * wave.repeat(n_spec, 1, 1).view(n_spec * n_ords, -1)
        return shifted_wave.view(n_spec, n_ords, n_pix)

    @staticmethod
    def interp(x_old, x_new, y, fill):
        n_spec = x_old.shape[0]
        n_ords = x_old.shape[1]
        n_pix_old = x_old.shape[2]
        n_pix_new = x_new.shape[1]
        x_old_ = x_old.view(n_spec * n_ords, -1)
        x_new_ = x_new.repeat(n_spec, 1, 1).view(n_spec * n_ords, -1)
        y_ = y.reshape(n_spec * n_ords, -1)
        out_of_bounds = (x_new_.T < x_old_[:, 0].T).T \
                        | (x_new_.T > x_old_[:, -1].T).T
        x_new_indices = torch.searchsorted(x_old_, x_new_)
        x_new_indices = x_new_indices.clamp(1, x_old.shape[-1] - 1)
        lo = x_new_indices - 1
        hi = x_new_indices
        dim1_idx = torch.arange(n_spec * n_ords).repeat_interleave(n_pix_new).view(n_spec * n_ords, -1)
        x_lo = x_old_[dim1_idx, lo]
        x_hi = x_old_[dim1_idx, hi]
        y_lo = y_[dim1_idx, lo]
        y_hi = y_[dim1_idx, hi]
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        y_new = slope * (x_new_ - x_lo) + y_lo
        y_new[out_of_bounds] = fill
        return y_new.view(n_spec, n_ords, n_pix_new)

    def interp_flux(self, old_wave, new_wave, flux, errs):
        if (flux.shape[0] == 1) and (old_wave.shape[0] > 1):
            # Handle case of singular model spec, multiple RV shifts (e.g., for RV initialization)
            flux = flux.repeat(old_wave.shape[0], 1, 1)
        elif (flux.shape[0] != old_wave.shape[0]):
            raise RuntimeError("RV input and stellar label input have conflicting shapes")
        intp_flux = self.interp(
            x_old=old_wave,
            x_new=new_wave,
            y=flux,
            fill=1.0,
        )
        if errs is not None:
            if (errs.shape[0] == 1) and (old_wave.shape[0] > 1):
                # Handle case of singular model spec, multiple RV shifts (e.g., for RV initialization)
                errs = errs.repeat(old_wave.shape[0], 1, 1)
            elif (errs.shape[0] != old_wave.shape[0]):
                raise RuntimeError("RV input and stellar label input have conflicting shapes")
            intp_errs = self.interp(
                x_old=old_wave,
                x_new=new_wave,
                y=errs,
                fill=1.0,
            )
        else:
            intp_errs = None
        return intp_flux, intp_errs

    def forward(self, stellar_labels, rv, vmacro=None, vsini=None, cont_coeffs=None, inst_res=None, skip_cont=False):
        # Generate Normalized Model Spectrum
        n_spec = stellar_labels.shape[0]
        norm_flux = nested_tensor([model(stellar_labels) for model in self.models]).to_padded_tensor(padding=1).movedim(
            0, 1)
        norm_errs = self.mod_errs.repeat(n_spec, 1, 1) if self.include_model_errs else None
        # Macroturbulent Broadening
        if vmacro is not None:
            conv_flux, conv_errs, _ = self.vmacro_broaden(
                wave=self.mod_wave,
                flux=norm_flux,
                errs=norm_errs,
                vmacro=vmacro,
            )
        else:
            conv_flux, conv_errs = norm_flux, norm_errs
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs, _ = self.rot_broaden(
                wave=self.mod_wave,
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # Doppler Shift
        doppler_wave = self.get_doppler_wave(
            wave=self.mod_wave,
            rv=rv * self.rv_scale,
        )
        # Instrumental Broadening
        if inst_res is not None:
            conv_flux, conv_errs = self.inst_broaden(
                wave=doppler_wave,
                flux=norm_flux,
                errs=norm_errs,
                inst_res=inst_res,
                model_res=self.model_res,
            )
        # Interpolation to Observed Wavelength
        intp_flux, intp_errs = self.interp_flux(
            old_wave=doppler_wave,
            new_wave=self.obs_wave,
            flux=conv_flux,
            errs=conv_errs,
        )
        if skip_cont or cont_coeffs is None:
            cont_flux = 1
        else:
            cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_) * self.obs_blaz
        if self.include_model_errs:
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)


class PayneStitchedEmulator(PayneEmulator):
    def __init__(
            self,
            models: List[torch.nn.Module],
            cont_deg,
            rv_scale=100,
            cont_wave_norm_range=(-10, 10),
            obs_wave=None,
            obs_blaz=None,
            model_res=None,
            include_model_errs=True,
            vmacro_method='iso_fft',
    ):
        super(PayneEmulator, self).__init__()
        self.models = models
        self.n_models = len(self.models)
        self.include_model_errs = include_model_errs
        self.mod_wave = torch.vstack(
            [ensure_tensor(model.wavelength, precision=torch.float64) for model in self.models]).flatten().unsqueeze(0)
        if not torch.all(self.mod_wave.diff() > 0):
            raise RuntimeError('Model Wavelengths are not in ascending order!')
        self.n_mod_pix_indiv = self.models[0].wavelength.shape[0]
        self.n_mod_pix_total = self.mod_wave.shape[1]
        if self.include_model_errs:
            self.mod_errs = torch.vstack([ensure_tensor(model.mod_errs) for model in self.models]).flatten().unsqueeze(0)
        else:
            self.mod_errs = None
        self.model_res = model_res
        self.labels = self.models[0].labels
        self.stellar_labels_min = ensure_tensor(list(self.models[0].x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(self.models[0].x_max.values()))
        self.n_stellar_labels = self.models[0].input_dim

        self.vmacro_method = vmacro_method
        if vmacro_method == 'iso':
            self.vmacro_broaden = self.vmacro_iso_broaden
        elif vmacro_method == 'iso_fft':
            self.vmacro_broaden = self.vmacro_iso_broaden_fft
        elif vmacro_method == 'rt_fft':
            self.vmacro_broaden = self.vmacro_rt_broaden_fft
        else:
            raise ValueError("vmacro_method must be one of 'iso', 'iso_fft', or 'rt_fft'")

        self.rv_scale = rv_scale
        self.cont_deg = cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = cont_wave_norm_range

        if obs_wave is not None:
            self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        else:
            self.obs_wave = ensure_tensor(self.mod_wave.view(1, -1), precision=torch.float64)
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

        if obs_blaz is not None:
            self.obs_blaz = ensure_tensor(obs_blaz)
        else:
            self.obs_blaz = torch.ones_like(self.obs_wave)

        self.n_mod_pix = self.mod_wave.shape[1]
        self.n_obs_ord = self.obs_wave.shape[0]
        self.n_obs_pix = self.obs_wave.shape[1]

    def set_obs_wave(self, obs_wave):
        self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        scale_wave_output = self.scale_wave(self.obs_wave.to(torch.float32))
        self.obs_norm_wave, self.obs_norm_wave_offset, self.obs_norm_wave_scale = scale_wave_output
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.n_cont_coeffs)], dim=0)

    @staticmethod
    def inst_vmacro_iso_broaden_fft(wave, flux, errs, inst_res, vmacro, model_res=None):
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        n_pix = flux.shape[2]
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[..., :-1], axis=-1).values
        eff_wave = torch.median(wave, axis=-1).values
        freq = torch.vstack([torch.fft.rfftfreq(n_pix, d) for d in dv]).to(torch.float64)
        sigma_out = 2.99792458e5 / (inst_res * 2.355)
        sigma_in = 0.0 if model_res is None else 2.99792458e5 / (model_res * 2.355)
        sigma_inst = torch.sqrt(sigma_out ** 2 - sigma_in ** 2).repeat_interleave(n_ords * freq.shape[-1]).view(
            n_spec * n_ords, -1)
        sigma_vmacro = vmacro.repeat_interleave(n_ords * freq.shape[-1]).view(n_spec * n_ords, -1)
        sigma_tot = torch.sqrt(sigma_inst ** 2 + sigma_vmacro ** 2)
        sigma_freq = sigma_tot * freq.repeat(n_spec, 1)
        kernel = torch.exp(-2 * (np.pi * sigma_freq) ** 2)
        flux_ff = torch.fft.rfft(flux).view(n_spec * n_ords, -1)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).view(n_spec * n_ords, -1)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def inst_broaden(wave, flux, errs, inst_res, model_res=None):
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        n_pix = flux.shape[2]
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[..., :-1], axis=-1).values
        eff_wave = torch.median(wave, axis=-1).values
        freq = torch.vstack([torch.fft.rfftfreq(n_pix, d) for d in dv]).to(torch.float64)
        sigma_out = 2.99792458e5 / (inst_res * 2.355)
        sigma_in = 0.0 if model_res is None else 2.99792458e5 / (model_res * 2.355)
        sigma = torch.sqrt(sigma_out ** 2 - sigma_in ** 2)
        sigma_freq = sigma.repeat_interleave(n_ords * freq.shape[-1]).view(n_spec * n_ords, -1) * freq.repeat(n_spec, 1)
        kernel = torch.exp(-2 * (np.pi * sigma_freq) ** 2)
        flux_ff = torch.fft.rfft(flux).view(n_spec * n_ords, -1)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).view(n_spec * n_ords, -1)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def vmacro_iso_broaden(wave, flux, errs, vmacro, ks: int = 21):
        wave = wave.to(torch.float32)
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        d_wave = wave[..., 1] - wave[..., 0]
        eff_wave = torch.median(wave, axis=-1).values
        good_ks = False
        while not good_ks:
            loc_steps = (torch.arange(ks) - (ks - 1) // 2).repeat_interleave(n_spec * n_ords).view(ks, n_spec * n_ords)
            loc = loc_steps * d_wave.repeat(ks, n_spec, 1).view(ks, n_spec * n_ords)
            scale = vmacro.repeat_interleave(n_ords).view(n_spec * n_ords) / 3e5 * eff_wave.repeat(n_spec).view(
                n_spec * n_ords)
            norm = torch.distributions.normal.Normal(
                loc=torch.zeros(ks, n_spec * n_ords),
                scale=scale.repeat(ks, 1).view(ks, n_spec * n_ords)
            )
            kernel = norm.log_prob(loc).exp()
            kernel = kernel / kernel.sum(axis=0)
            if torch.any(kernel[0, :] > 1e-3):
                ks += 10
            else:
                good_ks = True
        conv_spec = torch.nn.functional.conv1d(
            input=flux.view(1, n_spec * n_ords, -1),
            weight=kernel.T.view(n_spec * n_ords, 1, -1),
            padding=ks // 2,
            groups=n_spec * n_ords,
        ).view(n_spec, n_ords, -1)
        if errs is not None:
            conv_errs = torch.nn.functional.conv1d(
                input=errs.view(1, n_spec * n_ords, -1),
                weight=kernel.T.view(n_spec * n_ords, 1, -1),
                padding=ks // 2,
                groups=n_spec * n_ords,
            ).view(n_spec, n_ords, -1)
        else:
            conv_errs = None
        return conv_spec, conv_errs

    @staticmethod
    def vmacro_iso_broaden_fft(wave, flux, errs, vmacro):
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        n_pix = flux.shape[2]
        dv = 2.99792458e5 * torch.median(torch.diff(wave) / wave[..., :-1], axis=-1).values
        eff_wave = torch.median(wave, axis=-1).values
        freq = torch.vstack([torch.fft.rfftfreq(n_pix, d) for d in dv]).to(torch.float64)
        sigma_freq = vmacro.repeat_interleave(n_ords * freq.shape[-1]).view(n_spec * n_ords, -1) \
                     * freq.repeat(n_spec, 1)
        kernel = torch.exp(-2 * (np.pi * sigma_freq) ** 2)
        flux_ff = torch.fft.rfft(flux).view(n_spec * n_ords, -1)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).view(n_spec * n_ords, -1)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def vmacro_rt_broaden_fft(wave, flux, errs, vmacro):
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        n_pix = flux.shape[2]
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[..., :-1], axis=-1).values
        eff_wave = torch.median(wave, axis=-1).values
        freq = torch.vstack([torch.fft.rfftfreq(n_pix, d) for d in dv]).to(torch.float64)
        sigma_freq = vmacro.repeat_interleave(n_ords * freq.shape[-1]).view(n_spec * n_ords, -1) * freq.repeat(n_spec,
                                                                                                               1)
        kernel = (1 - torch.exp(-1 * (np.pi * sigma_freq) ** 2)) / (np.pi * sigma_freq) ** 2
        kernel[:, 0] = 1.0
        flux_ff = torch.fft.rfft(flux).view(n_spec * n_ords, -1)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).view(n_spec * n_ords, -1)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def rot_broaden(wave, flux, errs, vsini):
        n_spec = flux.shape[0]
        n_ords = flux.shape[1]
        n_pix = flux.shape[2]
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[..., :-1], axis=-1).values
        freq = torch.vstack([torch.fft.rfftfreq(n_pix, d) for d in dv]).to(torch.float64)
        ub = 2.0 * np.pi * vsini.repeat_interleave(n_ords * (freq[:, 1:].shape[-1])).view(n_spec * n_ords, -1) \
             * freq[:, 1:].repeat(n_spec, 1)
        j1_term = j_nu(ub, 1, n_tau=64) / ub
        cos_term = 3.0 * torch.cos(ub) / (2 * ub ** 2)
        sin_term = 3.0 * torch.sin(ub) / (2 * ub ** 3)
        sb = j1_term - cos_term + sin_term
        # Clean up rounding errors at low frequency; Should be safe for vsini > 0.1 km/s
        low_freq_idx = (freq[:, 1:].repeat(n_spec, 1).T < freq[:, 1:].repeat(n_spec, 1)[
            np.arange(n_spec * n_ords), torch.argmax(sb, dim=-1)]).T
        sb[low_freq_idx] = 1.0
        kernel = torch.hstack([torch.ones(n_spec * n_ords, 1), sb])
        flux_ff = torch.fft.rfft(flux).view(n_spec * n_ords, -1)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        if errs is not None:
            errs_ff = torch.fft.rfft(errs).view(n_spec * n_ords, -1)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=n_pix).view(n_spec, n_ords, n_pix)
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def get_doppler_wave(wave, rv):
        n_spec = rv.shape[0]
        n_ords = wave.shape[0]
        n_pix = wave.shape[1]
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((c - rv) / (c + rv))
        shifted_wave = doppler_factor.repeat_interleave(n_ords * n_pix).view(n_spec * n_ords, -1) \
                       * wave.repeat(n_spec, 1, 1).view(n_spec * n_ords, -1)
        return shifted_wave.view(n_spec, n_ords, n_pix)

    @staticmethod
    def interp(x_old, x_new, y, fill):
        n_spec = x_old.shape[0]
        n_ords = x_old.shape[1]
        n_pix_old = x_old.shape[2]
        n_pix_new = x_new.shape[1]
        n_ord_new = x_new.shape[0]
        x_old_ = x_old.view(n_spec * n_ords, -1)
        x_new_ = x_new.repeat(n_spec, 1, 1).view(n_spec * n_ords, -1)
        y_ = y.view(n_spec * n_ords, -1)
        out_of_bounds = (x_new_.T < x_old_[:, 0].T).T \
                        | (x_new_.T > x_old_[:, -1].T).T
        x_new_indices = torch.searchsorted(x_old_, x_new_)
        x_new_indices = x_new_indices.clamp(1, x_old.shape[-1] - 1)
        lo = x_new_indices - 1
        hi = x_new_indices
        dim1_idx = torch.arange(n_spec * n_ords).repeat_interleave(n_pix_new * n_ord_new).view(n_spec * n_ords, -1)
        x_lo = x_old_[dim1_idx, lo]
        x_hi = x_old_[dim1_idx, hi]
        y_lo = y_[dim1_idx, lo]
        y_hi = y_[dim1_idx, hi]
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        y_new = slope * (x_new_ - x_lo) + y_lo
        y_new[out_of_bounds] = fill
        return y_new.view(n_spec, n_ord_new, n_pix_new)

    def interp_flux(self, old_wave, new_wave, flux, errs):
        if (flux.shape[0] == 1) and (old_wave.shape[0] > 1):
            flux = flux.repeat(old_wave.shape[0], 1, 1)
        elif (flux.shape[0] != old_wave.shape[0]):
            raise RuntimeError("RV input and stellar label input have conflicting shapes")
        intp_flux = self.interp(
            x_old=old_wave,
            x_new=new_wave,
            y=flux,
            fill=1.0,
        )
        if errs is not None:
            if (errs.shape[0] == 1) and (old_wave.shape[0] > 1):
                errs = errs.repeat(old_wave.shape[0], 1, 1)
            elif (errs.shape[0] != old_wave.shape[0]):
                raise RuntimeError("RV input and stellar label input have conflicting shapes")
            intp_errs = self.interp(
                x_old=old_wave,
                x_new=new_wave,
                y=errs,
                fill=1.0,
            )
        else:
            intp_errs = None
        return intp_flux, intp_errs

    def scale_stellar_labels(self, unscaled_labels):
        x_min = self.stellar_labels_min
        x_max = self.stellar_labels_max
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = self.stellar_labels_min
        x_max = self.stellar_labels_max
        return (scaled_labels + 0.5) * (x_max - x_min) + x_min

    def forward_model_spec(self, norm_flux, norm_errs, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        raise NotImplementedError

    def forward(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None, skip_cont=False,):
        n_spec = stellar_labels.shape[0]
        norm_flux = torch.zeros(n_spec, self.n_models, self.n_mod_pix_indiv)
        # Model Spectrum
        for i, model in enumerate(self.models):
            norm_flux[:, i, :] = model(stellar_labels)
        norm_flux = norm_flux.flatten().view(n_spec, -1).unsqueeze(1)
        norm_errs = self.mod_errs.repeat(n_spec, 1, 1) if self.include_model_errs else None
        if (inst_res is not None) and (vmacro is not None) and (self.vmacro_method == 'iso_fft'):
            conv_flux, conv_errs = self.inst_vmacro_iso_broaden_fft(
                wave=self.mod_wave,
                flux=norm_flux,
                errs=norm_errs,
                inst_res=inst_res,
                vmacro=vmacro,
                model_res=self.model_res,
            )
        else:
            # Instrumental Broadening
            if inst_res is not None:
                conv_flux, conv_errs = self.inst_broaden(
                    wave=self.mod_wave,
                    flux=norm_flux,
                    errs=norm_errs,
                    inst_res=inst_res,
                    model_res=self.model_res,
                )
            else:
                conv_flux, conv_errs = norm_flux, norm_errs
            # Macroturbulent Broadening
            if vmacro is not None:
                conv_flux, conv_errs = self.vmacro_broaden(
                    wave=self.mod_wave,
                    flux=conv_flux,
                    errs=conv_errs,
                    vmacro=vmacro,
                )
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
                wave=self.mod_wave,
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # Doppler Shift
        doppler_wave = self.get_doppler_wave(
            wave=self.mod_wave,
            rv=rv * self.rv_scale,
        )
        # Interpolation to Observed Wavelength
        intp_flux, intp_errs = self.interp_flux(
            old_wave=doppler_wave,
            new_wave=self.obs_wave,
            flux=conv_flux,
            errs=conv_errs,
        )
        # Continuum Correction
        if skip_cont:
            cont_flux = 1
        else:
            cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_) * self.obs_blaz
        if self.include_model_errs:
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)


class PayneEmulatorMulti(torch.nn.Module):
    def __init__(
            self,
            emulators,
    ):
        super(PayneEmulatorMulti, self).__init__()
        self.emulators = emulators
        self.n_emulators = len(emulators)
        self.n_obs_ord = self.emulators[0].n_obs_ord
        self.n_obs_pix = self.emulators[0].n_obs_pix

        self.labels = self.emulators[0].labels
        self.n_stellar_labels = self.emulators[0].n_stellar_labels

        self.stellar_labels_min = self.emulators[0].stellar_labels_min
        self.stellar_labels_max = self.emulators[0].stellar_labels_max
        self.scale_stellar_labels = self.emulators[0].scale_stellar_labels
        self.unscale_stellar_labels = self.emulators[0].unscale_stellar_labels

        self.rv_scale = self.emulators[0].rv_scale

        self.cont_deg = self.emulators[0].cont_deg
        self.n_cont_coeffs = self.emulators[0].n_cont_coeffs
        self.cont_wave_norm_range = self.emulators[0].cont_wave_norm_range

    def calc_cont(self, cont_coeffs):
        self.out_shape = (self.n_emulators, self.n_obs_ord, self.n_obs_pix)
        cont = torch.zeros(self.out_shape)
        for i, emulator in enumerate(self.emulators):
            cont[i, :, :] = emulator.calc_cont(cont_coeffs[i], emulator.obs_wave_)
        return cont

    def forward(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None, skip_cont=False):
        n_spec = stellar_labels.shape[0]
        self.out_shape = (n_spec, self.n_emulators, self.n_obs_ord, self.n_obs_pix)
        mod_flux = torch.zeros(self.out_shape)
        mod_errs = torch.zeros(self.out_shape)
        for i, emulator in enumerate(self.emulators):
            mod_flux[:, i, :, :], mod_errs[:, i, :, :] = emulator.forward(stellar_labels, rv, vmacro, cont_coeffs[i],
                                                                          inst_res, vsini, skip_cont)
        return mod_flux, mod_errs


class PayneOptimizer:
    def __init__(
            self,
            emulator,
            loss_fn,
            learning_rates,
            learning_rate_decay,
            learning_rate_decay_ts,
            tolerances,
    ):
        self.emulator = emulator
        self.loss_fn = loss_fn
        self.learning_rates = learning_rates
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_ts = learning_rate_decay_ts
        self.tolerances = tolerances

        self.n_stellar_labels = self.emulator.n_stellar_labels

        self.cont_deg = self.emulator.cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = self.emulator.cont_wave_norm_range

    def init_values(self, plot_prefits=False):
        if self.init_params['f_out'] == 'prefit':
            if self.params['f_out'] == 'fit':
                self.f_out = self.prefit_f_out(plot=plot_prefits).requires_grad_()
            else:
                self.f_out = self.prefit_f_out(plot=plot_prefits)
        else:
            if self.params['f_out'] == 'f_out':
                self.f_out = self.init_params['f_out'].requires_grad_() if self.init_params[
                                                                                        'f_out'] is not None else None
            else:
                self.f_out = self.init_params['f_out'] if self.init_params['f_out'] is not None else None
        if self.init_params['inst_res'] == 'prefit':
            if self.params['inst_res'] == 'fit':
                self.inst_res = self.prefit_inst_res(plot=plot_prefits).requires_grad_()
            else:
                self.inst_res = self.prefit_inst_res(plot=plot_prefits)
        else:
            if self.params['inst_res'] == 'fit':
                self.inst_res = self.init_params['inst_res'].requires_grad_() if self.init_params[
                                                                                     'inst_res'] is not None else None
            else:
                self.inst_res = self.init_params['inst_res'] if self.init_params['inst_res'] is not None else None
        if self.init_params['rv'] == 'prefit':
            if self.params['rv'] == 'fit':
                self.rv = self.prefit_rv(plot=plot_prefits).requires_grad_()
            else:
                self.rv = self.prefit_rv(plot=plot_prefits)
        else:
            if self.params['rv'] == 'fit':
                self.rv = self.init_params['rv'].requires_grad_()
            else:
                self.rv = self.init_params['rv']
        if self.init_params['cont_coeffs'] == 'prefit':
            if self.params['cont_coeffs'] == 'fit':
                self.cont_coeffs = [coeffs.requires_grad_() for coeffs in self.prefit_cont(plot=plot_prefits)]
            else:
                self.cont_coeffs = [coeffs for coeffs in self.prefit_cont(plot=plot_prefits)]
        else:
            if self.params['cont_coeffs'] == 'fit':
                self.cont_coeffs = [self.c_flat[i].requires_grad_() for i in range(self.n_cont_coeffs)]
            else:
                self.cont_coeffs = [self.c_flat[i] for i in range(self.n_cont_coeffs)]
        if self.init_params['stellar_labels'] == 'prefit':
            if self.params['stellar_labels'] == 'fit':
                self.stellar_labels = self.prefit_stellar_labels(plot=plot_prefits).requires_grad_()
            else:
                self.stellar_labels = self.prefit_stellar_labels(plot=plot_prefits)
        else:
            if self.params['stellar_labels'] == 'fit':
                self.stellar_labels = self.init_params['stellar_labels'].requires_grad_()
            else:
                self.stellar_labels = self.init_params['stellar_labels']
        if self.init_params['log_vmacro'] == 'prefit':
            if self.params['log_vmacro'] == 'fit':
                self.log_vmacro = self.prefit_vmacro(plot=plot_prefits).requires_grad_()
            else:
                self.log_vmacro = self.prefit_vmacro(plot=plot_prefits)
        else:
            if self.params['log_vmacro'] == 'fit':
                self.log_vmacro = self.init_params['log_vmacro'].requires_grad_() if self.init_params[
                                                                                         'log_vmacro'] is not None else None
            else:
                self.vmacro = self.init_params['log_vmacro'] if self.init_params['log_vmacro'] is not None else None
        if self.init_params['log_vsini'] == 'prefit':
            if self.params['log_vsini'] == 'fit':
                self.log_vsini = self.prefit_vsini(plot=plot_prefits).requires_grad_()
            else:
                self.log_vsini = self.prefit_vsini(plot=plot_prefits)
        else:
            if self.params['log_vsini'] == 'fit':
                self.log_vsini = self.init_params['log_vsini'].requires_grad_() if self.init_params[
                                                                                       'log_vsini'] is not None else None
            else:
                self.log_vsini = self.init_params['log_vsini'] if self.init_params['log_vsini'] is not None else None

    def prefit_rv(
            self,
            log_vmacro0=None,
            log_vsini0=None,
            inst_res0=None,
            rv_range=(-5, 5),
            n_rv=501,
            plot=False,
    ):
        self.stellar_labels = torch.zeros(1, self.n_stellar_labels)
        self.log_vmacro = ensure_tensor(log_vmacro0) if log_vmacro0 is not None else None
        self.log_vsini = ensure_tensor(log_vsini0) if log_vsini0 is not None else None
        # self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        rv0 = torch.linspace(rv_range[0], rv_range[1], n_rv)
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=rv0,
            cont_coeffs=self.c_flat,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        if self.loss_fn == 'neg_log_posterior':
            loss0 = self.neg_log_posterior(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        elif self.loss_fn == 'neg_log_posterior_mixture':
            loss0 = self.neg_log_posterior_mixture(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
                f_out=ensure_tensor(0),
            )
        else:
            loss0 = self.loss_fn(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        if plot:
            plt.plot(rv0, loss0.detach().numpy(), c='k')
            plt.axvline(rv0[loss0.argmin()].unsqueeze(0).detach().numpy(), c='r')
            plt.show()
        return rv0[loss0.argmin()].unsqueeze(0)

    def prefit_cont(
            self,
            log_vmacro0=None,
            log_vsini0=None,
            inst_res0=None,
            plot=False,
    ):
        self.stellar_labels = torch.zeros(1, self.n_stellar_labels)
        self.log_vmacro = ensure_tensor(log_vmacro0) if log_vmacro0 is not None else None
        self.log_vsini = ensure_tensor(log_vsini0) if log_vsini0 is not None else None
        # self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            cont_coeffs=self.c_flat,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        c0 = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        footprint = np.concatenate(
            [np.ones(self.prefit_cont_window), np.zeros(self.prefit_cont_window), np.ones(self.prefit_cont_window)])
        tot_errs = torch.sqrt((mod_errs[0] * self.obs_blaz) ** 2 + self.obs_errs ** 2)
        scaling = self.obs_flux / (mod_flux[0] * self.obs_blaz)
        scaling[~self.obs_mask] = 1.0
        for i in range(self.n_obs_ord):
            filtered_scaling = percentile_filter(scaling[i].detach().numpy(), percentile=25, footprint=footprint)
            filtered_scaling = percentile_filter(filtered_scaling, percentile=75, footprint=footprint)
            filtered_scaling[filtered_scaling <= 0.0] = 1.0
            p = Polynomial.fit(
                x=self.obs_norm_wave[i].detach().numpy(),
                y=filtered_scaling,
                deg=self.cont_deg,
                w=(tot_errs[i] ** -1).detach().numpy(),
                window=self.cont_wave_norm_range
            )
            if plot:
                plt.figure(figsize=(20, 1))
                plt.scatter(
                    self.obs_wave[i][self.obs_mask[i]].detach().numpy(),
                    self.obs_flux[i][self.obs_mask[i]].detach().numpy(),
                    c='k', marker='.', alpha=0.8
                )
                plt.plot(
                    self.obs_wave[i].detach().numpy(),
                    (mod_flux[0, i] * self.obs_blaz[i] * p(self.obs_norm_wave[i])).detach().numpy(),
                    c='r', alpha=0.8
                )
                plt.ylim(0, 1.5 * np.quantile(self.obs_flux[i][self.obs_mask[i]].detach().numpy(), 0.95))
                plt.show()
                plt.close('all')
            c0[:, i] = ensure_tensor(p.coef)
            if torch.isnan(c0).any():
                raise RuntimeError("NaN value returned for c0")
        return [c0[i] for i in range(self.n_cont_coeffs)]

    def prefit_vmacro(self, plot=False):
        raise NotImplementedError

    def prefit_vsini(self, plot=False):
        raise NotImplementedError

    def prefit_inst_res(self, plot=False):
        raise NotImplementedError

    def prefit_f_out(self, plot=False):
        raise NotImplementedError

    def prefit_stellar_labels(self, plot=False):
        n_spec = 100
        stellar_labels0 = torch.zeros(n_spec, self.n_stellar_labels)
        # Initialize [Fe/H]
        fe0 = self.priors['stellar_labels']['Fe'].sample(n_spec)
        if self.use_gaia_phot:
            fe_phot = torch.vstack([
                fe0,
                self.gaia_bprp,
                self.gaia_g,
            ]).T
            logg, logTeff = self._gaia_fn(fe_phot.detach().numpy()).flatten()
        for i, label in enumerate(self.emulator.labels):
            if (label == 'Teff') and self.use_gaia_phot:
                x0 = 10 ** logTeff
                #x0 = self.emulator.scale_stellar_labels(
                #    10 ** logTeff * torch.ones(self.n_stellar_labels)
                #)[i]
            elif (label == 'logg') and self.use_gaia_phot:
                x0 = logg
                #x0 = self.emulator.scale_stellar_labels(
                #    logg * torch.ones(self.n_stellar_labels)
                #)[i]
            elif (label == 'v_micro'):  # and not configs['fitting']['fit_vmicro']:
                # Set v_micro using Holtzman2015 scaling
                with torch.no_grad():
                    x0 = self.holtzman2015(stellar_labels0[:, 1])
                    x0 = self.emulator.unscale_stellar_labels(
                        x0.unsqueeze(1) * torch.ones(n_spec, self.n_stellar_labels)
                    )[:, i]
            elif label == 'Fe':
                # Set [Fe/H]
                x0 = fe0
            elif label in self.priors['stellar_labels']:
                x0 = self.priors['stellar_labels'][label].sample(n_spec)
            else:
                raise RuntimeError(f"Prior not defined for {label} --- cannot initialize")
            if label not in ['Teff', 'logg', 'v_micro', 'Fe']:
                # Scale all abundances by [Fe/H]
                x0 += fe0
            # Scale Stellar Labels
            x0 = self.emulator.scale_stellar_labels(
                ensure_tensor(x0).unsqueeze(1) * torch.ones(n_spec, self.n_stellar_labels)
            )[:, i]
            with torch.no_grad():
                stellar_labels0[:, i] = x0
        mod_flux, mod_errs = self.emulator(
            stellar_labels=stellar_labels0,
            rv=self.rv.unsqueeze(0),
            cont_coeffs=self.cont_coeffs,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        if self.loss_fn == 'neg_log_posterior':
            loss0 = self.neg_log_posterior(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        elif self.loss_fn == 'neg_log_posterior_mixture':
            loss0 = self.neg_log_posterior_mixture(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
                f_out=self.f_out,
            )
        else:
            loss0 = self.loss_fn(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
            if plot == True:
                print('Plotting not implemented for prefit_stellar_labels')
        return stellar_labels0[loss0.argmin()].unsqueeze(0)

    def init_optimizer_scheduler(self):
        optim_params = []
        lr_lambdas = []
        if self.params['stellar_labels'] == 'fit':
            optim_params.append({'params': [self.stellar_labels], 'lr': self.learning_rates['stellar_labels']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['stellar_labels'] ** (
                    epoch // self.learning_rate_decay_ts['stellar_labels']))
        if self.params['rv'] == 'fit':
            optim_params.append({'params': [self.rv], 'lr': self.learning_rates['rv']})
            lr_lambdas.append(
                lambda epoch: self.learning_rate_decay['rv'] ** (epoch // self.learning_rate_decay_ts['rv']))
        if self.params['log_vmacro'] == 'fit':
            optim_params.append({'params': [self.log_vmacro], 'lr': self.learning_rates['log_vmacro']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['log_vmacro'] ** (
                    epoch // self.learning_rate_decay_ts['log_vmacro']))
        if self.params['log_vsini'] == 'fit':
            optim_params.append({'params': [self.log_vsini], 'lr': self.learning_rates['log_vsini']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['log_vsini'] ** (
                    epoch // self.learning_rate_decay_ts['log_vsini']))
        if self.params['inst_res'] == 'fit':
            optim_params.append({'params': [self.inst_res], 'lr': self.learning_rates['inst_res']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['inst_res'] ** (
                    epoch // self.learning_rate_decay_ts['inst_res']))
        if self.params['f_out'] == 'fit':
            optim_params.append({'params': [self.f_out], 'lr': self.learning_rates['f_out']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['f_out'] ** (
                    epoch // self.learning_rate_decay_ts['f_out']))
        if self.params['cont_coeffs'] == 'fit':
            optim_params += [
                {'params': [self.cont_coeffs[i]],
                 'lr': torch.abs(self.cont_coeffs[i].mean()) * self.learning_rates['cont_coeffs']}
                for i in range(self.n_cont_coeffs)
            ]
            lr_lambdas += [
                lambda epoch: self.learning_rate_decay['cont_coeffs'] ** (
                        epoch // self.learning_rate_decay_ts['cont_coeffs'])
                for i in range(self.n_cont_coeffs)
            ]
        optimizer = torch.optim.Adam(optim_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambdas
        )
        return optimizer, scheduler

    def atm_from_gaia_phot(self, scaled_feh):
        n_spec = scaled_feh.shape[0]
        teff_min = self.emulator.stellar_labels_min[0]
        teff_max = self.emulator.stellar_labels_max[0]
        logg_min = self.emulator.stellar_labels_min[1]
        logg_max = self.emulator.stellar_labels_max[1]
        fe_min = self.emulator.stellar_labels_min[self.fe_idx]
        fe_max = self.emulator.stellar_labels_max[self.fe_idx]
        unscaled_fe = (scaled_feh + 0.5) * (fe_max - fe_min) + fe_min
        vec = torch.zeros(
            (self._gaia_n_cmd + self._gaia_n_pow, n_spec),
            dtype=torch.float64
        )
        feh_phot = torch.vstack(
            [unscaled_fe, self.gaia_bprp.repeat(n_spec), self.gaia_g.repeat(n_spec)]
        ).T * self._gaia_eps
        feh_phot_hat = (feh_phot - self._gaia_shift) / self._gaia_scale
        d = feh_phot.unsqueeze(0) - self._gaia_cmd.unsqueeze(1)
        r = torch.linalg.norm(d, axis=2)
        vec[:self._gaia_n_cmd, :] = thin_plate_spline(r)
        feh_phot_hat_powers = feh_phot_hat.unsqueeze(0) ** self._gaia_pow.unsqueeze(1)
        vec[self._gaia_n_cmd:, :] = torch.prod(feh_phot_hat_powers, axis=2)
        logg_logteff = torch.tensordot(self._gaia_coeffs, vec, dims=[[0], [0]])
        scaled_logg = (logg_logteff[0] - logg_min) / (logg_max - logg_min) - 0.5
        scaled_teff = (10 ** logg_logteff[1] - teff_min) / (teff_max - teff_min) - 0.5
        scaled_logg_teff = torch.vstack([scaled_teff, scaled_logg]).T
        return scaled_logg_teff

    def holtzman2015(self, scaled_logg):
        logg_min = self.emulator.stellar_labels_min[1]
        logg_max = self.emulator.stellar_labels_max[1]
        vmicro_min = self.emulator.stellar_labels_min[2]
        vmicro_max = self.emulator.stellar_labels_max[2]
        unscaled_logg = (scaled_logg + 0.5) * (logg_max - logg_min) + logg_min
        unscaled_vmicro = 2.478 - 0.325 * unscaled_logg
        scaled_vmicro = (unscaled_vmicro - vmicro_min) / (vmicro_max - vmicro_min) - 0.5
        return scaled_vmicro

    def log_priors(self, log_likelihood):
        log_priors = torch.zeros_like(log_likelihood)
        unscaled_stellar_labels = self.emulator.unscale_stellar_labels(self.stellar_labels)
        unscaled_stellar_labels = unscaled_stellar_labels - self.fe_scaler[0] * unscaled_stellar_labels[:, self.fe_idx]
        for i, label in enumerate(self.emulator.labels):
            log_priors += self.priors['stellar_labels'][label](unscaled_stellar_labels[:, i])
        if self.log_vmacro is not None:
            log_priors += self.priors['log_vmacro'](self.log_vmacro[:, 0])
        if self.log_vsini is not None:
            log_priors += self.priors['log_vsini'](self.log_vsini[:, 0])
        if self.inst_res is not None:
            log_priors += self.priors['inst_res'](self.inst_res[:, 0])
        return log_priors

    def neg_log_posterior(self, pred, target, pred_errs, target_errs):
        log_likelihood = gaussian_log_likelihood(pred, target, pred_errs, target_errs)
        log_priors = self.log_priors(log_likelihood)
        return -1 * (log_likelihood + log_priors)

    def neg_log_posterior_mixture(self, pred, target, pred_errs, target_errs, f_out, out_err_scale=50):
        log_likelihood_in = torch.log(1 - f_out) + gaussian_log_likelihood(
            pred, target, pred_errs, target_errs
        )
        log_likelihood_out = torch.log(f_out) + gaussian_log_likelihood(
            pred, target, out_err_scale * pred_errs, out_err_scale * target_errs
        )
        log_likelihood = torch.logaddexp(log_likelihood_in, log_likelihood_out)
        log_priors = self.log_priors(log_likelihood)
        return -1 * (log_likelihood + log_priors)

    def forward(self):
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            cont_coeffs=torch.stack(self.cont_coeffs),
            inst_res=self.inst_res,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
        )
        return mod_flux * self.obs_blaz, mod_errs * self.obs_blaz

    def fit(
            self,
            obs_flux,
            obs_errs,
            obs_wave,
            obs_blaz=None,
            obs_mask=None,
            params=dict(
                stellar_labels='fit',
                rv='fit',
                vmacro='const',
                vsini='const',
                inst_res='const',
                cont_coeffs='fit',
                f_out='const',
            ),
            init_params=dict(
                stellar_labels=torch.zeros(1, 12),
                rv='prefit',
                vmacro=None,
                vsini=None,
                inst_res=None,
                cont_coeffs='prefit',
                f_out=None,
            ),
            priors=None,
            min_epochs=1000,
            max_epochs=10000,
            prefit_cont_window=55,
            use_holtzman2015=False,
            use_gaia_phot=False,
            gaia_g=None,
            gaia_bprp=None,
            gaia_cmd_interpolator=None,
            verbose=False,
            print_update_every=20,
            plot_prefits=False,
            plot_fit_every=None,
    ):
        self.obs_flux = ensure_tensor(obs_flux)
        self.obs_errs = ensure_tensor(obs_errs)
        self.obs_snr = self.obs_flux / self.obs_errs
        self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        self.obs_blaz = ensure_tensor(obs_blaz) if obs_blaz is not None else torch.ones_like(self.obs_flux)
        self.obs_mask = ensure_tensor(obs_mask, precision=bool) \
            if obs_mask is not None \
            else torch.ones_like(self.obs_flux, dtype=bool)
        if torch.all(self.obs_wave != self.emulator.obs_wave):
            raise RuntimeError("obs_wave of Emulator and Optimizer differ!")
        self.obs_norm_wave = self.emulator.obs_norm_wave
        self.obs_wave_ = self.emulator.obs_wave_
        self.n_obs_ord = self.obs_wave.shape[0]
        self.n_obs_pix_per_ord = self.obs_wave.shape[0]

        self.c_flat = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        self.c_flat[0] = 1.0
        self.prefit_cont_window = prefit_cont_window

        self.params = params
        self.init_params = init_params

        self.priors = priors
        self.fe_idx = self.emulator.labels.index('Fe')
        self.fe_scaled_idx = [
            i for i, label in enumerate(self.emulator.labels)
            if label not in ['Teff', 'logg', 'v_micro', 'Fe']
        ]
        self.fe_scaler = torch.zeros(2, self.emulator.n_stellar_labels)
        self.fe_scaler[:, self.fe_scaled_idx] = 1
        self.stellar_label_bounds = torch.Tensor([
            [prior.lower_bound, prior.upper_bound]
            for label, prior in self.priors['stellar_labels'].items()
        ]).T
        self.use_holtzman2015 = use_holtzman2015
        self.use_gaia_phot = use_gaia_phot
        if self.use_gaia_phot:
            if gaia_g is None or gaia_bprp is None:
                raise ValueError("Gaia photometry is not provided")
            if gaia_cmd_interpolator is None:
                raise ValueError("Gaia CMD interpolator is not provided")
            self.gaia_g = ensure_tensor(gaia_g)
            self.gaia_bprp = ensure_tensor(gaia_bprp)
            self._gaia_fn = gaia_cmd_interpolator
            self._gaia_cmd = ensure_tensor(self._gaia_fn.y * self._gaia_fn.epsilon, precision=torch.float64)
            self._gaia_eps = ensure_tensor(self._gaia_fn.epsilon, precision=torch.float64)
            self._gaia_pow = ensure_tensor(self._gaia_fn.powers, precision=torch.float64)
            self._gaia_coeffs = ensure_tensor(self._gaia_fn._coeffs, precision=torch.float64)
            self._gaia_shift = ensure_tensor(self._gaia_fn._shift, precision=torch.float64)
            self._gaia_scale = ensure_tensor(self._gaia_fn._scale, precision=torch.float64)
            self._gaia_n_cmd = self._gaia_cmd.shape[0]
            self._gaia_n_pow = self._gaia_pow.shape[0]
            self._gaia_n_coeffs = self._gaia_coeffs.shape[1]

        # Initialize Starting Values
        self.init_values(plot_prefits=plot_prefits)

        # Initialize Optimizer & Learning Rate Scheduler
        optimizer, scheduler = self.init_optimizer_scheduler()

        # Initialize Convergence Criteria
        self.epoch = 0
        self.loss = ensure_tensor(np.inf)
        delta_loss = ensure_tensor(np.inf)
        delta_stellar_labels = ensure_tensor(np.inf)
        delta_log_vmacro = ensure_tensor(np.inf)
        delta_rv = ensure_tensor(np.inf)
        delta_inst_res = ensure_tensor(np.inf)
        delta_log_vsini = ensure_tensor(np.inf)
        delta_f_out = ensure_tensor(np.inf)
        delta_frac_weighted_cont = ensure_tensor(np.inf)
        last_cont = torch.zeros_like(self.obs_blaz)

        # Initialize History
        self.history = dict(
            stellar_labels=[],
            log_vmacro=[],
            rv=[],
            inst_res=[],
            log_vsini=[],
            f_out=[],
            cont_coeffs=[],
            loss=[]
        )

        while (
                (self.epoch < min_epochs)
                or (
                        (self.epoch < max_epochs)
                        and (
                                (delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'])
                                or (delta_frac_weighted_cont.abs().max() > self.tolerances['d_cont'])
                                or (delta_log_vmacro.abs() > self.tolerances['d_log_vmacro'])
                                or (delta_rv.abs() > self.tolerances['d_rv'])
                                or (delta_inst_res.abs() > self.tolerances['d_inst_res'])
                                or (delta_log_vsini.abs() > self.tolerances['d_log_vsini'])
                                or (delta_f_out.abs() > self.tolerances['d_f_out'])
                        )
                        and (
                                delta_loss.abs() > self.tolerances['d_loss']
                        )
                        and (
                                self.loss > self.tolerances['loss']
                        )
                )
        ):
            # Forward Pass
            mod_flux, mod_errs = self.forward()
            if self.loss_fn == 'neg_log_posterior':
                loss_epoch = self.neg_log_posterior(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                )
            elif self.loss_fn == 'neg_log_posterior_mixture':
                loss_epoch = self.neg_log_posterior_mixture(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                    f_out=self.f_out,
                )
            else:
                loss_epoch = self.loss_fn(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                )
            if torch.isnan(loss_epoch):
                raise RuntimeError('NaN value returned for loss')

            # Log Results / History
            delta_loss = self.loss - loss_epoch
            self.loss = loss_epoch.item()
            self.history['stellar_labels'].append(torch.clone(self.stellar_labels).detach())
            self.history['rv'].append(torch.clone(self.rv))
            if self.log_vmacro is None:
                self.history['log_vmacro'].append(None)
            else:
                self.history['log_vmacro'].append(torch.clone(self.log_vmacro))
            if self.inst_res is None:
                self.history['inst_res'].append(None)
            else:
                self.history['inst_res'].append(torch.clone(self.inst_res))
            if self.log_vsini is None:
                self.history['log_vsini'].append(None)
            else:
                self.history['log_vsini'].append(torch.clone(self.log_vsini))
            if self.f_out is None:
                self.history['f_out'].append(None)
            else:
                self.history['f_out'].append(torch.clone(self.f_out))
            self.history['cont_coeffs'].append(torch.clone(torch.stack(self.cont_coeffs)).detach())
            self.history['loss'].append(self.loss)

            # Backward Pass
            optimizer.zero_grad()
            loss_epoch.backward()
            optimizer.step()
            scheduler.step()
            if torch.isnan(self.stellar_labels).any():
                raise RuntimeError('NaN value(s) suggested for stellar_labels')
            if torch.isnan(self.rv).any():
                raise RuntimeError('NaN value(s) suggested for rv')
            if torch.isnan(torch.stack(self.cont_coeffs)).any():
                raise RuntimeError('NaN value(s) suggested for cont_coeffs')

            # Set Bounds
            with torch.no_grad():
                # Enforce Stellar Label Priors
                unscaled_stellar_labels = self.emulator.unscale_stellar_labels(self.stellar_labels)
                scaled_stellar_bounds = self.emulator.scale_stellar_labels(
                    self.stellar_label_bounds + self.fe_scaler * unscaled_stellar_labels[:, self.fe_idx]
                )
                for i, label in enumerate(self.emulator.labels):
                    self.stellar_labels[:, i].clamp_(
                        min=torch.max(scaled_stellar_bounds[0, i], ensure_tensor(-0.5)).item(),
                        max=torch.min(scaled_stellar_bounds[1, i], ensure_tensor(0.5)).item(),
                    )
                if self.use_gaia_phot:
                    self.stellar_labels[:, :2] = self.atm_from_gaia_phot(self.stellar_labels[:, self.fe_idx])
                if self.use_holtzman2015:
                    self.stellar_labels[:, 2] = self.holtzman2015(self.stellar_labels[:, 1])
                if self.log_vmacro is not None:
                    self.log_vmacro.clamp_(
                        min=self.priors['log_vmacro'].lower_bound,
                        max=self.priors['log_vmacro'].upper_bound,
                    )
                if self.log_vsini is not None:
                    self.log_vsini.clamp_(
                        min=self.priors['log_vsini'].lower_bound,
                        max=self.priors['log_vsini'].upper_bound,
                    )
                if self.inst_res is not None:
                    self.inst_res.clamp_(
                        min=self.priors['inst_res'].lower_bound,
                        max=self.priors['inst_res'].upper_bound,
                    )
                if self.f_out is not None:
                    self.f_out.clamp_(
                        min=self.priors['f_out'].lower_bound,
                        max=self.priors['f_out'].upper_bound,
                    )

            # Check Convergence
            delta_stellar_labels = self.stellar_labels - self.history['stellar_labels'][-1]
            delta_log_vmacro = ensure_tensor(0) if self.log_vmacro is None else self.log_vmacro - \
                                                                                self.history['log_vmacro'][-1]
            delta_rv = self.rv - self.history['rv'][-1]
            delta_inst_res = ensure_tensor(0) if self.inst_res is None else self.inst_res - self.history['inst_res'][-1]
            delta_log_vsini = ensure_tensor(0) if self.log_vsini is None else self.log_vsini - \
                                                                              self.history['log_vsini'][-1]
            delta_f_out = ensure_tensor(0) if self.f_out is None else self.f_out - self.history['f_out'][-1]
            delta_cont_coeffs = torch.stack(self.cont_coeffs) - self.history['cont_coeffs'][-1]
            if self.epoch % print_update_every == 0:
                cont = self.emulator.calc_cont(
                    torch.stack(self.cont_coeffs),
                    self.obs_wave_
                )
                delta_cont = cont - last_cont
                delta_frac_weighted_cont = delta_cont / cont * self.obs_snr
                last_cont = cont
                if verbose:
                    print(f"Epoch: {self.epoch}, Current Loss: {self.loss:.6f}")
            if (plot_fit_every is not None) and (self.epoch % plot_fit_every == 0):
                tot_errs = torch.sqrt(mod_errs ** 2 + self.obs_errs ** 2)
                mse = ((mod_flux[0] - self.obs_flux) / tot_errs) ** 2
                mask = torch.isfinite(tot_errs)
                for i in range(self.n_obs_ord):
                    fig = plt.figure(figsize=(20, 2))
                    gs = GridSpec(3, 1)
                    gs.update(hspace=0.0)
                    ax1 = plt.subplot(gs[:2, 0])
                    ax2 = plt.subplot(gs[2, 0], sharex=ax1)
                    ax1.scatter(self.obs_wave[i][mask[0, i]].detach().numpy(),
                                self.obs_flux[i][mask[0, i]].detach().numpy(), c='k', marker='.', alpha=0.8, )
                    ax1.plot(self.obs_wave[i].detach().numpy(), mod_flux[0, i].detach().numpy(), c='r', alpha=0.8)
                    ax2.scatter(self.obs_wave[i][mask[0, i]].detach().numpy(), mse[0, i][mask[0, i]].detach().numpy(),
                                c='k', marker='.', alpha=0.8)
                    ax1.tick_params('x', labelsize=0)
                    plt.show()
                    plt.close('all')
            self.epoch += 1

        # Recover Best Epoch
        self.stellar_labels = self.history['stellar_labels'][np.argmin(self.history['loss'])]
        self.log_vmacro = self.history['log_vmacro'][np.argmin(self.history['loss'])]
        self.rv = self.history['rv'][np.argmin(self.history['loss'])]
        self.inst_res = self.history['inst_res'][np.argmin(self.history['loss'])]
        self.log_vsini = self.history['log_vsini'][np.argmin(self.history['loss'])]
        self.f_out = self.history['f_out'][np.argmin(self.history['loss'])]
        self.cont_coeffs = [self.history['cont_coeffs'][np.argmin(self.history['loss'])][i] for i in
                            range(self.n_cont_coeffs)]
        self.loss = np.min(self.history['loss'])
        self.best_model, self.best_model_errs = self.forward()

        print(f"Best Epoch: {np.argmin(self.history['loss'])}, Best Loss: {self.loss:.6f}")

        if not self.epoch < max_epochs and verbose:
            print('max_epochs reached')
        if not delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'] and verbose:
            print('d_stellar_labels tolerance reached')
        if not delta_rv.abs().max() > self.tolerances['d_rv']:
            print('d_rv tolerance reached')
        if not delta_log_vmacro.abs().max() > self.tolerances[
            'd_log_vmacro'] and self.log_vmacro is not None and verbose:
            print('d_log_vmacro tolerance reached')
        if not delta_inst_res.abs().max() > self.tolerances['d_inst_res'] and self.inst_res is not None:
            print('d_inst_res tolerance reached')
        if not delta_log_vsini.abs().max() > self.tolerances['d_log_vsini'] and self.log_vsini is not None:
            print('d_log_vsini tolerance reached')
        if not delta_f_out.abs().max() > self.tolerances['d_f_out'] and self.f_out is not None:
            print('d_f_out tolerance reached')
        if not delta_frac_weighted_cont.abs().max() > self.tolerances['d_cont']:
            print('d_cont tolerance reached')
        if not delta_loss.abs() > self.tolerances['d_loss']:
            print('d_loss tolerance reached')
        if not self.loss > self.tolerances['loss']:
            print('loss tolerance reached')


class PayneOptimizerMulti:
    def __init__(
            self,
            emulator,
            loss_fn,
            learning_rates,
            learning_rate_decay,
            learning_rate_decay_ts,
            tolerances,
    ):
        self.emulator = emulator
        self.n_obs = len(self.emulator.emulators)
        self.loss_fn = loss_fn
        self.learning_rates = learning_rates
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_ts = learning_rate_decay_ts
        self.tolerances = tolerances

        self.n_stellar_labels = self.emulator.n_stellar_labels

        self.cont_deg = self.emulator.cont_deg
        self.n_cont_coeffs = self.cont_deg + 1
        self.cont_wave_norm_range = self.emulator.cont_wave_norm_range

    def init_values(self, plot_prefits=False):
        if self.init_params['inst_res'] == 'prefit':
            if self.params['inst_res'] == 'fit':
                self.inst_res = self.prefit_inst_res(plot=plot_prefits).requires_grad_()
            else:
                self.inst_res = self.prefit_inst_res(plot=plot_prefits)
        else:
            if self.params['inst_res'] == 'fit':
                self.inst_res = self.init_params['inst_res'].requires_grad_() if self.init_params[
                                                                                     'inst_res'] is not None else None
            else:
                self.inst_res = self.init_params['inst_res'] if self.init_params['inst_res'] is not None else None
        if self.init_params['rv'] == 'prefit':
            if self.params['rv'] == 'fit':
                self.rv = self.prefit_rv(plot=plot_prefits).requires_grad_()
            else:
                self.rv = self.prefit_rv(plot=plot_prefits)
        else:
            if self.params['rv'] == 'fit':
                self.rv = self.init_params['rv'].requires_grad_()
            else:
                self.rv = self.init_params['rv']
        if self.init_params['cont_coeffs'] == 'prefit':
            if self.params['cont_coeffs'] == 'fit':
                self.cont_coeffs = [[coeffs.requires_grad_() for coeffs in coeffs_obs] for coeffs_obs in self.prefit_cont(plot=plot_prefits)]
                self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])
            else:
                self.cont_coeffs = [[coeffs for coeffs in coeffs_obs] for coeffs_obs in self.prefit_cont(plot=plot_prefits)]
                self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])
        else:
            if self.params['cont_coeffs'] == 'fit':
                self.cont_coeffs = [[self.c_flat[i].requires_grad_() for i in range(self.n_cont_coeffs)] for i in range(self.n_obs)]
                self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])
            else:
                self.cont_coeffs = [[self.c_flat[i] for i in range(self.n_cont_coeffs)] for i in range(self.n_obs)]
                self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])
        if self.init_params['stellar_labels'] == 'prefit':
            if self.params['stellar_labels'] == 'fit':
                self.stellar_labels = self.prefit_stellar_labels(plot=plot_prefits).requires_grad_()
            else:
                self.stellar_labels = self.prefit_stellar_labels(plot=plot_prefits)
        else:
            if self.params['stellar_labels'] == 'fit':
                self.stellar_labels = self.init_params['stellar_labels'].requires_grad_()
            else:
                self.stellar_labels = self.init_params['stellar_labels']
        if self.init_params['log_vmacro'] == 'prefit':
            if self.params['log_vmacro'] == 'fit':
                self.log_vmacro = self.prefit_vmacro(plot=plot_prefits).requires_grad_()
            else:
                self.log_vmacro = self.prefit_vmacro(plot=plot_prefits)
        else:
            if self.params['log_vmacro'] == 'fit':
                self.log_vmacro = self.init_params['log_vmacro'].requires_grad_() if self.init_params[
                                                                                         'log_vmacro'] is not None else None
            else:
                self.vmacro = self.init_params['log_vmacro'] if self.init_params['log_vmacro'] is not None else None
        if self.init_params['log_vsini'] == 'prefit':
            if self.params['log_vsini'] == 'fit':
                self.log_vsini = self.prefit_vsini(plot=plot_prefits).requires_grad_()
            else:
                self.log_vsini = self.prefit_vsini(plot=plot_prefits)
        else:
            if self.params['log_vsini'] == 'fit':
                self.log_vsini = self.init_params['log_vsini'].requires_grad_() if self.init_params[
                                                                                       'log_vsini'] is not None else None
            else:
                self.log_vsini = self.init_params['log_vsini'] if self.init_params['log_vsini'] is not None else None

    def prefit_rv(
            self,
            log_vmacro0=None,
            log_vsini0=None,
            inst_res0=None,
            rv_range=(-3, 3),
            n_rv=301,
            plot=False,
    ):
        self.stellar_labels = torch.zeros(n_rv, self.n_stellar_labels)
        self.log_vmacro = ensure_tensor(log_vmacro0) if log_vmacro0 is not None else None
        self.log_vsini = ensure_tensor(log_vsini0) if log_vsini0 is not None else None
        # self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        rv0 = torch.linspace(rv_range[0], rv_range[1], n_rv)
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=rv0,
            cont_coeffs=[self.c_flat for i in range(self.n_obs)],
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        if self.loss_fn == 'neg_log_posterior':
            loss0 = self.neg_log_posterior(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        else:
            loss0 = self.loss_fn(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        if plot:
            plt.plot(rv0, loss0.detach().numpy(), c='k')
            plt.axvline(rv0[loss0.argmin()].unsqueeze(0).detach().numpy(), c='r')
            plt.show()
        return rv0[loss0.argmin()].unsqueeze(0)

    def prefit_cont(
            self,
            log_vmacro0=None,
            log_vsini0=None,
            inst_res0=None,
            plot=False,
    ):
        self.stellar_labels = torch.zeros(1, self.n_stellar_labels)
        self.log_vmacro = ensure_tensor(log_vmacro0) if log_vmacro0 is not None else None
        self.log_vsini = ensure_tensor(log_vsini0) if log_vsini0 is not None else None
        # self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            cont_coeffs=[self.c_flat for i in range(self.n_obs)],
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        c0 = torch.zeros(self.n_obs, self.n_cont_coeffs, self.n_obs_ord)
        footprint = np.concatenate(
            [np.ones(self.prefit_cont_window), np.zeros(self.prefit_cont_window), np.ones(self.prefit_cont_window)])
        tot_errs = torch.sqrt((mod_errs[0] * self.obs_blaz) ** 2 + self.obs_errs ** 2)
        scaling = self.obs_flux / (mod_flux[0] * self.obs_blaz)
        scaling[~self.obs_mask] = 1.0
        for o in range(self.n_obs):
            for i in range(self.n_obs_ord):
                filtered_scaling = percentile_filter(scaling[o,i].detach().numpy(), percentile=25, footprint=footprint)
                filtered_scaling = percentile_filter(filtered_scaling, percentile=75, footprint=footprint)
                filtered_scaling[filtered_scaling <= 0.0] = 1.0
                p = Polynomial.fit(
                    x=self.obs_norm_wave[o,i].detach().numpy(),
                    y=filtered_scaling,
                    deg=self.cont_deg,
                    w=(tot_errs[o,i] ** -1).detach().numpy(),
                    window=self.cont_wave_norm_range
                )
                if plot:
                    plt.figure(figsize=(20, 1))
                    plt.scatter(
                        self.obs_wave[o,i][self.obs_mask[o,i]].detach().numpy(),
                        self.obs_flux[o,i][self.obs_mask[o,i]].detach().numpy(),
                        c='k', marker='.', alpha=0.8
                    )
                    plt.plot(
                        self.obs_wave[o,i].detach().numpy(),
                        (mod_flux[0, o,i] * self.obs_blaz[o,i] * p(self.obs_norm_wave[o,i])).detach().numpy(),
                        c='r', alpha=0.8
                    )
                    plt.ylim(0, 1.5 * np.quantile(self.obs_flux[o,i][self.obs_mask[o,i]].detach().numpy(), 0.95))
                    plt.show()
                    plt.close('all')
                c0[o, :, i] = ensure_tensor(p.coef)
                if torch.isnan(c0).any():
                    raise RuntimeError("NaN value returned for c0")
        return [[c0[o,i] for i in range(self.n_cont_coeffs)] for o in range(self.n_obs)]

    def prefit_vmacro(self, plot=False):
        raise NotImplementedError

    def prefit_vsini(self, plot=False):
        raise NotImplementedError

    def prefit_inst_res(self, plot=False):
        raise NotImplementedError

    def prefit_stellar_labels(self):
        n_spec = 100
        stellar_labels0 = torch.zeros(100, self.n_stellar_labels)
        if type(self.priors['Fe']) == GaussianLogPrior:
            fe0 = torch.zeros(n_spec).normal_(
                self.priors['Fe'].mu,
                self.priors['Fe'].sigma,
            )
        else:
            fe0 = torch.zeros(n_spec).uniform_(
                self.priors['Fe'].lower_bound,
                self.priors['Fe'].upper_bound,
            )
        if self.use_gaia_phot:
            fe_phot = torch.vstack([
                fe0,
                self.gaia_bprp,
                self.gaia_g,
            ]).T
            logg, logTeff = self._gaia_fn(fe_phot.detach().numpy()).flatten()
        for i, label in enumerate(self.emulator.labels):
            if (label == 'Teff') and self.use_gaia_phot:
                x0 = self.emulator.scale_stellar_labels(
                    10 ** logTeff * torch.ones(self.n_stellar_labels)
                )[i]
            elif (label == 'logg') and self.use_gaia_phot:
                x0 = self.emulator.scale_stellar_labels(
                    logg * torch.ones(self.n_stellar_labels)
                )[i]
            elif label in self.priors:
                if type(self.priors[label]) == GaussianLogPrior:
                    x0 = torch.zeros(n_spec).normal_(
                        self.priors[label].mu,
                        self.priors[label].sigma,
                    )
                else:
                    x0 = torch.zeros(n_spec).uniform_(
                        self.priors[label].lower_bound,
                        self.priors[label].upper_bound,
                    )
                if label == 'Fe':
                    x0 = fe0
                elif label not in ['Teff', 'logg', 'v_micro', 'Fe']:
                    x0 += fe0
                x0 = self.emulator.scale_stellar_labels(
                    x0 * torch.ones(self.n_stellar_labels)
                )[i]
            else:
                x0 = torch.zeros(n_spec).uniform_(-0.55, 0.55)
            with torch.no_grad():
                stellar_labels0[:, i] = ensure_tensor(x0)
        if self.use_holtzman2015:
            with torch.no_grad():
                stellar_labels0[:, 2] = self.emulator.scale_stellar_labels(
                    (2.478 - 0.325 * self.emulator.unscale_stellar_labels(stellar_labels0)[:, 1]) * torch.ones(
                        self.emulator.n_stellar_labels)
                )[2]
        mod_flux, mod_errs = self.emulator(
            stellar_labels=stellar_labels0,
            rv=self.rv,
            cont_coeffs=self.cont_coeffs,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
            inst_res=self.inst_res,
        )
        if self.loss_fn == 'neg_log_posterior':
            loss0 = self.neg_log_posterior(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        else:
            loss0 = self.loss_fn(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=torch.zeros_like(mod_flux),
                target_errs=self.obs_errs,
            )
        return stellar_labels0[loss0.argmin()].unsqueeze(0)

    def init_optimizer_scheduler(self):
        optim_params = []
        lr_lambdas = []
        if self.params['stellar_labels'] == 'fit':
            optim_params.append({'params': [self.stellar_labels], 'lr': self.learning_rates['stellar_labels']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['stellar_labels'] ** (
                    epoch // self.learning_rate_decay_ts['stellar_labels']))
        if self.params['rv'] == 'fit':
            optim_params.append({'params': [self.rv], 'lr': self.learning_rates['rv']})
            lr_lambdas.append(
                lambda epoch: self.learning_rate_decay['rv'] ** (epoch // self.learning_rate_decay_ts['rv']))
        if self.params['log_vmacro'] == 'fit':
            optim_params.append({'params': [self.log_vmacro], 'lr': self.learning_rates['log_vmacro']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['log_vmacro'] ** (
                    epoch // self.learning_rate_decay_ts['log_vmacro']))
        if self.params['log_vsini'] == 'fit':
            optim_params.append({'params': [self.log_vsini], 'lr': self.learning_rates['log_vsini']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['log_vsini'] ** (
                    epoch // self.learning_rate_decay_ts['log_vsini']))
        if self.params['inst_res'] == 'fit':
            optim_params.append({'params': [self.inst_res], 'lr': self.learning_rates['inst_res']})
            lr_lambdas.append(lambda epoch: self.learning_rate_decay['inst_res'] ** (
                    epoch // self.learning_rate_decay_ts['inst_res']))
        if self.params['cont_coeffs'] == 'fit':
            optim_params += [
                {'params': [self.cont_coeffs[o][i]],
                 'lr': torch.abs(self.cont_coeffs[o][i].mean()) * self.learning_rates['cont_coeffs']}
                for i in range(self.n_cont_coeffs) for o in range(self.n_obs)
            ]
            lr_lambdas += [
                lambda epoch: self.learning_rate_decay['cont_coeffs'] ** (
                        epoch // self.learning_rate_decay_ts['cont_coeffs'])
                for i in range(self.n_cont_coeffs) for o in range(self.n_obs)
            ]
        optimizer = torch.optim.Adam(optim_params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambdas
        )
        return optimizer, scheduler

    def atm_from_gaia_phot(self, scaled_feh):
        n_spec = scaled_feh.shape[0]
        teff_min = self.emulator.stellar_labels_min[0]
        teff_max = self.emulator.stellar_labels_max[0]
        logg_min = self.emulator.stellar_labels_min[1]
        logg_max = self.emulator.stellar_labels_max[1]
        fe_min = self.emulator.stellar_labels_min[self.fe_idx]
        fe_max = self.emulator.stellar_labels_max[self.fe_idx]
        unscaled_fe = (scaled_feh + 0.5) * (fe_max - fe_min) + fe_min
        vec = torch.zeros(
            (self._gaia_n_cmd + self._gaia_n_pow, n_spec),
            dtype=torch.float64
        )
        feh_phot = torch.vstack(
            [unscaled_fe, self.gaia_bprp.repeat(n_spec), self.gaia_g.repeat(n_spec)]
        ).T * self._gaia_eps
        feh_phot_hat = (feh_phot - self._gaia_shift) / self._gaia_scale
        d = feh_phot.unsqueeze(0) - self._gaia_cmd.unsqueeze(1)
        r = torch.linalg.norm(d, axis=2)
        vec[:self._gaia_n_cmd, :] = thin_plate_spline(r)
        feh_phot_hat_powers = feh_phot_hat.unsqueeze(0) ** self._gaia_pow.unsqueeze(1)
        vec[self._gaia_n_cmd:, :] = torch.prod(feh_phot_hat_powers, axis=2)
        logg_logteff = torch.tensordot(self._gaia_coeffs, vec, dims=[[0], [0]])
        scaled_logg = (logg_logteff[0] - logg_min) / (logg_max - logg_min) - 0.5
        scaled_teff = (10 ** logg_logteff[1] - teff_min) / (teff_max - teff_min) - 0.5
        scaled_logg_teff = torch.vstack([scaled_teff, scaled_logg]).T
        return scaled_logg_teff

    def holtzman2015(self, scaled_logg):
        logg_min = self.emulator.stellar_labels_min[1]
        logg_max = self.emulator.stellar_labels_max[1]
        vmicro_min = self.emulator.stellar_labels_min[2]
        vmicro_max = self.emulator.stellar_labels_max[2]
        unscaled_logg = (scaled_logg + 0.5) * (logg_max - logg_min) + logg_min
        unscaled_vmicro = 2.478 - 0.325 * unscaled_logg
        scaled_vmicro = (unscaled_vmicro - vmicro_min) / (vmicro_max - vmicro_min) - 0.5
        return scaled_vmicro

    def neg_log_posterior(self, pred, target, pred_errs, target_errs):
        log_likelihood = gaussian_log_likelihood_multi(pred, target, pred_errs, target_errs)
        log_priors = torch.zeros_like(log_likelihood)
        unscaled_stellar_labels = self.emulator.unscale_stellar_labels(self.stellar_labels)
        fe_idx = self.emulator.labels.index('Fe')
        for i, label in enumerate(self.emulator.labels):
            if label in ['Teff', 'logg', 'v_micro', 'Fe']:
                log_priors += self.priors['stellar_labels'][i](unscaled_stellar_labels[:, i])
            else:
                log_priors += self.priors['stellar_labels'][i](
                    unscaled_stellar_labels[:, i] - unscaled_stellar_labels[:, fe_idx]
                )
        if self.log_vmacro is not None:
            log_priors += self.priors['log_vmacro'](self.log_vmacro[:, 0])
        if self.log_vsini is not None:
            log_priors += self.priors['log_vsini'](self.log_vsini[:, 0])
        if self.inst_res is not None:
            log_priors += self.priors['inst_res'](self.inst_res[:, 0])
        return -1 * (log_likelihood + log_priors)

    def forward(self):
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            vmacro=None if self.log_vmacro is None else 10 ** self.log_vmacro,
            cont_coeffs=[torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)],
            inst_res=self.inst_res,
            vsini=None if self.log_vsini is None else 10 ** self.log_vsini,
        )
        return mod_flux * self.obs_blaz, mod_errs * self.obs_blaz

    def fit(
            self,
            obs_flux,
            obs_errs,
            obs_wave,
            obs_blaz=None,
            obs_mask=None,
            params=dict(
                stellar_labels='fit',
                rv='fit',
                vmacro='const',
                vsini='const',
                inst_res='const',
                cont_coeffs='fit',
            ),
            init_params=dict(
                stellar_labels=torch.zeros(1, 12),
                rv='prefit',
                vmacro=None,
                vsini=None,
                inst_res=None,
                cont_coeffs='prefit'
            ),
            priors=None,
            min_epochs=1000,
            max_epochs=10000,
            prefit_cont_window=55,
            use_holtzman2015=False,
            use_gaia_phot=False,
            gaia_g=None,
            gaia_bprp=None,
            gaia_cmd_interpolator=None,
            verbose=False,
            plot_prefits=False,
            plot_fit_every=None,
    ):
        self.obs_flux = ensure_tensor(obs_flux)
        self.obs_errs = ensure_tensor(obs_errs)
        self.obs_snr = self.obs_flux / self.obs_errs
        self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        self.obs_blaz = ensure_tensor(obs_blaz) if obs_blaz is not None else torch.ones_like(self.obs_flux)
        self.obs_mask = ensure_tensor(obs_mask, precision=bool) \
            if obs_mask is not None \
            else torch.ones_like(self.obs_flux, dtype=bool)
        #if torch.all(self.obs_wave != self.emulator.obs_wave):
        #    raise RuntimeError("obs_wave of Emulator and Optimizer differ!")
        self.obs_norm_wave = torch.stack([emulator.obs_norm_wave for emulator in self.emulator.emulators])
        #self.obs_wave_ = self.emulator.obs_wave_
        self.n_obs_ord = self.emulator.n_obs_ord
        self.n_obs_pix_per_ord = self.emulator.n_obs_pix

        self.c_flat = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        self.c_flat[0] = 1.0
        self.prefit_cont_window = prefit_cont_window

        self.params = params
        self.init_params = init_params

        self.priors = priors
        self.fe_idx = self.emulator.labels.index('Fe')
        self.fe_scaled_idx = [
            i for i, label in enumerate(self.emulator.labels)
            if label not in ['Teff', 'logg', 'v_micro', 'Fe']
        ]
        self.fe_scaler = torch.zeros(2, self.emulator.n_stellar_labels)
        self.fe_scaler[:, self.fe_scaled_idx] = 1
        self.stellar_label_bounds = torch.Tensor([
            [prior.lower_bound, prior.upper_bound]
            if type(prior) == UniformLogPrior
            else [-np.inf, np.inf]
            for prior in self.priors['stellar_labels']
        ]).T
        self.use_holtzman2015 = use_holtzman2015
        self.use_gaia_phot = use_gaia_phot
        if self.use_gaia_phot:
            if gaia_g is None or gaia_bprp is None:
                raise ValueError("Gaia photometry is not provided")
            if gaia_cmd_interpolator is None:
                raise ValueError("Gaia CMD interpolator is not provided")
            self.gaia_g = ensure_tensor(gaia_g)
            self.gaia_bprp = ensure_tensor(gaia_bprp)
            self._gaia_fn = gaia_cmd_interpolator
            self._gaia_cmd = ensure_tensor(self._gaia_fn.y * self._gaia_fn.epsilon, precision=torch.float64)
            self._gaia_eps = ensure_tensor(self._gaia_fn.epsilon, precision=torch.float64)
            self._gaia_pow = ensure_tensor(self._gaia_fn.powers, precision=torch.float64)
            self._gaia_coeffs = ensure_tensor(self._gaia_fn._coeffs, precision=torch.float64)
            self._gaia_shift = ensure_tensor(self._gaia_fn._shift, precision=torch.float64)
            self._gaia_scale = ensure_tensor(self._gaia_fn._scale, precision=torch.float64)
            self._gaia_n_cmd = self._gaia_cmd.shape[0]
            self._gaia_n_pow = self._gaia_pow.shape[0]
            self._gaia_n_coeffs = self._gaia_coeffs.shape[1]

        # Initialize Starting Values
        self.init_values(plot_prefits=plot_prefits)

        # Initialize Optimizer & Learning Rate Scheduler
        optimizer, scheduler = self.init_optimizer_scheduler()

        # Initialize Convergence Criteria
        epoch = 0
        self.loss = ensure_tensor(np.inf)
        delta_loss = ensure_tensor(np.inf)
        delta_stellar_labels = ensure_tensor(np.inf)
        delta_log_vmacro = ensure_tensor(np.inf)
        delta_rv = ensure_tensor(np.inf)
        delta_inst_res = ensure_tensor(np.inf)
        delta_log_vsini = ensure_tensor(np.inf)
        delta_frac_weighted_cont = ensure_tensor(np.inf)
        last_cont = torch.zeros_like(self.obs_blaz)

        # Initialize History
        self.history = dict(
            stellar_labels=[],
            log_vmacro=[],
            rv=[],
            inst_res=[],
            log_vsini=[],
            cont_coeffs=[],
            loss=[]
        )

        while (
                (epoch < min_epochs)
                or (
                        (epoch < max_epochs)
                        and (
                                (delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'])
                                or (delta_frac_weighted_cont.abs().max() > self.tolerances['d_cont'])
                                or (delta_log_vmacro.abs() > self.tolerances['d_log_vmacro'])
                                or (delta_rv.abs() > self.tolerances['d_rv'])
                                or (delta_inst_res.abs() > self.tolerances['d_inst_res'])
                                or (delta_log_vsini.abs() > self.tolerances['d_log_vsini'])
                        )
                        and (
                                delta_loss.abs() > self.tolerances['d_loss']
                        )
                        and (
                                self.loss > self.tolerances['loss']
                        )
                )
        ):
            # Forward Pass
            #with torch.autograd.detect_anomaly():
            mod_flux, mod_errs = self.forward()
            if self.loss_fn == 'neg_log_posterior':
                loss_epoch = self.neg_log_posterior(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                )
            else:
                loss_epoch = self.loss_fn(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                )
            if torch.isnan(loss_epoch):
                raise RuntimeError('NaN value returned for loss')

            # Log Results / History
            delta_loss = self.loss - loss_epoch
            self.loss = loss_epoch.item()
            self.history['stellar_labels'].append(torch.clone(self.stellar_labels).detach())
            self.history['rv'].append(torch.clone(self.rv))
            if self.log_vmacro is None:
                self.history['log_vmacro'].append(None)
            else:
                self.history['log_vmacro'].append(torch.clone(self.log_vmacro))
            if self.inst_res is None:
                self.history['inst_res'].append(None)
            else:
                self.history['inst_res'].append(torch.clone(self.inst_res))
            if self.log_vsini is None:
                self.history['log_vsini'].append(None)
            else:
                self.history['log_vsini'].append(torch.clone(self.log_vsini))
            self.history['cont_coeffs'].append(torch.clone(self.cont_coeffs_tensor))
            self.history['loss'].append(self.loss)

            # Backward Pass
            #with torch.autograd.detect_anomaly():
            optimizer.zero_grad()
            loss_epoch.backward()
            optimizer.step()
            scheduler.step()
            if torch.isnan(self.stellar_labels).any():
                raise RuntimeError('NaN value(s) suggested for stellar_labels')

            # Set Bounds
            with torch.no_grad():
                # Enforce Stellar Label Priors
                # self.stellar_labels.clamp_(min=-0.55, max=0.55)
                unscaled_stellar_labels = self.emulator.unscale_stellar_labels(self.stellar_labels)
                scaled_stellar_bounds = self.emulator.scale_stellar_labels(
                    self.stellar_label_bounds + self.fe_scaler * unscaled_stellar_labels[:, self.fe_idx]
                )
                for i in range(self.n_stellar_labels):
                    self.stellar_labels[:, i].clamp_(
                        min=torch.max(scaled_stellar_bounds[0, i], ensure_tensor(-0.5)).item(),
                        max=torch.min(scaled_stellar_bounds[1, i], ensure_tensor(0.5)).item(),
                    )
                # self.stellar_labels.clamp_(  # Allowed in pytorch 1.9.0 but not in 1.8.0
                #    min=scaled_stellar_bounds[0],
                #    max=scaled_stellar_bounds[1],
                # )
                if self.use_gaia_phot:
                    self.stellar_labels[:, :2] = self.atm_from_gaia_phot(self.stellar_labels[:, self.fe_idx])
                if self.use_holtzman2015:
                    self.stellar_labels[:, 2] = self.holtzman2015(self.stellar_labels[:, 1])
                if self.log_vmacro is not None:
                    self.log_vmacro.clamp_(min=-1.0, max=1.3)
                if self.log_vsini is not None:
                    self.log_vsini.clamp_(min=-1.0, max=2.5)
                if self.inst_res is not None:
                    self.inst_res.clamp_(
                        min=100,
                        max=self.emulator.model_res if self.emulator.model_res is not None else np.inf
                    )
            self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])

            # Check Convergence
            delta_stellar_labels = self.stellar_labels - self.history['stellar_labels'][-1]
            delta_log_vmacro = ensure_tensor(0) if self.log_vmacro is None else self.log_vmacro - \
                                                                                self.history['log_vmacro'][-1]
            delta_rv = self.rv - self.history['rv'][-1]
            delta_inst_res = ensure_tensor(0) if self.inst_res is None else self.inst_res - self.history['inst_res'][-1]
            delta_log_vsini = ensure_tensor(0) if self.log_vsini is None else self.log_vsini - \
                                                                              self.history['log_vsini'][-1]
            delta_cont_coeffs = self.cont_coeffs_tensor - self.history['cont_coeffs'][-1]
            if epoch % 20 == 0:
                cont = self.emulator.calc_cont(
                    [torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)]
                )
                delta_cont = cont - last_cont
                delta_frac_weighted_cont = delta_cont / cont * self.obs_snr
                last_cont = cont
                if verbose:
                    print(f"Epoch: {epoch}, Current Loss: {self.loss:.6f}")
            if (plot_fit_every is not None) and (epoch % plot_fit_every == 0):
                tot_errs = torch.sqrt(mod_errs ** 2 + self.obs_errs ** 2)
                mse = ((mod_flux[0] - self.obs_flux) / tot_errs) ** 2
                mask = torch.isfinite(tot_errs)
                for o in range(self.n_obs):
                    for i in range(self.n_obs_ord):
                        fig = plt.figure(figsize=(20, 2))
                        gs = GridSpec(3, 1)
                        gs.update(hspace=0.0)
                        ax1 = plt.subplot(gs[:2, 0])
                        ax2 = plt.subplot(gs[2, 0], sharex=ax1)
                        ax1.scatter(self.obs_wave[o, i][mask[0, o, i]].detach().numpy(),
                                    self.obs_flux[o, i][mask[0, o, i]].detach().numpy(), c='k', marker='.', alpha=0.8, )
                        ax1.plot(self.obs_wave[o, i].detach().numpy(), mod_flux[0, o, i].detach().numpy(), c='r', alpha=0.8)
                        ax2.scatter(self.obs_wave[o, i][mask[0, o, i]].detach().numpy(), mse[0, o, i][mask[0, o, i]].detach().numpy(),
                                    c='k', marker='.', alpha=0.8)
                        ax1.tick_params('x', labelsize=0)
                        plt.show()
                        plt.close('all')
            epoch += 1

        # Recover Best Epoch
        self.stellar_labels = self.history['stellar_labels'][np.argmin(self.history['loss'])]
        self.log_vmacro = self.history['log_vmacro'][np.argmin(self.history['loss'])]
        self.rv = self.history['rv'][np.argmin(self.history['loss'])]
        self.inst_res = self.history['inst_res'][np.argmin(self.history['loss'])]
        self.log_vsini = self.history['log_vsini'][np.argmin(self.history['loss'])]
        self.cont_coeffs = [
            [
                self.history['cont_coeffs'][np.argmin(self.history['loss'])][o,i]
                for i in range(self.n_cont_coeffs)
            ]
            for o in range(self.n_obs)
        ]
        self.cont_coeffs_tensor = torch.stack([torch.stack(self.cont_coeffs[o]) for o in range(self.n_obs)])
        self.loss = np.min(self.history['loss'])
        self.best_model, self.best_model_errs = self.forward()

        print(f"Best Epoch: {np.argmin(self.history['loss'])}, Best Loss: {self.loss:.6f}")

        if not epoch < max_epochs and verbose:
            print('max_epochs reached')
        if not delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'] and verbose:
            print('d_stellar_labels tolerance reached')
        if not delta_rv.abs().max() > self.tolerances['d_rv']:
            print('d_rv tolerance reached')
        if not delta_log_vmacro.abs().max() > self.tolerances[
            'd_log_vmacro'] and self.log_vmacro is not None and verbose:
            print('d_log_vmacro tolerance reached')
        if not delta_inst_res.abs().max() > self.tolerances['d_inst_res'] and self.inst_res is not None:
            print('d_inst_res tolerance reached')
        if not delta_log_vsini.abs().max() > self.tolerances['d_log_vsini'] and self.log_vsini is not None:
            print('d_log_vsini tolerance reached')
        if not delta_frac_weighted_cont.abs().max() > self.tolerances['d_cont']:
            print('d_cont tolerance reached')
        if not delta_loss.abs() > self.tolerances['d_loss']:
            print('d_loss tolerance reached')
        if not self.loss > self.tolerances['loss']:
            print('loss tolerance reached')


class PayneLikelihood(torch.nn.Module):
    def __init__(self, model, obs, init_params=None):
        super(PayneLikelihood, self).__init__()
        self.model = model
        self.init_params = init_params
        self.obs = obs
        self.obs_spec = ensure_tensor(self.obs['spec'])
        self.obs_errs = ensure_tensor(self.obs['errs'])
        self.obs_blaz = ensure_tensor(self.obs['scaled_blaz'])
        self.obs_mask = ensure_tensor(self.obs['mask'], precision=bool)
        self.obs_norm_spec = self.obs_spec / self.obs_blaz
        self.obs_norm_errs = self.obs_errs / self.obs_blaz
        if init_params is not None:
            self.stellar_labels = torch.nn.Parameter(ensure_tensor(init_params['stellar_labels']))
            self.rv = torch.nn.Parameter(ensure_tensor(init_params['rv']))
            self.vmacro = torch.nn.Parameter(ensure_tensor(init_params['vmacro']))
            self.cont_coeffs = torch.nn.Parameter(ensure_tensor(init_params['cont_coeffs']))
        else:
            self.stellar_labels = torch.nn.Parameter(torch.zeros(model.n_stellar_labels))
            self.rv = torch.nn.Parameter(torch.zeros(1))
            self.vmacro = torch.nn.Parameter(torch.ones(1))
            c0 = torch.zeros(model.n_cont_coeffs, model.n_obs_ord)
            c0[0] = 1.0
            self.cont_coeffs = torch.nn.Parameter(c0)

    def forward(self):
        mod_spec, mod_errs = self.model(
            stellar_labels=self.stellar_labels.unsqueeze(0),
            rv=self.rv,
            vmacro=self.vmacro,
            cont_coeffs=self.cont_coeffs
        )
        tot_vars = mod_errs.squeeze() ** 2 + self.obs_norm_errs ** 2
        loglike = -0.5 * (
                    torch.log(2 * np.pi * tot_vars) + (self.obs_norm_spec - mod_spec.squeeze()) ** 2 / (2 * tot_vars))
        return torch.sum(loglike[self.obs_mask])


