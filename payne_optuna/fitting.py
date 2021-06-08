from typing import List
import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import percentile_filter
import torch
from .utils import ensure_tensor, j_nu
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def mse_loss(pred, target, pred_errs, target_errs):
    total_errs = torch.sqrt(pred_errs ** 2 + target_errs ** 2)
    return torch.mean(((pred - target) / total_errs) ** 2, axis=[1, 2])


class UniformLogPrior:
    def __init__(self, label, lower_bound, upper_bound):
        self.label = label
        self.lower_bound = ensure_tensor(lower_bound)
        self.upper_bound = ensure_tensor(upper_bound)

    def __call__(self, x):
        log_prior = torch.zeros_like(x)
        log_prior[(x < self.lower_bound) | (x > self.upper_bound)] = -np.inf
        return log_prior


class GaussianLogPrior:
    def __init__(self, label, mu, sigma):
        self.label = label
        self.mu = ensure_tensor(mu)
        self.sigma = ensure_tensor(sigma)

    def __call__(self, x):
        return torch.log(1.0 / (np.sqrt(2 * np.pi) * self.sigma)) - 0.5 * (x - self.mu) ** 2 / self.sigma ** 2


class FlatLogPrior:
    def __init__(self, label):
        self.label = label

    def __call__(self, x):
        return torch.zeros_like(x)


def gaussian_log_likelihood(pred, target, pred_errs, target_errs):  # , mask):
    tot_vars = pred_errs ** 2 + target_errs ** 2
    loglike = -0.5 * (
            torch.log(2 * np.pi * tot_vars) + (target - pred) ** 2 / (2 * tot_vars)
    )
    mask = torch.isfinite(loglike)
    loglike[..., ~mask] = 0  # Cludgy masking
    return torch.sum(loglike, axis=[1, 2])


class PayneEmulator:
    def __init__(
        self,
        model,
        mod_errs,
        cont_deg,
        rv_scale=100,
        cont_wave_norm_range=(-10,10),
        obs_wave=None,
        model_res=None,
        include_model_errs=True,
        vmacro_method='iso',
    ):
        self.model = model
        self.include_model_errs = include_model_errs
        self.mod_wave = ensure_tensor(self.model.wavelength, precision=torch.float64)
        self.mod_errs = ensure_tensor(mod_errs) if mod_errs is not None else torch.zeros_like(self.mod_wave)
        self.labels = model.labels
        self.stellar_labels_min = ensure_tensor(list(model.x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(model.x_max.values()))
        self.n_stellar_labels = self.model.input_dim

        self.model_res = model_res
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
        x_min = np.array(list(self.model.x_min.values()))
        x_max = np.array(list(self.model.x_max.values()))
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
                flux=norm_flux,
                errs=self.mod_errs,
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
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)

    def numpy(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        flux, errs = self(stellar_labels, rv, vmacro, cont_coeffs, inst_res, vsini)
        return flux.detach().numpy(), errs.detach().numpy()

    def __call__(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
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
                flux=norm_flux,
                errs=self.mod_errs,
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
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)


class CompositePayneEmulator(torch.nn.Module):
    def __init__(
            self,
            models: List[torch.nn.Module],
            model_bounds,
            cont_deg,
            rv_scale=100,
            cont_wave_norm_range=(-10, 10),
            obs_wave=None,
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
        sigma_ = sigma.repeat(sigma.shape[0], ss.shape[0]).view(ss.shape[0], -1).T
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

    def scale_stellar_labels(self, unscaled_labels):
        x_min = np.array(list(self.models[0].x_min.values()))
        x_max = np.array(list(self.models[0].x_max.values()))
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = np.array(list(self.models[0].x_min.values()))
        x_max = np.array(list(self.models[0].x_max.values()))
        return (scaled_labels + 0.5) * (x_max - x_min) + x_min

    def forward(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None , vsini=None):
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
            return intp_flux * cont_flux, intp_errs * cont_flux
        else:
            return intp_flux * cont_flux, torch.zeros_like(intp_flux)


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
                self.vmacro = self.init_params['vmacro'] if self.init_params['vmacro'] is not None else None
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
        self.stellar_labels = torch.zeros(1, self.n_stellar_labels)
        self.log_vmacro = ensure_tensor(log_vmacro0) if log_vmacro0 is not None else None
        self.log_vsini = ensure_tensor(log_vsini0) if log_vsini0 is not None else None
        #self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        rv0 = torch.linspace(rv_range[0], rv_range[1], n_rv)
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=rv0,
            cont_coeffs=self.c_flat,
            vmacro=None if self.log_vmacro is None else 10**self.log_vmacro,
            vsini=None if self.log_vsini is None else 10**self.log_vsini,
            inst_res=self.inst_res,
        )
        if self.loss_fn == 'neg_log_posterior':
            loss0 = self.neg_log_posterior(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=mod_errs * self.obs_blaz,
                target_errs=self.obs_errs,
            )
        else:
            loss0 = self.loss_fn(
                pred=mod_flux * self.obs_blaz,
                target=self.obs_flux,
                pred_errs=mod_errs * self.obs_blaz,
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
        #self.inst_res = ensure_tensor(inst_res0) if inst_res0 is not None else None
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            cont_coeffs=self.c_flat,
            vmacro=None if self.log_vmacro is None else 10**self.log_vmacro,
            vsini=None if self.log_vsini is None else 10**self.log_vsini,
            inst_res=self.inst_res,
        )
        c0 = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        footprint = np.concatenate([np.ones(self.prefit_cont_window), np.zeros(self.prefit_cont_window), np.ones(self.prefit_cont_window)])
        tot_errs = torch.sqrt((mod_errs[0] * self.obs_blaz)**2 + self.obs_errs**2)
        mask = torch.isfinite(tot_errs)
        scaling = self.obs_flux / (mod_flux[0] * self.obs_blaz)
        for i in range(self.n_obs_ord):
            filtered_scaling = percentile_filter(scaling[i].detach().numpy(), percentile=25, footprint=footprint)
            filtered_scaling = percentile_filter(filtered_scaling, percentile=75, footprint=footprint)
            filtered_scaling[~mask[i]] = 1.0
            p = Polynomial.fit(
                x=self.obs_norm_wave[i].detach().numpy(),
                y=filtered_scaling,
                deg=self.cont_deg,
                w=(tot_errs[i]**-1).detach().numpy(),
                window=self.cont_wave_norm_range
            )
            if plot:
                plt.figure(figsize=(20,1))
                plt.scatter(
                    self.obs_wave[i][mask[i]].detach().numpy(),
                    self.obs_flux[i][mask[i]].detach().numpy(),
                    c='k', marker='.', alpha=0.8
                )
                plt.plot(
                    self.obs_wave[i].detach().numpy(),
                    (mod_flux[0,i] * self.obs_blaz[i] * p(self.obs_norm_wave[i])).detach().numpy(),
                    c='r', alpha=0.8
                )
                plt.show()
            c0[:, i] = ensure_tensor(p.coef)
        return [c0[i] for i in range(self.n_cont_coeffs)]

    def prefit_vmacro(self, plot=False):
        raise NotImplementedError

    def prefit_vsini(self, plot=False):
        raise NotImplementedError

    def prefit_inst_res(self, plot=False):
        raise NotImplementedError

    def prefit_stellar_labels(self, plot=False):
        raise NotImplementedError

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

    def holtzman2015(self, scaled_logg):
        logg_min = np.array(list(self.emulator.models[0].x_min.values()))[1]
        logg_max = np.array(list(self.emulator.models[0].x_max.values()))[1]
        vmicro_min = np.array(list(self.emulator.models[0].x_min.values()))[2]
        vmicro_max = np.array(list(self.emulator.models[0].x_max.values()))[2]
        unscaled_logg = (scaled_logg + 0.5) * (logg_max - logg_min) + logg_min
        unscaled_vmicro = 2.478 - 0.325 * unscaled_logg
        scaled_vmicro = (unscaled_vmicro - vmicro_min) / (vmicro_max - vmicro_min) - 0.5
        return scaled_vmicro

    def neg_log_posterior(self, pred, target, pred_errs, target_errs):#, mask):
        log_likelihood = gaussian_log_likelihood(pred, target, pred_errs, target_errs)#, mask)
        log_priors = torch.zeros_like(log_likelihood)
        for i, label in enumerate(self.emulator.labels):
            log_priors += self.priors['stellar_labels'][i](self.stellar_labels[:, i])
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
            vmacro=None if self.log_vmacro is None else 10**self.log_vmacro,
            cont_coeffs=torch.stack(self.cont_coeffs),
            inst_res=self.inst_res,
            vsini=None if self.log_vsini is None else 10**self.log_vsini,
        )
        return mod_flux * self.obs_blaz, mod_errs * self.obs_blaz

    def fit(
            self,
            obs_flux,
            obs_errs,
            obs_wave,
            obs_blaz=None,
            params=dict(
                stellar_labels='fit',
                rv='fit',
                vmacro='const',
                vsini='const',
                inst_res='const',
                cont_coeffs='fit',
            ),
            init_params=dict(
                stellar_labels=torch.zeros(1,12),
                rv='prefit',
                vmacro=None,
                vsini=None,
                inst_res=None,
                cont_coeffs='prefit'
            ),
            priors=None,
            max_epochs=10000,
            prefit_cont_window=55,
            use_holtzman2015=False,
            verbose=False,
            plot_prefits=False,
            plot_fit_every=None,
    ):
        self.obs_flux = ensure_tensor(obs_flux)
        self.obs_errs = ensure_tensor(obs_errs)
        self.obs_snr = self.obs_flux / self.obs_errs
        self.obs_wave = ensure_tensor(obs_wave, precision=torch.float64)
        self.obs_blaz = ensure_tensor(obs_blaz) if obs_blaz is not None else torch.ones_like(self.obs_flux)
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

        self.use_holtzman2015 = use_holtzman2015

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
            else:
                loss_epoch = self.loss_fn(
                    pred=mod_flux,
                    target=self.obs_flux,
                    pred_errs=mod_errs,
                    target_errs=self.obs_errs,
                )

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
            self.history['cont_coeffs'].append(torch.clone(torch.stack(self.cont_coeffs)).detach())
            self.history['loss'].append(self.loss)

            # Backward Pass
            optimizer.zero_grad()
            loss_epoch.backward()
            optimizer.step()
            scheduler.step()

            # Set Bounds
            with torch.no_grad():
                self.stellar_labels.clamp_(min=-0.55, max=0.55)
                if self.use_holtzman2015:
                    self.stellar_labels[:, 2] = self.holtzman2015(self.stellar_labels[:, 1])
                if self.log_vmacro is not None:
                    self.log_vmacro.clamp_(min=-3.0, max=1.5)
                if self.log_vsini is not None:
                    self.log_vsini.clamp_(min=-1.0, max=2.5)
                if self.inst_res is not None:
                    self.inst_res.clamp_(
                        min=100,
                        max=self.emulator.model_res if self.emulator.model_res is not None else np.inf
                    )

            # Check Convergence
            delta_stellar_labels = self.stellar_labels - self.history['stellar_labels'][-1]
            delta_log_vmacro = ensure_tensor(0) if self.log_vmacro is None else self.log_vmacro - self.history['log_vmacro'][-1]
            delta_rv = self.rv - self.history['rv'][-1]
            delta_inst_res = ensure_tensor(0) if self.inst_res is None else self.inst_res - self.history['inst_res'][-1]
            delta_log_vsini = ensure_tensor(0) if self.log_vsini is None else self.log_vsini - self.history['log_vsini'][-1]
            delta_cont_coeffs = torch.stack(self.cont_coeffs) - self.history['cont_coeffs'][-1]
            if epoch % 20 == 0:
                cont = self.emulator.calc_cont(
                    torch.stack(self.cont_coeffs),
                    self.obs_wave_
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
            epoch += 1

        # Recover Best Epoch
        self.stellar_labels = self.history['stellar_labels'][np.argmin(self.history['loss'])]
        self.log_vmacro = self.history['log_vmacro'][np.argmin(self.history['loss'])]
        self.rv = self.history['rv'][np.argmin(self.history['loss'])]
        self.inst_res = self.history['inst_res'][np.argmin(self.history['loss'])]
        self.log_vsini = self.history['log_vsini'][np.argmin(self.history['loss'])]
        self.cont_coeffs = [self.history['cont_coeffs'][np.argmin(self.history['loss'])][i] for i in
                            range(self.n_cont_coeffs)]
        self.loss = np.min(self.history['loss'])
        self.best_model, self.best_model_errs = self.forward()

        print(f"Best Epoch: {np.argmin(self.history['loss'])}, Best Loss: {self.loss:.6f}")

        if not epoch < max_epochs and verbose:
            print('max_epochs reached')
        if not delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'] and verbose:
            print('d_stellar_labels tolerance reached')
        if not delta_rv.abs().max() > self.tolerances['d_rv']:
            print('d_rv tolerance reached')
        if not delta_log_vmacro.abs().max() > self.tolerances['d_log_vmacro'] and self.log_vmacro is not None and verbose:
            print('d_log_vmacro tolerance reached')
        if not delta_inst_res.abs().max() > self.tolerances['d_inst_res'] and self.inst_res is not None :
            print('d_inst_res tolerance reached')
        if not delta_log_vsini.abs().max() > self.tolerances['d_log_vsini'] and self.log_vsini is not None :
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


