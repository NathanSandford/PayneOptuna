from typing import List
import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import percentile_filter
import torch
from .utils import ensure_tensor, j_nu, interp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def mse_loss(pred, target, pred_errs, target_errs):
    total_errs = torch.sqrt(pred_errs ** 2 + target_errs ** 2)
    return torch.mean(((pred - target) / total_errs) ** 2, axis=[1, 2])


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
        vmacro_method='iso',
    ):
        self.model = model
        self.mod_wave = ensure_tensor(self.model.wavelength, precision=torch.float64)
        self.mod_errs = ensure_tensor(mod_errs) if mod_errs is not None else torch.zeros_like(self.mod_wave)
        self.labels = model.labels
        self.stellar_labels_min = ensure_tensor(list(model.x_min.values()))
        self.stellar_labels_max = ensure_tensor(list(model.x_max.values()))
        self.n_stellar_labels = self.model.input_dim

        self.model_res = model_res
        self.rv_scale = rv_scale
        if vmacro_method == 'rt_fft':
            self.vmacro_broaden = self.vmacro_rt_broaden_fft
        elif vmacro_method == 'iso_fft':
            self.vmacro_broaden = self.vmacro_iso_broaden_fft
        elif vmacro_method == 'iso':
            self.vmacro_broaden = self.vmacro_iso_broaden
        else:
            print("vmacro_method not recognized, defaulting to 'iso'")
            self.vmacro_broaden = self.vmacro_iso_broaden

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

    def scale_labels(self, labels):
        scaled_labels = (labels - self.stellar_labels_min) / (self.stellar_labels_max - self.stellar_labels_min) - 0.5
        return scaled_labels

    def rescale_labels(self, scaled_labels):
        rescaled_labels = (scaled_labels + 0.5) * (self.stellar_labels_max - self.stellar_labels_min) + self.stellar_labels_min
        return rescaled_labels

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
    def doppler_shift(wave, flux, errs, rv, fill=1.0):
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((1 - rv / c) / (1 + rv / c))
        new_wave = wave.unsqueeze(0) * doppler_factor.unsqueeze(-1)
        shifted_flux = interp(wave, flux, new_wave, fill)
        shifted_errs = interp(wave, errs, new_wave, fill)
        return shifted_flux.squeeze(), shifted_errs.squeeze()

    def inst_broaden(self, wave, flux, errs, inst_res):
        sigma_out = (inst_res * 2.355) ** -1
        if self.model_res is None:
            sigma_in = 0.0
        else:
            sigma_in = (self.model_res * 2.355) ** -1
        sigma = torch.sqrt(sigma_out ** 2 - sigma_in ** 2)
        inv_res_grid = torch.diff(torch.log(wave))
        dx = torch.median(inv_res_grid)
        ss = torch.fft.rfftfreq(wave.shape[-1], d=dx)
        kernel = torch.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        flux_ff = torch.fft.rfft(flux)
        errs_ff = torch.fft.rfft(errs)
        flux_ff *= kernel
        errs_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv, errs_conv

    @staticmethod
    def vmacro_iso_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        errs_ff = torch.fft.rfft(errs)
        kernel = torch.exp(-2 * (np.pi * vmacro * freq) ** 2)
        flux_ff *= kernel
        errs_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv.squeeze(), errs_conv.squeeze()

    @staticmethod
    def vmacro_rt_broaden_fft(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        errs_ff = torch.fft.rfft(errs)
        kernel = (1 - torch.exp(-1*(np.pi*vmacro*freq)**2))/(np.pi*vmacro*freq)**2
        kernel[0] = 1.0
        flux_ff *= kernel
        errs_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv.squeeze(), errs_conv.squeeze()

    @staticmethod
    def vmacro_iso_broaden(wave, flux, errs, vmacro, ks=21):
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
        )
        conv_errs = torch.nn.functional.conv1d(
            input=errs.repeat(1, n_spec, 1),
            weight=kernel.T.view(n_spec, 1, -1),
            padding=ks // 2,
            groups=n_spec,
        )
        return conv_spec.squeeze(), conv_errs.squeeze()

    @staticmethod
    def rot_broaden(wave, flux, errs, vsini):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        errs_ff = torch.fft.rfft(errs)
        ub = 2.0 * np.pi * vsini * freq[1:]
        j1_term = j_nu(ub, 1) / ub
        cos_term = 3.0 * torch.cos(ub) / (2 * ub ** 2)
        sin_term = 3.0 * torch.sin(ub) / (2 * ub ** 3)
        sb = j1_term - cos_term + sin_term
        # Clean up rounding errors at low frequency; Should be safe for vsini > 0.1 km/s
        sb[freq[1:] < freq[1:][torch.argmax(sb)]] = 1.0
        flux_ff *= torch.cat([torch.Tensor([1.0]), sb])
        errs_ff *= torch.cat([torch.Tensor([1.0]), sb])
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv, errs_conv

    def forward_model_spec(self, norm_flux, norm_errs, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        # Instrumental Broadening
        if inst_res is not None:
            conv_flux, conv_errs = self.inst_broaden(
                self.mod_wave,
                flux=norm_flux,
                errs=norm_errs,
                inst_res=inst_res,
            )
        else:
            conv_flux = norm_flux
            conv_errs = norm_errs
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # Macroturbulent Broadening
        conv_flux, conv_errs = self.vmacro_broaden(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            vmacro=vmacro,
        )
        # RV Shift
        shifted_flux, shifted_errs = self.doppler_shift(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            rv=rv * self.rv_scale,
            fill=1.0,
        )
        # Interpolate to Observed Wavelength
        intp_flux = interp(
            x=self.mod_wave,
            y=shifted_flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        intp_errs = interp(
            x=self.mod_wave,
            y=shifted_errs,
            x_new=self.obs_wave,
            fill=1.0,
        )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        return intp_flux * cont_flux, intp_errs * cont_flux

    def numpy(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        flux, errs = self(stellar_labels, rv, vmacro, cont_coeffs, inst_res, vsini)
        return flux.detach().numpy(), errs.detach().numpy()

    def __call__(self, stellar_labels, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        # Model Spectrum
        norm_flux = self.model(stellar_labels)
        # Instrumental Broadening
        if inst_res is not None:
            conv_flux, conv_errs = self.inst_broaden(
                self.mod_wave,
                flux=norm_flux,
                errs=self.mod_errs,
                inst_res=inst_res,
            )
        else:
            conv_flux = norm_flux
            conv_errs = self.mod_errs
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
                flux=conv_flux,
                errs=conv_errs,
                vsini=vsini,
            )
        # Macroturbulent Broadening
        conv_flux, conv_errs = self.vmacro_broaden(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            vmacro=vmacro,
        )
        # RV Shift
        shifted_flux, shifted_errs = self.doppler_shift(
            wave=self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            rv=rv * self.rv_scale,
            fill=1.0,
        )
        # Interpolate to Observed Wavelength
        intp_flux = interp(
            x=self.mod_wave,
            y=shifted_flux,
            x_new=self.obs_wave,
            fill=1.0,
        )
        intp_errs = interp(
            x=self.mod_wave,
            y=shifted_errs,
            x_new=self.obs_wave,
            fill=1.0,
        )
        # Calculate Continuum Flux
        cont_flux = self.calc_cont(cont_coeffs, self.obs_wave_)
        return intp_flux * cont_flux, intp_errs * cont_flux


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
        self.model_res = ensure_tensor(model_res)
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
        y_lo = y_[:, lo]
        y_hi = y_[:, hi]
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        y_new = slope * (x_new - x_lo) + y_lo
        y_new[:, out_of_bounds] = fill
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
        kernel = torch.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= kernel
            errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        else:
            errs_conv = None
        return flux_conv, errs_conv

    @staticmethod
    def rot_broaden(wave, flux, errs, vsini):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        ub = 2.0 * np.pi * vsini * freq[1:]
        j1_term = j_nu(ub, 1) / ub
        cos_term = 3.0 * torch.cos(ub) / (2 * ub ** 2)
        sin_term = 3.0 * torch.sin(ub) / (2 * ub ** 3)
        sb = j1_term - cos_term + sin_term
        # Clean up rounding errors at low frequency; Should be safe for vsini > 0.1 km/s
        sb[freq[1:] < freq[1:][torch.argmax(sb)]] = 1.0
        flux_ff = torch.fft.rfft(flux)
        flux_ff *= torch.cat([torch.Tensor([1.0]), sb])
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        if errs is not None:
            errs_ff = torch.fft.rfft(errs)
            errs_ff *= torch.cat([torch.Tensor([1.0]), sb])
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
        x_max = np.array(list(self.models[0].x_min.values()))
        return (unscaled_labels - x_min) / (x_max - x_min) - 0.5

    def unscale_stellar_labels(self, scaled_labels):
        x_min = np.array(list(self.models[0].x_min.values()))
        x_max = np.array(list(self.models[0].x_min.values()))
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
                    flux=norm_flux,
                    errs=self.mod_errs[i],
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

    def prefit_rv(
            self,
            vmacro0=0.5,
            rv_range=(-3, 3),
            n_rv=301,
            plot=False,
    ):
        stellar_labels0 = torch.zeros(1, self.n_stellar_labels)
        vmacro0 = ensure_tensor(vmacro0)
        rv0 = torch.linspace(rv_range[0], rv_range[1], n_rv)
        mod_flux, mod_errs = self.emulator(
            stellar_labels=stellar_labels0,
            vmacro=vmacro0,
            rv=rv0,
            cont_coeffs=self.c_flat,
            inst_res=self.inst_res,
        )
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
        vmacro0=0.5,
        plot=False,
    ):
        stellar_labels0 = torch.zeros(1, self.n_stellar_labels)
        vmacro0 = ensure_tensor(vmacro0)
        rv0 = self.rv
        mod_flux, mod_errs = self.emulator(
            stellar_labels=stellar_labels0,
            vmacro=vmacro0,
            rv=rv0,
            cont_coeffs=self.c_flat,
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
                plt.scatter(self.obs_wave[i][mask[i]].detach().numpy(), self.obs_flux[i][mask[i]].detach().numpy(), c='k', marker='.', alpha=0.8)
                plt.plot(self.obs_wave[i].detach().numpy(), (mod_flux[0,i] * self.obs_blaz[i] * p(self.obs_norm_wave[i])).detach().numpy(), c='r', alpha=0.8)
                plt.show()
            c0[:, i] = ensure_tensor(p.coef)
        return [c0[i] for i in range(self.n_cont_coeffs)]

    def prefit_vmacro(self, plot=False):
        raise NotImplementedError

    def prefit_stellar_labels(self, plot=False):
        raise NotImplementedError

    def init_starting_labels(self, verbose=False, plot=False):
        if 'rv' in self.prefit:
            if verbose:
                print('Performing RV Pre-Fit')
            self.rv = self.prefit_rv(plot=plot).requires_grad_()
            if verbose:
                print(f'Best Initial RV: {self.rv.item() * self.emulator.rv_scale:.0f} km/s')
        else:
            self.rv = ensure_tensor(0.0).requires_grad_()
        if 'cont' in self.prefit:
            if verbose:
                print('Performing Continuum Pre-Fit')
            self.cont_coeffs = [coeffs.requires_grad_() for coeffs in self.prefit_cont(plot=plot)]
        else:
            self.cont_coeffs = [
                self.c_flat[i] for i in range(self.n_cont_coeffs)
            ]
        if 'vmacro' in self.prefit:
            self.vmacro = self.prefit_vmacro(plot=plot).requires_grad_()
        else:
            self.vmacro = ensure_tensor(1.0).requires_grad_()
        if 'stellar_labels' in self.prefit:
            self.stellar_labels = self.prefit_stellar_labels(plot=plot).requires_grad_()
        else:
            self.stellar_labels = torch.zeros(1, self.n_stellar_labels).requires_grad_()

    def forward(self):
        mod_flux, mod_errs = self.emulator(
            stellar_labels=self.stellar_labels,
            rv=self.rv,
            vmacro=self.vmacro,
            cont_coeffs=torch.stack(self.cont_coeffs),
            inst_res=self.inst_res,
            vsini=self.vsini,
        )
        return mod_flux * self.obs_blaz, mod_errs * self.obs_blaz

    def fit(
            self,
            obs_flux,
            obs_errs,
            obs_wave,
            obs_blaz=None,
            inst_res=None,
            vsini=None,
            max_epochs=1000,
            prefit=None,
            prefit_cont_window=55,
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
        self.inst_res = inst_res
        self.vsini = vsini

        self.c_flat = torch.zeros(self.n_cont_coeffs, self.n_obs_ord)
        self.c_flat[0] = 1.0

        # Initialize Starting Values
        self.prefit = prefit if prefit is not None else []
        self.prefit_cont_window = prefit_cont_window
        self.init_starting_labels(verbose=verbose, plot=plot_prefits)

        # Initialize Optimizer & Learning Rate Scheduler
        optimizer = torch.optim.Adam(
            [
                {'params': [self.stellar_labels], 'lr': self.learning_rates['stellar_labels']},
                {'params': [self.vmacro], 'lr': self.learning_rates['vmacro']},
                {'params': [self.rv], 'lr': self.learning_rates['rv']},
                #{'params': [self.inst_res], 'lr': self.learning_rates['inst_res']},
                #{'params': [self.vsini], 'lr': self.learning_rates['vsini']},
            ] +
            [
                {'params': [self.cont_coeffs[i]],
                 'lr': torch.abs(self.cont_coeffs[i].mean()) * self.learning_rates['cont_coeffs']}
                for i in range(self.n_cont_coeffs)
            ]
        )
        lr_lambda_stellar_labels = lambda epoch: self.learning_rate_decay['stellar_labels'] ** (epoch // self.learning_rate_decay_ts['stellar_labels'])
        lr_lambda_vmacro = lambda epoch: self.learning_rate_decay['vmacro'] ** (epoch // self.learning_rate_decay_ts['vmacro'])
        lr_lambda_rv = lambda epoch: self.learning_rate_decay['rv'] ** (epoch // self.learning_rate_decay_ts['rv'])
        #lr_lambda_inst_res = lambda epoch: self.learning_rate_decay['inst_res'] ** (epoch // self.learning_rate_decay_ts['inst_res'])
        #lr_lambda_vsini = lambda epoch: self.learning_rate_decay['vsini'] ** (epoch // self.learning_rate_decay_ts['vsini'])
        lr_lambda_cont_coeffs = lambda epoch: self.learning_rate_decay['cont_coeffs'] ** (epoch // self.learning_rate_decay_ts['cont_coeffs'])
        lr_lambda_list = [
            lr_lambda_stellar_labels,
            lr_lambda_vmacro,
            lr_lambda_rv,
            #lr_lambda_inst_res,
            #lr_lambda_vsini,
        ] + [lr_lambda_cont_coeffs for i in range(self.n_cont_coeffs)]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_lambda_list
        )

        # Initialize Convergence Criteria
        epoch = 0
        self.loss = ensure_tensor(np.inf)
        delta_loss = ensure_tensor(np.inf)
        delta_stellar_labels = ensure_tensor(np.inf)
        delta_vmacro = ensure_tensor(np.inf)
        delta_rv = ensure_tensor(np.inf)
        #delta_inst_res = ensure_tensor(np.inf)
        #delta_vsini = ensure_tensor(np.inf)
        delta_frac_weighted_cont = ensure_tensor(np.inf)
        last_cont = torch.zeros_like(self.obs_blaz)

        # Initialize Hostory
        self.history = dict(
            stellar_labels=[],
            vmacro=[],
            rv=[],
            #inst_res=[],
            #vsini=[],
            cont_coeffs=[],
            loss=[]
        )

        while (
                (epoch < max_epochs)
                and (
                        (delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'])
                        or (delta_frac_weighted_cont.abs().max() > self.tolerances['d_cont'])
                        or (delta_vmacro.abs() > self.tolerances['d_vmacro'])
                        or (delta_rv.abs() > self.tolerances['d_rv'])
                        #or (delta_inst_res.abs() > self.tolerances['d_inst_res'])
                        #or (delta_vsini.abs() > self.tolerances['d_vsini'])
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
            self.history['vmacro'].append(torch.clone(self.vmacro))
            self.history['rv'].append(torch.clone(self.rv))
            #self.history['inst_res'].append(torch.clone(self.inst_res))
            #self.history['vsini'].append(torch.clone(self.vsini))
            self.history['cont_coeffs'].append(torch.clone(torch.stack(self.cont_coeffs)).detach())
            self.history['loss'].append(self.loss)

            # Backward Pass
            optimizer.zero_grad()
            loss_epoch.backward()
            optimizer.step()
            scheduler.step()

            # Set Bounds
            with torch.no_grad():
                self.vmacro.clamp_(min=1e-3, max=15.0)
                self.stellar_labels.clamp_(min=-0.55, max=0.55)
                #self.vsini.clamp_(min=0.0)
                #self.inst_res.clamp_(
                #    min=100,
                #    max=self.emulator.mod_res if self.emulator.mod_res is not None else np.inf
                #)

            # Check Convergence
            delta_stellar_labels = self.stellar_labels - self.history['stellar_labels'][-1]
            delta_vmacro = self.vmacro - self.history['vmacro'][-1]
            delta_rv = self.rv - self.history['rv'][-1]
            #delta_inst_res = self.inst_res - self.history['inst_res'][-1]
            #delta_vsini = self.vsini - self.history['vsini'][-1]
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
        self.vbroad = self.history['vmacro'][np.argmin(self.history['loss'])]
        self.rv = self.history['rv'][np.argmin(self.history['loss'])]
        #self.inst_res = self.history['inst_res'][np.argmin(self.history['loss'])]
        #self.vsini = self.history['vsini'][np.argmin(self.history['loss'])]
        self.cont_coeffs = [self.history['cont_coeffs'][np.argmin(self.history['loss'])][i] for i in
                            range(self.n_cont_coeffs)]
        self.loss = np.min(self.history['loss'])
        self.best_model, self.best_model_errs = self.forward()

        print(f"Best Epoch: {np.argmin(self.history['loss'])}, Best Loss: {self.loss:.6f}")

        if not epoch < max_epochs and verbose:
            print('max_epochs reached')
        if not delta_stellar_labels.abs().max() > self.tolerances['d_stellar_labels'] and verbose:
            print('d_stellar_labels tolerance reached')
        if not delta_vmacro.abs().max() > self.tolerances['d_vmacro'] and verbose:
            print('d_vmacro tolerance reached')
        if not delta_rv.abs().max() > self.tolerances['d_rv']:
            print('d_rv tolerance reached')
        #if not delta_inst_res.abs().max() > self.tolerances['d_inst_res']:
        #    print('d_inst_res tolerance reached')
        #if not delta_vsini.abs().max() > self.tolerances['d_vsini']:
        #    print('d_vsini tolerance reached')
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


