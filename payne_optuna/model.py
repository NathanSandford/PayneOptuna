from typing import Union, List, Tuple, Dict
from pathlib import PosixPath
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from . import radam
from .utils import j_nu, interp


class PaynePerceptron(torch.nn.Module):
    """
    Dense Pytorch model for the Payne.

    :param int input_dim: Input dimension (i.e., the number of labels)
    :param int output_dim: Output dimension (i.e., the number of pixels in the spectrum)
    :param int n_layers: Number of dense layers in the model. Default = 2.
    :param str activation: Activation function of the layers. Can be any of those included in torch.nn.
        Default = "LeakyReLU".
    :param Union[int, List[int]] n_neurons: Number of neurons in each layer.
        If a scalar, the same number of neurons will be assumed in each layer. Default = 300.
    :param Union[int, List[int]] dropout: Fractional dropout in each layer.
        If a scalar, the same dropout will be assumed in each layer. Default = 0.0 (i.e., no dropout).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        activation: str = "LeakyReLU",
        n_neurons: Union[int, List[int]] = 300,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:
        super(PaynePerceptron, self).__init__()
        self.layers = []
        activation_fn = getattr(torch.nn, activation)
        if isinstance(n_neurons, (list, tuple)):
            if len(n_neurons) != n_layers:
                raise ValueError("len(n_neurons) != n_layers")
        elif isinstance(n_neurons, int):
            n_neurons = tuple([n_neurons for _ in range(n_layers)])
        else:
            raise TypeError("n_neurons must be an integer or list of integers")
        if isinstance(dropout, (list, tuple)):
            if len(dropout) != n_layers:
                raise ValueError("len(dropout) != n_layers")
        elif isinstance(dropout, float):
            dropout = tuple([dropout for _ in range(n_layers)])
        else:
            raise TypeError("dropout must be a float or list of floats")
        for i in range(n_layers):
            layer_output_dim = n_neurons[i]
            self.layers.append(torch.nn.Linear(input_dim, layer_output_dim))
            self.layers.append(activation_fn())
            self.layers.append(torch.nn.Dropout(dropout[i]))
            input_dim = layer_output_dim
        self.layers.append(torch.nn.Linear(input_dim, output_dim))
        self.features = torch.nn.Sequential(*self.layers)

    def forward(self, data):
        return self.features(data)


class LightningPaynePerceptron(pl.LightningModule):
    """
    Pytorch Lightning wrapper for the PaynePerceptron.

    :param int input_dim: Input dimension (i.e., the number of labels)
    :param int output_dim: Output dimension (i.e., the number of pixels in the spectrum)
    :param int n_layers: Number of dense layers in the model. Default = 2.
    :param str activation: Activation function of the layers. Can be any of those included in torch.nn.
        Default = "LeakyReLU".
    :param Union[int, List[int]] n_neurons: Number of neurons in each layer.
        If a scalar, the same number of neurons will be assumed in each layer. Default = 300.
    :param Union[int, List[int]] dropout: Fractional dropout in each layer.
        If a scalar, the same dropout will be assumed in each layer. Default = 0.0 (i.e., no dropout).
    :param float lr: Learning rate of the model.
    :param str optimizer: Optimizer algorithm. Either "RAdam" or any of torch.optim. Default = "RAdam".

    :ivar object loss_fn: Loss function used in training. Currently set to MAE and ultimately multiplied by 1e4.
    :ivar Optional[Dict] meta: Dictionary of training meta data. Necessary for reconstructing the model after training.
    :ivar Optional[Dict[str, float]] x_min: Minimum value in the dataset for each stellar label. Needed for scaling labels.
    :ivar Optional[Dict[str, float]] x_max: Maximum value in the dataset for each stellar label. Needed for scaling labels.
    :ivar Optional[np.ndarray] wavelength: Wavelength array corresponding to the model spectrum.
    :ivar Optional[List[str]] labels: Names of the input model labels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        activation: str = "LeakyReLU",
        n_neurons: Union[int, List[int]] = 300,
        dropout: Union[float, List[float]] = 0.0,
        lr: float = 1e-4,
        optimizer: str = "RAdam",
    ) -> None:
        super(LightningPaynePerceptron, self).__init__()
        self.model = PaynePerceptron(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            activation=activation,
            n_neurons=n_neurons,
            dropout=dropout,
        )
        self.lr = lr
        self.optimizer_name = optimizer
        self.loss_fn = pl.metrics.regression.MeanAbsoluteError()

        self.meta = None
        self.x_min = None
        self.x_max = None
        self.wavelength = None
        self.labels = None

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        x = batch["labels"]
        y = batch["spectrum"]
        output = self.forward(x)
        loss = self.loss_fn(output, y) * 1e4
        self.log("train-loss", loss, on_step=False, on_epoch=True)
        return loss  # {"loss": loss}

    def validation_step(self, batch, batch_nb):
        x = batch["labels"]
        y = batch["spectrum"]
        output = self.forward(x)
        loss = self.loss_fn(output, y) * 1e4
        self.log("val-loss", loss, on_step=False, on_epoch=True)
        return loss

    # def validation_epoch_end(self, outputs):
    #    avg_loss = sum(x["batch_val_loss"] for x in outputs) / len(outputs)
    #    self.log('val_loss_epoch', avg_loss)

    # def validation_epoch_end(self, outputs):
    #    avg_loss = sum(x["batch_val_loss"] for x in outputs) / len(outputs)
    #    self.log('val_loss', avg_loss)
    #    return {"val_loss": avg_loss}

    def configure_optimizers(self):
        if self.optimizer_name == "RAdam":
            optimizer = radam.RAdam(
                [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
            )
        else:
            optimizer = getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), lr=self.lr
            )
        return optimizer

    def load_meta(self, meta: Union[str, Dict]) -> None:
        if isinstance(meta, (str, PosixPath)):
            with open(meta) as file:
                self.meta = yaml.load(file, Loader=yaml.UnsafeLoader)
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("meta must be either dictionary or path to yaml file.")
        self.input_dim = meta["input_dim"]
        self.output_dim = meta["output_dim"]
        self.x_min = meta["x_min"]
        self.x_max = meta["x_max"]
        self.wavelength = meta["wave"]
        self.labels = meta["labels"]


class PayneEmulator:
    def __init__(
        self,
        model,
        model_errs,
        cont_deg,
        cont_wave_norm_range=(-10,10),
        obs_wave=None
    ):
        self.model = model
        self.mod_wave = torch.from_numpy(self.model.wavelength)
        if model_errs is None:
            self.model_errs = torch.zeros_like(self.mod_wave)
        else:
            if not isinstance(model_errs, torch.Tensor):
                self.model_errs = torch.from_numpy(model_errs).to(torch.float32)
            else:
                self.model_errs = model_errs
        self.labels = model.labels
        self.x_min = torch.Tensor(list(model.x_min.values()))
        self.x_max = torch.Tensor(list(model.x_max.values()))

        self.cont_deg = cont_deg
        self.cont_wave_norm_range = cont_wave_norm_range

        if obs_wave is None:
            self.obs_wave = self.mod_wave
        else:
            self.obs_wave = torch.from_numpy(obs_wave) if not isinstance(obs_wave, torch.Tensor) else obs_wave
        self.obs_norm_wave, self.wave_norm_offset, self.wave_norm_scale = self.scale_wave(self.obs_wave)
        self.obs_wave_ = torch.stack([self.obs_norm_wave ** i for i in range(self.cont_deg + 1)], dim=0)

    def scale_labels(self, labels):
        scaled_labels = (labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        return scaled_labels

    def rescale_labels(self, scaled_labels):
        rescaled_labels = (scaled_labels + 0.5) * (self.x_max - self.x_min) + self.x_min
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
    def doppler_shift(wave, flux, errs, rv, fill=1):
        c = torch.tensor([2.99792458e5])  # km/s
        doppler_factor = torch.sqrt((1 - rv / c) / (1 + rv / c))
        new_wave = wave.unsqueeze(0) * doppler_factor.unsqueeze(-1)
        shifted_flux = interp(wave, flux, new_wave, fill)
        shifted_errs = interp(wave, errs, new_wave, fill)
        return shifted_flux.squeeze(), shifted_errs.squeeze()

    @staticmethod
    def inst_broaden(wave, flux, errs, R_out, R_in=None):
        sigma_out = (R_out * 2.355) ** -1
        if R_in is None:
            sigma_in = 0.0
        else:
            sigma_in = (R_in * 2.355) ** -1
        sigma = torch.sqrt(sigma_out ** 2 - sigma_in ** 2)
        inv_res_grid = torch.diff(torch.log(wave))
        dx = torch.median(inv_res_grid)
        ss = torch.fft.rfftfreq(wave.shape[-1], d=dx)
        kernel = torch.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
        flux_ff = np.fft.rfft(flux)
        errs_ff = np.fft.rfft(errs)
        flux_ff *= kernel
        errs_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv, errs_conv

    @staticmethod
    def vmacro_broaden(wave, flux, errs, vmacro):
        dv = 2.99792458e5 * torch.min(torch.diff(wave) / wave[:-1])
        eff_wave = torch.median(wave)
        freq = torch.fft.rfftfreq(flux.shape[-1], dv).to(torch.float64)
        flux_ff = torch.fft.rfft(flux)
        errs_ff = torch.fft.rfft(errs)
        sigma = vmacro / 3e5 * eff_wave  # Is there a better kernel that doesn't rely on eff_wave
        kernel = torch.exp(-2 * (np.pi * sigma * freq) ** 2)
        flux_ff *= kernel
        errs_ff *= kernel
        flux_conv = torch.fft.irfft(flux_ff, n=flux.shape[-1])
        errs_conv = torch.fft.irfft(errs_ff, n=errs.shape[-1])
        return flux_conv.squeeze(), errs_conv.squeeze()

    '''
    Old vmacro broadening using Gaussian Kernel instead of FFTs
    '''
    #@staticmethod
    #def vmacro_broaden(wave, flux, errs, vmacro, ks=21):
    #    n_spec = flux.shape[0]
    #    d_wave = wave[1] - wave[0]
    #    eff_wave = torch.median(wave)
    #    loc = (torch.arange(ks) - (ks - 1) // 2) * d_wave
    #    scale = vmacro / 3e5 * eff_wave
    #    norm = torch.distributions.normal.Normal(
    #        loc=torch.zeros(ks, 1),
    #        scale=scale.view(1, -1).repeat(ks, 1)
    #    )
    #    kernel = norm.log_prob(loc.view(-1, 1).repeat(1, n_spec)).exp()
    #    kernel = kernel / kernel.sum(axis=0)
    #    conv_spec = torch.nn.functional.conv1d(
    #        input=flux.view(1, n_spec, -1),
    #        weight=kernel.T.view(n_spec, 1, -1),
    #        padding=ks // 2,
    #        groups=n_spec,
    #    )
    #    conv_errs = torch.nn.functional.conv1d(
    #        input=errs.repeat(1, n_spec, 1),
    #        weight=kernel.T.view(n_spec, 1, -1),
    #        padding=ks // 2,
    #        groups=n_spec,
    #    )
    #    return conv_spec.squeeze(), conv_errs.squeeze()

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

    def __call__(self, x, rv, vmacro, cont_coeffs, inst_res=None, vsini=None):
        # Model Spectrum
        norm_flux = self.model(x)
        # Instrumental Broadening
        if inst_res is not None:
            conv_flux, conv_errs = self.inst_broaden(
                self.mod_wave,
                flux=norm_flux,
                errs=self.model_errs,
                vmacro=vmacro,
            )
        else:
            conv_flux = norm_flux
            conv_errs = self.model_errs
        # Macroturbulent Broadening
        conv_flux, conv_errs = self.vmacro_broaden(
            self.mod_wave,
            flux=conv_flux,
            errs=conv_errs,
            vmacro=vmacro,
        )
        # Rotational Broadening
        if vsini is not None:
            conv_flux, conv_errs = self.rot_broaden(
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
