import argparse
from pathlib import Path
import yaml
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from astropy.io import fits

import torch
from pytorch_lightning import seed_everything
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.data import PayneDataModule
from payne_optuna.fitting import PayneEmulator, PayneOptimizer, mse_loss
from payne_optuna.fitting import ensure_tensor


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Train the Payne")
    parser.add_argument("config_file", help="Fitting config yaml file")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


class MasterFlats():
    def __init__(self, file):
        with fits.open(file) as hdul:
            self.pixelflat_model = hdul['PIXELFLAT_MODEL'].data
        self.spec_dim, self.spat_dim = self.pixelflat_model.shape
        self.spec_arr = np.arange(self.spec_dim)
        self.spat_arr = np.arange(self.spat_dim)
        self.pixelflat_model[~np.isfinite(self.pixelflat_model)] = 1e10
        self.f_flat_2d = RectBivariateSpline(x=self.spec_arr, y=self.spat_arr, z=self.pixelflat_model, s=0)

    def get_raw_blaze(self, trace_spat):
        return self.f_flat_2d.ev(self.spec_arr, trace_spat)

    def get_blaze(self, trace_spat, std=2500):
        raw_blaze = self.get_raw_blaze(trace_spat)
        if np.any(raw_blaze < 0):
            idx = np.argwhere(raw_blaze < 0).flatten()[-1] + 5
        else:
            idx = 0
        f_flat_1d = UnivariateSpline(x=self.spec_arr[idx:], y=raw_blaze[idx:], s=self.spec_dim * std,
                                     ext='extrapolate')
        return f_flat_1d(self.spec_arr)


def load_spectrum(spec_file, orders_to_fit, extraction='OPT', flats=None):
    with fits.open(spec_file) as hdul:
        hdul_ext = list(hdul[0].header['EXT[0-9]*'].values())
        n_pix = hdul[1].data.shape[0]
        obs_ords = np.zeros(len(orders_to_fit))
        obs_dets = np.zeros(len(orders_to_fit))
        obs_spat = np.zeros((len(orders_to_fit), n_pix))
        obs_wave = np.zeros((len(orders_to_fit), n_pix))
        obs_spec = np.zeros((len(orders_to_fit), n_pix))
        obs_errs = np.zeros((len(orders_to_fit), n_pix))
        obs_blaz = np.ones((len(orders_to_fit), n_pix))
        obs_mask = np.ones((len(orders_to_fit), n_pix), dtype=bool)
        for i, order in enumerate(orders_to_fit):
            order_ext = [ext for ext in hdul_ext if f'{order:04.0f}' in ext][0]
            obs_ords[i] = int(order)
            obs_dets[i] = int(order_ext.split('DET')[1][:2])
            obs_spat[i] = hdul[order_ext].data[f'TRACE_SPAT']
            obs_wave[i] = hdul[order_ext].data[f'{extraction.upper()}_WAVE']
            obs_spec[i] = hdul[order_ext].data[f'{extraction.upper()}_COUNTS']
            obs_errs[i] = hdul[order_ext].data[f'{extraction.upper()}_COUNTS_SIG']
            obs_mask[i] = hdul[order_ext].data[f'{extraction.upper()}_MASK']
            if flats is not None:
                obs_blaz[i] = flats[obs_dets[i]].get_blaze(obs_spat[i])
        obs_dict = {
            'ords': obs_ords,
            'dets': obs_dets,
            'spat': obs_spat,
            'wave': obs_wave,
            'spec': obs_spec,
            'errs': obs_errs,
            'mask': obs_mask,
            'blaz': obs_blaz,
        }
        return obs_dict


def main(args):
    """
    Validate "The Payne" on synthetic validation spectra.

    :param args: Training configuration yaml file.

    The structure of the training config file is as follows:

    paths:
        input_dir: /PATH/TO/DIRECTORY/OF/SPECTRA
        output_dir: /PATH/TO/DIRECTORY/OF/MODELS
        spectra_file: training_spectra_and_labels.h5
    training:
        labels:
        - List
        - Of
        - Labels
        - To
        - Train
        - On
        learning_rate: 0.0001
        optimizer: RAdam
        train_fraction: 0.8
        batchsize: 512
        epochs: 10000
        patience: 1000
        precision: 16
        random_state: 9876
    architecture:
        n_layers: 2
        activation: LeakyReLU
        n_neurons: 300
        dropout: 0.0
    """

    # Set Tensor Type
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(dtype)

    # Load Configs & Set Paths
    with open(args.config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    with open(configs["paths"]["training_config"]) as file:
        training_configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = training_configs["name"]
    input_dir = Path(training_configs["paths"]["input_dir"])
    input_file = input_dir.joinpath(training_configs["paths"]["spectra_file"])
    output_dir = Path(training_configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    ckpt_file = sorted(list(ckpt_dir.glob('*.ckpt')))[-1]
    results_dir = model_dir.joinpath('validation_fitting')
    valid_label_file = results_dir.joinpath("validation_labels.npz")
    fit_label_file = results_dir.joinpath("best_fit_labels.npz")

    # Load Meta
    with open(meta_file) as file:
        meta = yaml.load(file, Loader=yaml.UnsafeLoader)

    # Load the Payne
    NN_model = LightningPaynePerceptron.load_from_checkpoint(
        ckpt_file,
        input_dim=meta['input_dim'],
        output_dim=meta['output_dim']
    )
    NN_model.load_meta(meta)

    try:
        validation_file = model_dir.joinpath('validation_results.npz')
        with np.load(validation_file) as tmp:
            mod_errs = tmp['median_approx_err_wave']
    except FileNotFoundError:
        print('validation_results.npz does not exist; assuming zero model error.')
        mod_errs = np.zeros_like(NN_model.wavelength)

    # Set Random Seed
    seed_everything(configs["training"]["random_state"])

    # Load Validation Data
    datamodule = PayneDataModule(
        input_file=input_file,
        labels_to_train_on=configs["training"]["labels"],
        train_fraction=configs["training"]["train_fraction"],
        batchsize=configs["training"]["batchsize"],
        dtype=dtype,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup()
    training_dataset = datamodule.training_dataset.dataset.__getitem__(datamodule.training_dataset.indices)
    validation_dataset = datamodule.validation_dataset.dataset.__getitem__(datamodule.validation_dataset.indices)
    n_valid = validation_dataset['labels'].shape[-1]

    # Load Real Data (for sample wavelength grid and blaze function)
    obs_dir = Path(configs["paths"]["obs_dir"])
    obs_spec_file = obs_dir.joinpath(configs["paths"]["obs_file"])
    flat_files = sorted(list(obs_dir.glob('MasterFlat*')))
    flats = {}
    for flat_file in flat_files:
        det = int(flat_file.with_suffix('').name[-2:])
        flats[det] = MasterFlats(flat_file)
    obs = load_spectrum(
        spec_file=obs_spec_file,
        orders_to_fit=np.arange(configs["max_order"], configs["min_order"] - 1, -1, dtype=int),
        extraction='OPT',
        flats=flats,
    )
    obs['mask'][:, :30] = False  # Mask detector edges
    obs['mask'][:, 3970:] = False  # Mask detector edges
    obs['mask'][obs['spec'] < 0] = False  # Mask negative fluxes
    obs['errs'][~obs['mask']] = np.inf

    # Initialize Emulator
    payne = PayneEmulator(
        model=NN_model,
        mod_errs=mod_errs,
        cont_deg=6,
        cont_wave_norm_range=(-10, 10),
        obs_wave=obs['wave'],
        model_res=None,
        vmacro_method='iso',
    )

    # Initialize Optimizer
    optimizer = PayneOptimizer(
        emulator=payne,
        loss_fn=mse_loss,
        learning_rates=dict(
            stellar_labels=5e-2,
            vmacro=5e-2,
            rv=1e-3,
            cont_coeffs=5e-2
        ),
        tolerances=dict(
            d_stellar_labels=1e-4,
            d_vmacro=1e-4,
            d_rv=1e-4,
            d_cont=1e-1,
            d_loss=-np.inf,
            loss=-np.inf,
        ),
    )

    # Validation Labels
    valid_stellar_labels = validation_dataset['labels'].detach().numpy().T
    valid_vmacro = 10 ** np.random.uniform(-1, 1, (n_valid, 1))
    valid_rv = np.random.uniform(-3, 3, (n_valid, 1))
    valid_cont_coeffs = np.zeros((n_valid, optimizer.n_cont_coeffs, optimizer.emulator.n_obs_ord))
    valid_cont_coeffs[:, 0, :] = 1.0
    np.savez(
        valid_label_file,
        stellar_labels=valid_stellar_labels,
        rv=valid_rv,
        vmacro=valid_vmacro,
        cont_coeffs=valid_cont_coeffs,
    )

    # Initialize fitting arrays
    fit_stellar_labels = np.zeros((n_valid, optimizer.n_stellar_labels))
    fit_rv = np.zeros((n_valid, 1))
    fit_vmacro = np.zeros((n_valid, 1))
    fit_cont_coeffs = np.zeros((n_valid, optimizer.n_cont_coeffs, optimizer.emulator.n_obs_ord))

    for i in range(n_valid):
        print(f'Fitting Spectra {i}/{n_valid}')
        valid_spec = validation_dataset['spectrum'][:, i].unsqueeze(0)
        valid_errs = 1e-3 * valid_spec
        mock_spec, mock_errs = optimizer.emulator.forward_model_spec(
            norm_flux=valid_spec,
            norm_errs=valid_errs,
            rv=ensure_tensor(valid_rv[i]),
            vmacro=ensure_tensor(valid_vmacro[i]),
            cont_coeffs=ensure_tensor(valid_cont_coeffs[i]),
            inst_res=None,
            vsini=None,
        )
        mock_errs[0, :, :30] = ensure_tensor(np.inf)
        mock_errs[0, :, 3970:] = ensure_tensor(np.inf)
        mock_spec *= obs['blaz']
        mock_errs *= obs['blaz']
        optimizer.fit(
            obs_flux=mock_spec.squeeze(),
            obs_errs=mock_errs.squeeze(),
            obs_wave=obs['wave'],
            obs_blaz=obs['blaz'],
            max_epochs=5000,
            prefit=['cont', 'rv'],
            verbose=True,
            plot_prefits=False,
        )
        fit_stellar_labels[i] = optimizer.stellar_labels.detach().numpy()
        fit_rv[i] = optimizer.rv.detach().numpy()
        fit_vmacro[i] = optimizer.vmacro.detach().numpy()
        fit_cont_coeffs[i] = torch.stack(optimizer.cont_coeffs).detach().numpy()
        history = {key: torch.stack(value).squeeze().detach().numpy() for key, value in optimizer.history.items() if key != "loss"}
        history["loss"] = np.array(optimizer.history["loss"])
        np.savez(
            results_dir.joinpath(f"optimization_history_{i}.npz"),
            **optimizer.history
        )

    np.savez(
        fit_label_file,
        stellar_labels=fit_stellar_labels,
        rv=fit_rv,
        vmacro=fit_vmacro,
        cont_coeffs=fit_cont_coeffs,
    )
