import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.data import PayneDataModule
from payne_optuna.plotting import validation_plots

def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Train the Payne")
    parser.add_argument("config_file", help="Training config yaml file")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


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
    #torch.set_default_tensor_type(dtype)

    # Load Configs & Set Paths
    with open(args.config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    input_dir = Path(configs["paths"]["input_dir"])
    output_dir = Path(configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    ckpt_file = sorted(list(ckpt_dir.glob('*.ckpt')))[-1]
    results_file = model_dir.joinpath('validation_results.npz')
    figure_file = model_dir.joinpath('validation_results.png')

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

    # Set Random Seed
    pl.seed_everything(configs["training"]["random_state"])

    # Initialize DataModule
    datamodule = PayneDataModule(
        input_file=input_dir,
        labels_to_train_on=configs["training"]["labels"],
        train_fraction=configs["training"]["train_fraction"],
        batchsize=configs["training"]["batchsize"],
        dtype=dtype,
        num_workers=0,
        pin_memory=False,
    )
    datamodule.setup()
    #training_dataset = datamodule.training_dataset.dataset.__getitem__(datamodule.training_dataset.indices)
    validation_dataset = datamodule.validation_dataset.dataset.__getitem__(datamodule.validation_dataset.indices)

    # Perform Validation
    model_spec = NN_model(validation_dataset['labels'].T).detach().numpy()
    valid_spec = validation_dataset['spectrum'].T.detach().numpy()
    approx_err = np.abs(model_spec - valid_spec)
    median_approx_err_star = np.median(approx_err, axis=1)
    median_approx_err_wave = np.median(approx_err, axis=0)
    twosigma_approx_err_wave = np.quantile(approx_err, q=0.9545, axis=0)

    np.savez(
        results_file,
        median_approx_err_star=median_approx_err_star,
        median_approx_err_wave=median_approx_err_wave,
        twosigma_approx_err_wave=twosigma_approx_err_wave,
    )

    fig = validation_plots(
        wavelength=NN_model.wavelength,
        valid_spec=valid_spec,
        approx_err=approx_err,
        median_approx_err_star=median_approx_err_star,
        median_approx_err_wave=median_approx_err_wave,
        twosigma_approx_err_wave=twosigma_approx_err_wave
    )
    fig.savefig(figure_file)
