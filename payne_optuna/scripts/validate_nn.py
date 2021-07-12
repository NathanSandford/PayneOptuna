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
        big_dataset: False
        labels:
        - List
        - Of
        - Labels
        - To
        - Train
        - On
        iron_scale: False
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
    figure_file_valid = model_dir.joinpath('validation_results_valid.png')
    figure_file_train = model_dir.joinpath('validation_results_train.png')
    big_dataset = configs["training"]["big_dataset"]
    if big_dataset:
        input_path = input_dir
    else:
        input_path = input_dir.joinpath(configs["paths"]["spectra_file"])

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
    try:
        input_dir.joinpath('virtual_dataset.h5').unlink()
    except FileNotFoundError:
        pass
    datamodule = PayneDataModule(
        input_path=input_path,
        labels_to_train_on=configs["training"]["labels"],
        train_fraction=configs["training"]["train_fraction"],
        iron_scale=configs["training"]["iron_scale"],
        batchsize=configs["training"]["batchsize"],
        dtype=dtype,
        num_workers=0,
        pin_memory=False,
        big_dataset=big_dataset,
    )
    datamodule.setup()
    validation_dataset = datamodule.validation_dataset.dataset.__getitem__(sorted(datamodule.validation_dataset.indices))
    training_dataset = datamodule.training_dataset.dataset.__getitem__(sorted(datamodule.training_dataset.indices))

    # Perform Validation
    model_spec_valid = NN_model(validation_dataset['labels']).detach().numpy()
    model_spec_train = NN_model(training_dataset['labels']).detach().numpy()
    valid_spec = validation_dataset['spectrum'].detach().numpy()
    train_spec = training_dataset['spectrum'].detach().numpy()
    approx_err_valid = np.abs(model_spec_valid - valid_spec)
    approx_err_train = np.abs(model_spec_train - train_spec)
    median_approx_err_star_valid = np.median(approx_err_valid, axis=1)
    median_approx_err_wave_valid = np.median(approx_err_valid, axis=0)
    median_approx_err_star_train = np.median(approx_err_train, axis=1)
    median_approx_err_wave_train = np.median(approx_err_train, axis=0)
    twosigma_approx_err_wave_valid = np.quantile(approx_err_valid, q=0.9545, axis=0)
    twosigma_approx_err_wave_train = np.quantile(approx_err_train, q=0.9545, axis=0)

    np.savez(
        results_file,
        median_approx_err_star_valid=median_approx_err_star_valid,
        median_approx_err_wave_valid=median_approx_err_wave_valid,
        twosigma_approx_err_wave_valid=twosigma_approx_err_wave_valid,
        median_approx_err_star_train=median_approx_err_star_train,
        median_approx_err_wave_train=median_approx_err_wave_train,
        twosigma_approx_err_wave_train=twosigma_approx_err_wave_train,
    )

    fig = validation_plots(
        wavelength=NN_model.wavelength,
        valid_spec=valid_spec,
        approx_err=approx_err_valid,
        median_approx_err_star=median_approx_err_star_valid,
        median_approx_err_wave=median_approx_err_wave_valid,
        twosigma_approx_err_wave=twosigma_approx_err_wave_valid,
    )
    fig.savefig(figure_file_valid)

    fig = validation_plots(
        wavelength=NN_model.wavelength,
        valid_spec=train_spec,
        approx_err=approx_err_train,
        median_approx_err_star=median_approx_err_star_train,
        median_approx_err_wave=median_approx_err_wave_train,
        twosigma_approx_err_wave=twosigma_approx_err_wave_train,
    )
    fig.savefig(figure_file_train)
