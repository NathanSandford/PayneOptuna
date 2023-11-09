import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.data import PayneDataModule
from payne_optuna.callbacks import MetricsCallback, CheckpointCallback, EarlyStoppingCallback


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Train the Payne")
    parser.add_argument("config_file", help="Training config yaml file")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from most recent checkpoint")
    parser.add_argument("--checkpoint", "-ckpt", help="Checkpoint to resume training from.")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    if args.checkpoint and args.resume:
        args.error("Cannot use both --resume and --checkpoint.")
    return args


def main(args):
    """
    Train "The Payne" on synthetic spectra.

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
    print(f'Default tensor type: {dtype}')

    torch.set_float32_matmul_precision('medium')

    # Load Configs & Set Paths
    with open(args.config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    input_dir = Path(configs["paths"]["input_dir"])
    output_dir = Path(configs["paths"]["output_dir"])
    model_dir = output_dir.joinpath(model_name)
    meta_file = model_dir.joinpath("training_meta.yml")
    ckpt_dir = model_dir.joinpath("ckpts")
    logger_dir = output_dir.joinpath("tb_logs")
    if not model_dir.is_dir():
        model_dir.mkdir()
    big_dataset = configs["training"]["big_dataset"]
    if big_dataset:
        input_path = input_dir
    else:
        input_path = input_dir.joinpath(configs["paths"]["spectra_file"])

    # Initialize Callbacks
    metrics_callback = MetricsCallback(["train-loss", "val-loss"])
    checkpoint_callback = CheckpointCallback(
        dirpath=ckpt_dir,
        filename="{epoch:05d}-{val-loss:.2f}",
        monitor="val-loss",
        mode="min",
        every_n_epochs=1,
        verbose=True,
    )
    early_stopping_callback = EarlyStoppingCallback(
        monitor="val-loss",
        mode="min",
        patience=configs["training"]["patience"],
        verbose=True,
    )

    # Initialize Logger
    logger = TensorBoardLogger(logger_dir, name=model_name)

    # Set Random Seed
    pl.seed_everything(configs["training"]["random_state"])

    # Set checkpoint if resuming a training session
    if args.checkpoint:
        checkpoint = ckpt_dir.joinpath(args.checkpoint)
    elif args.resume:
        all_ckpts = sorted(list(ckpt_dir.glob("*.ckpt")))
        if len(all_ckpts) > 0:
            idx = np.argmin([float(ckpt.name.split('=')[-1].split('ckpt')[0][:-1]) for ckpt in all_ckpts])
            checkpoint = all_ckpts[idx]
        else:
            print('No checkpoint to resume; beginning training from scratch')
            checkpoint = None
    else:
        checkpoint = None

    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir=model_dir,
        logger=logger,
        profiler="advanced",
        max_epochs=configs["training"]["epochs"],
        accelerator='gpu' if torch.cuda.is_available() else "cpu",
        strategy='ddp',
        precision=configs["training"]["precision"],
        callbacks=[metrics_callback, checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=1,
        deterministic=True,
    )
    #trainer = pl.Trainer(
    #    default_root_dir=model_dir,
    #    logger=logger,
    #    profiler="advanced",
    #    max_epochs=configs["training"]["epochs"],
    #    accelerator='gpu' if torch.cuda.is_available() else "cpu",
    #    gpus=-1 if torch.cuda.is_available() else None,
    #    strategy='ddp',
    #    precision=configs["training"]["precision"],
    #    callbacks=[metrics_callback, checkpoint_callback, early_stopping_callback],
    #    check_val_every_n_epoch=1,
    #    deterministic=True,
    #)

    # Initialize DataModule
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
    datamodule.save_training_meta(meta_file)

    # Initialize Model
    model = LightningPaynePerceptron(
        input_dim=datamodule.input_dim,
        output_dim=datamodule.output_dim,
        n_layers=configs["architecture"]["n_layers"],
        activation=configs["architecture"]["activation"],
        n_neurons=configs["architecture"]["n_neurons"],
        dropout=configs["architecture"]["dropout"],
        lr=configs["training"]["learning_rate"],
        optimizer=configs["training"]["optimizer"],
    )

    # Train Model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)

    print("Training Complete!")
