from typing import Dict
import argparse
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from payne_optuna.model import LightningPaynePerceptron
from payne_optuna.data import PayneDataModule
from payne_optuna.callbacks import MetricsCallback, CheckpointCallback, EarlyStoppingCallback, PruningCallback


def parse_args(options=None):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description="Tune Hyperparameters of the Payne")
    parser.add_argument("config_file", help="Tuning config yaml file")
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


class Objective:
    """
    Objective Class for Optuna hyperparameter optimization.

    :param Dict configs: Dictionary of configs.

    The Structure of configs is as follows:

    configs = dict(
        paths = dict(
            input_dir: str = /PATH/TO/DIRECTORY/OF/SPECTRA
            output_dir: str = /PATH/TO/DIRECTORY/OF/MODELS
            spectra_file: str = training_spectra_and_labels.h5
        )
        training = dict(
            labels: List[str] = [list, of, labels, to, train, on,]  # list of model labels
            learning_rate: float = 0.0001  # learning rate of the training
            optimizer: str = "RAdam"  # Optimizer to use
            train_fraction: float = 0.8  # Fraction of spectra/labels to train on. The remainder is used for validation.
            batchsize: int = 512  # Number of spectra/labels to train on at a time
            epochs: int = 10000  # Number of epochs to train for
            patience: int = 1000  # Number of epochs w/o improvement to wait for before stopping early
            precision: int = 16  # Bit precision of training
            random_state: int = 9876  # Random seed of the training
        )
        tuning = dict(
            trials: int = 100  # Number of Optuna trials to run
            timeout: int = 144000  # Max runtime of Optuna hyperparameter searching in seconds
            pruner: str = MedianPruner  # Optuna Pruner to use to stop unpromising trials
            pruner_kwargs = dict(
                n_startup_trials: int = 5  # Number of trials to run before pruning
                n_warmup_steps: int = 5  # Number of epochs to wait in each trial before pruning
                interval_steps: int = 1  # Number of epochs to wait between pruning checks
            )
        )
    )
    """
    def __init__(self, configs: Dict) -> None:
        self.configs = configs
        self.model_name = configs["name"]
        self.input_dir = Path(configs["paths"]["input_dir"])
        self.output_dir = Path(configs["paths"]["output_dir"])
        self.model_dir = self.output_dir.joinpath(self.model_name)
        self.logger_dir = self.output_dir.joinpath("tb_logs")
        if not self.model_dir.is_dir():
            self.model_dir.mkdir()
        self.big_dataset = configs["training"]["big_dataset"]
        if self.big_dataset:
            self.input_path = self.input_dir
        else:
            self.input_path = self.input_dir.joinpath(configs["paths"]["spectra_file"])

        self.labels_to_train_on = configs["training"]["labels"]
        self.iron_scale = configs["training"]["iron_scale"]
        self.train_fraction = configs["training"]["train_fraction"]
        self.batchsize = configs["training"]["batchsize"]
        self.epochs = configs["training"]["epochs"]
        self.patience = configs["training"]["patience"]
        self.precision = configs["training"]["precision"]
        self.random_state = configs["training"]["random_state"]
        self.dtype = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        torch.set_default_tensor_type(self.dtype)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        # Set Paths
        trial_dir = self.model_dir.joinpath(f"trial_{trial.number:02d}")
        ckpt_dir = trial_dir.joinpath("ckpts")

        # Initialize Callbacks
        metrics_callback = MetricsCallback(["train-loss", "val-loss"])
        checkpoint_callback = CheckpointCallback(
            dirpath=ckpt_dir,
            filename="{epoch:05d}-{val-loss:.2f}",
            monitor="val-loss",
            mode="min",
            period=1,
            trial_number=trial.number,
            verbose=True,
        )
        early_stopping_callback = EarlyStoppingCallback(
            monitor="val-loss", mode="min", patience=self.patience
        )
        pruning_callback = PruningCallback(trial, monitor="val-loss", mode="min")

        # Initialize Logger
        logger = TensorBoardLogger(
            self.logger_dir, name=f"{self.model_name}_{trial.number:02d}"
        )

        # Set Random Seed
        pl.seed_everything(self.random_state)

        # Initialize Trainer
        trainer = pl.Trainer(
            default_root_dir=self.model_dir,
            logger=logger,
            profiler=False,
            max_epochs=self.epochs,
            gpus=1 if torch.cuda.is_available() else None,
            precision=self.precision,
            callbacks=[
                metrics_callback,
                checkpoint_callback,
                early_stopping_callback,
                pruning_callback,
            ],
            progress_bar_refresh_rate=0,
            check_val_every_n_epoch=1,
            deterministic=True,
        )

        # Initialize DataModule
        datamodule = PayneDataModule(
            input_path=self.input_path,
            labels_to_train_on=self.labels_to_train_on,
            train_fraction=self.train_fraction,
            iron_scale=self.iron_scale,
            batchsize=self.batchsize,
            dtype=self.dtype,
            num_workers=0,
            pin_memory=False,
            big_dataset=self.big_dataset,
        )
        datamodule.setup()
        self.input_dim = datamodule.input_dim
        self.output_dim = datamodule.output_dim

        # Initialize Model
        model = self.define_model(trial)

        # Train Model
        trainer.fit(model=model, datamodule=datamodule)

        # Return Best Metric
        return min(metrics_callback.metrics["val-loss"])

    def define_model(self, trial: optuna.trial.Trial) -> pl.LightningModule:
        """
        Wrapper for LightningPaynePerceptron allowing for Optuna trials to set hyperparameters.
        Currently allows for 1, 2, or 3 dense layers each with 100-1000 neurons and dropout fractions of 0.0-0.5.
        Activation functions allowed include ReLU, LeakyRelU, and sigmoid.
        Optimizers include RAdam, Adam, RMSprop, and SGD.
        The learning rates may be in the range 1e-5 and 1e-1.

        :param optuna.trial.Trial trial: Optuna trial
        :return pl.LightningModule: LightningPaynePerceptron model to train.
        """
        n_layers = trial.suggest_int("n_layers", 1, 3)
        activation = trial.suggest_categorical(
            "activation_fn", ["ReLU", "LeakyReLU", "Sigmoid"]
        )
        n_neurons = []
        dropout = []
        for i in range(n_layers):
            n_neurons.append(trial.suggest_int(f"n_neurons_{i}", 100, 1000))
            dropout.append(trial.suggest_float(f"dropout_{i}", 0.00, 0.5))
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["RAdam", "Adam", "RMSprop", "SGD"]
        )
        model = LightningPaynePerceptron(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_layers=n_layers,
            activation=activation,
            n_neurons=n_neurons,
            dropout=dropout,
            lr=lr,
            optimizer=optimizer_name,
        )
        return model


def main(args):
    """
    Tune Hyperparameters for "The Payne"

    :param args: Tuning configuration yaml file.

    The structure of the tuning config file is as follows:

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
    tuning:
        trials: 100
        timeout: 144000
        pruner: MedianPruner
        pruner_kwargs:
            n_startup_trials: 5
            n_warmup_steps: 200
            interval_steps: 1
    """

    # Load Configs
    with open(args.config_file) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    model_name = configs["name"]
    storage = f"sqlite:///{Path(configs['paths']['output_dir'])}/{model_name}.db"

    # Initialize Sampler
    sampler = optuna.samplers.TPESampler(seed=configs["training"]["random_state"])

    # Initialize Pruner
    pruner_class = getattr(optuna.pruners, configs["tuning"]["pruner"])
    pruner = pruner_class(**configs["tuning"]["pruner_kwargs"])

    # Initialize Study
    study = optuna.create_study(
        study_name=model_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )

    # Optimize Model Hyperparameters
    study.optimize(
        Objective(configs),
        n_trials=configs["tuning"]["trials"],
        timeout=configs["tuning"]["timeout"],
    )

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
