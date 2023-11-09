from typing import Union, List, Dict
from pathlib import PosixPath
import yaml
import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanSquaredError
from . import radam


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
        self.loss_fn = MeanSquaredError()

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
