from typing import Optional, List, Tuple, Union
from pathlib import Path
import yaml
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class SpectraHDF5Dataset(Dataset):
    """
    Represents an HDF5 dataset of spectra. Currently loads the entire dataset into memory.

    :param Union[str,Path] data_file: Path to the HDF5 file containing the dataset
    :param List[str] labels_to_train_on: Stellar Labels to include in the training
    :param dtype: Data type of dataset --- torch.cuda.FloatTensor for GPU or torch.FloatTensor for CPU
    :param Optional[Union[List[transforms],Tuple[transforms],transforms]] transform: PyTorch transform to apply
        to every data instance

    :ivar transform x_transform: Transforms applied to the labels
    :ivar transform y_transform: Transforms applied to the spectra
    :ivar pd.DataFrame labels: Scaled stellar labels of the dataset
    :ivar pd.DataFrame spectra: Spectra of the dataset
    :ivar float x_min: Minimum value in the dataset for each stellar label
    :ivar float x_max: Maximum value in the dataset for each stellar label
    """

    def __init__(
        self,
        data_file: Union[str, Path],
        labels_to_train_on: List[str],
        dtype,
        transform=None,
    ) -> None:
        super().__init__()
        self.labels_to_train_on = labels_to_train_on
        self.dtype = dtype

        # Parse Transformations
        if isinstance(transform, (list, tuple)):
            self.xtransform = transform[0]
            self.ytransform = transform[1]
        else:
            self.xtransform = transform
            self.ytransform = transform

        # Check Data File Path
        if not isinstance(data_file, Path):
            data_file = Path(data_file)
        if not data_file.exists():
            raise RuntimeError(f"{data_file} does not exist")
        self.data_file = data_file

        # Load & Scale Labels
        self.labels, self.x_min, self.x_max = self._load_and_scale_labels()
        self.spectra = self._load_spectra()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load Stellar Parameters
        x = self.labels.iloc[:, idx].values
        if self.xtransform:
            x = self.xtransform(x)
        x = torch.from_numpy(x).type(self.dtype)
        # Load Spectrum
        # y = self._load_spectrum(idx).values
        y = self.spectra.iloc[:, idx].values
        if self.ytransform:
            y = self.ytransform(y)
        y = torch.from_numpy(y).type(self.dtype)
        return {"labels": x, "spectrum": y}

    def __len__(self):
        return self.labels.shape[1]

    def _load_and_scale_labels(self):
        # Load Labels
        labels = pd.read_hdf(self.data_file, "labels")
        # Scale by Fe
        labels.loc[set(labels.index) ^ {"Teff", "logg", "v_micro", "Fe"}] -= labels.loc[
            "Fe"
        ]
        # Select labels to train on
        labels = labels.loc[self.labels_to_train_on]
        # Scale Labels
        x_min = labels.min(axis=1)
        x_max = labels.max(axis=1)
        num = labels.sub(x_min, axis=0)
        den = x_max - x_min
        scaled_labels = num.div(den, axis=0) - 0.5
        return scaled_labels, x_min, x_max

    def _load_spectrum(self, idx):
        # Load Labels
        spectrum = pd.read_hdf(self.data_file, "spectra", start=idx, stop=idx + 1)
        return spectrum

    def _load_spectra(self):
        spectra = pd.read_hdf(self.data_file, "spectra")
        return spectra


class ToTensor(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, sample):
        label, spectrum = sample["labels"], sample["spectrum"]
        return {
            "labels": torch.from_numpy(label.values).type(self.dtype),
            "spectrum": torch.from_numpy(spectrum.values).type(self.dtype),
        }


class PayneDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for Payne Spectra.

    :param Union[str,Path] input_file: Path to the HDF5 file containing the dataset
    :param labels_to_train_on: Stellar Labels to include in the training
    :param train_fraction: Fraction of dataset to train on. The remainder will be used for validation.
    :param int batchsize: Number of spectra per training batch.
    :param torch.tensortype dtype: Data type of dataset --- torch.cuda.FloatTensor for GPU or torch.FloatTensor for CPU
    :param int num_workers: Number of workers to load and batch dataset. Default = 0.
    :param bool pin_memory: Pin dataset to GPU memory. Not recommended since dataset is already loaded into GPU memory.
        Default = False.
    """
    def __init__(
        self,
        input_file: Union[str, Path],
        labels_to_train_on: List[str],
        train_fraction: float,
        batchsize: int,
        dtype: torch.tensortype,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.input_file = input_file
        self.labels_to_train_on = labels_to_train_on
        self.train_fraction = train_fraction
        self.batchsize = batchsize
        self.dtype = dtype
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = None  # transforms.Compose([ToTensor(dtype)])
        self.training_dataset = None
        self.validation_dataset = None
        self.input_dim = None
        self.output_dim = None
        self.x_min = None
        self.x_max = None
        self.wavelength = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = SpectraHDF5Dataset(
                self.input_file,
                self.labels_to_train_on,
                self.dtype,
                transform=self.transform,
            )
            n_train = int(len(dataset) * self.train_fraction)
            n_valid = int(len(dataset) - n_train)
            self.training_dataset, self.validation_dataset = random_split(
                dataset, [n_train, n_valid],
            )
            self.input_dim = len(self.training_dataset[0]["labels"])
            self.output_dim = len(self.training_dataset[0]["spectrum"])
            self.x_min = dataset.x_min
            self.x_max = dataset.x_max
            self.wavelength = pd.read_hdf(self.input_file, "wavelength")[0].values

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def save_training_meta(self, meta_file: Union[str, Path]):
        """
        Save meta data of the training.
        Includes in/output dimensions, which labels were trained on, x_min/max used for scaling,
        and the wavelength array of the spectra.

        :param Union[str,Path] meta_file: File to save meta data to.
        """
        training_meta = dict(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            labels=self.labels_to_train_on,
            x_min=self.x_min.to_dict(),
            x_max=self.x_max.to_dict(),
            wave=self.wavelength,
        )
        with open(meta_file, "w") as outfile:
            yaml.dump(training_meta, outfile, sort_keys=False)
