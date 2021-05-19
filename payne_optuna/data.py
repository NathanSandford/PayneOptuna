from typing import Optional, List, Tuple, Union
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import pytorch_lightning as pl


class SpectraHDF5Dataset(Dataset):
    """
    Represents an HDF5 dataset of spectra. Currently loads the entire dataset into memory.

    :param Union[str,Path] data_file: Path to the HDF5 file containing the dataset
    :param List[str] labels_to_train_on: Stellar Labels to include in the training
    :param dtype: Data type of dataset --- torch.cuda.FloatTensor for GPU or torch.FloatTensor for CPU
    :param x_transform: PyTorch transform to apply to labels
    :param y_transform: PyTorch transform to apply to spectra
    :param bool iron_scale: Scale Labels by [Fe/H]?

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
        x_transform=None,
        y_transform=None,
        iron_scale: bool = False,
    ) -> None:
        super().__init__()
        self.labels_to_train_on = labels_to_train_on
        self.dtype = dtype
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iron_scale = iron_scale

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
        if self.x_transform:
            x = self.x_transform(x)
        x = torch.from_numpy(x).type(self.dtype)
        # Load Spectrum
        # y = self._load_spectrum(idx).values
        y = self.spectra.iloc[:, idx].values
        if self.y_transform:
            y = self.y_transform(y)
        y = torch.from_numpy(y).type(self.dtype)
        return {"labels": x, "spectrum": y}

    def __len__(self):
        return self.labels.shape[1]

    def _load_and_scale_labels(self):
        # Load Labels
        labels = pd.read_hdf(self.data_file, "labels")
        # Scale by Fe
        if self.iron_scale:
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

    def _load_spectra(self):
        spectra = pd.read_hdf(self.data_file, "spectra")
        return spectra


class SpectraHDF5BigDataset(Dataset):
    """
    Represents an HDF5 dataset of spectra. Currently loads the entire dataset into memory.

    :param Union[str,Path] data_file: Path to the HDF5 file containing the dataset
    :param List[str] labels_to_train_on: Stellar Labels to include in the training
    :param dtype: Data type of dataset --- torch.cuda.FloatTensor for GPU or torch.FloatTensor for CPU
    :param x_transform: PyTorch transform to apply to labels
    :param y_transform: PyTorch transform to apply to spectra
    :param bool iron_scale: Scale Labels by [Fe/H]?

    :ivar pd.DataFrame labels: Scaled stellar labels of the dataset
    :ivar pd.DataFrame spectra: Spectra of the dataset
    :ivar float x_min: Minimum value in the dataset for each stellar label
    :ivar float x_max: Maximum value in the dataset for each stellar label
    """

    def __init__(
        self,
        data_file_path: Union[str, Path],
        labels_to_train_on: List[str],
        dtype,
        x_transform=None,
        y_transform=None,
        iron_scale: bool = False,
    ) -> None:
        super().__init__()
        self.labels_to_train_on = labels_to_train_on
        self.dtype = dtype
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iron_scale = iron_scale
        self.virtual_dataset = Path(data_file_path).joinpath('virtual_dataset.h5')

        # Make & Load Virtual Dataset
        if not isinstance(data_file_path, Path):
            data_file_path = Path(data_file_path)
        if not data_file_path.exists():
            raise RuntimeError(f"{data_file_path} does not exist")
        if data_file_path.suffix == '.h5':
            data_files = [data_file_path]
        else:
            data_files = sorted(data_file_path.glob('*.h5'))
        if self.virtual_dataset in data_files:
            data_files.remove(self.virtual_dataset)
        if len(data_files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        self.make_virtual_dataset(data_files)
        self.load_virtual_dataset()
        self.scale_labels(iron_scale=self.iron_scale)

    def __getitem__(self, idx):
        if not hasattr(self, 'spectra'):
            self.load_virtual_dataset()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #idx.sort()
        # Load Stellar Parameters
        x = self.labels[idx]
        if self.x_transform:
            x = self.x_transform(x)
        x = torch.from_numpy(x).type(self.dtype)
        # Load Spectrum
        y = self.spectra[idx]
        if self.y_transform:
            y = self.y_transform(y)
        y = torch.from_numpy(y).type(self.dtype)
        return {"labels": x, "spectrum": y}

    def __len__(self):
        return self.n_spec

    def make_virtual_dataset(self, files):
        sources_spectra = []
        sources_labels = []
        n_spec_list = []
        n_pix_list = []
        n_labels_list = []
        n_datasets = len(files)
        for i, file in enumerate(files):
            with h5py.File(file, 'r') as input_file:
                if i == 0:
                    vsource_wavelength = h5py.VirtualSource(input_file['wavelength/block0_values'])
                    self.label_names = input_file['labels/axis0'][()].astype(str)
                vsource_spectra = h5py.VirtualSource(input_file['spectra/block0_values'])
                vsource_labels = h5py.VirtualSource(input_file['labels/block0_values'])
                sources_spectra.append(vsource_spectra)
                sources_labels.append(vsource_labels)
                n_spec_list.append(vsource_spectra.shape[0])
                n_pix_list.append(vsource_spectra.shape[1])
                n_labels_list.append(vsource_labels.shape[1])
        self.n_spec = np.sum(n_spec_list)
        if len(set(n_pix_list)) > 1:
            raise RuntimeError("Not all datasets have the same number of wavelength pixels.")
        self.n_pix = n_pix_list[0]
        if len(set(n_labels_list)) > 1:
            raise RuntimeError("Not all datasets have the same number of labels.")
        self.n_labels = n_labels_list[0]
        virtual_layout_spectra = h5py.VirtualLayout(
            shape=(self.n_spec, self.n_pix),
            dtype=np.float
        )
        virtual_layout_labels = h5py.VirtualLayout(
            shape=(self.n_spec, self.n_labels),
            dtype=np.float
        )
        virtual_layout_wavelength = h5py.VirtualLayout(
            shape=(self.n_pix,),
            dtype=np.float
        )
        offset = 0
        for i in range(n_datasets):
            length = n_spec_list[i]
            virtual_layout_spectra[offset: offset + length] = sources_spectra[i]
            virtual_layout_labels[offset: offset + length] = sources_labels[i]
            offset += length
        virtual_layout_wavelength[:] = vsource_wavelength
        with h5py.File(self.virtual_dataset, 'w', libver='latest') as f:
            f.create_virtual_dataset('spectra', virtual_layout_spectra, fillvalue=-999)
            f.create_virtual_dataset('labels', virtual_layout_labels, fillvalue=-999)
            f.create_virtual_dataset('wavelength', virtual_layout_wavelength, fillvalue=-999)

    def load_virtual_dataset(self):
        h5_file = h5py.File(self.virtual_dataset, "r")
        self.spectra = h5_file['spectra']
        self.raw_labels = h5_file['labels']
        self.wavelength = h5_file['wavelength']

    def scale_labels(self, iron_scale=False):
        # Sometimes it's just easier to work with Pandas Dataframes...
        self.labels_df = pd.DataFrame(self.raw_labels[()], columns=self.label_names)
        if iron_scale:
            self.labels_df.loc[:, set(self.labels_df.index) ^ {"Teff", "logg", "v_micro", "Fe"}] -= self.labels_df.loc[: , "Fe"]
        self.labels_df = self.labels_df.loc[:, self.labels_to_train_on]
        self.x_min = self.labels_df.min(axis=0)
        self.x_max = self.labels_df.max(axis=0)
        num = self.labels_df.sub(self.x_min, axis=1)
        den = self.x_max - self.x_min
        self.labels_df = num.div(den, axis=1) - 0.5
        self.labels = self.labels_df.values

class PayneDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for Payne Spectra.

    :param Union[str,Path] input_path: Path to the HDF5 file or directory containing the dataset
    :param List[str] labels_to_train_on: Stellar Labels to include in the training
    :param bool iron_scale: Scale Labels by [Fe/H]?
    :param train_fraction: Fraction of dataset to train on. The remainder will be used for validation.
    :param int batchsize: Number of spectra per training batch.
    :param torch.tensortype dtype: Data type of dataset --- torch.cuda.FloatTensor for GPU or torch.FloatTensor for CPU
    :param int num_workers: Number of workers to load and batch dataset. Default = 0.
    :param bool pin_memory: Pin dataset to GPU memory. Not recommended since dataset is already loaded into GPU memory.
    :param bool big_dataset: Use dataloader protocol for bigger multifile datasets.
        Default = False.
    """
    def __init__(
        self,
        input_path: Union[str, Path],
        labels_to_train_on: List[str],
        iron_scale: bool,
        train_fraction: float,
        batchsize: int,
        dtype,
        num_workers: int = 0,
        pin_memory: bool = False,
        big_dataset: bool = False,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.labels_to_train_on = labels_to_train_on
        self.iron_scale = iron_scale
        self.train_fraction = train_fraction
        self.batchsize = batchsize
        self.dtype = dtype
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.x_transform = None  # transforms.Compose([ToTensor(dtype)])
        self.y_transform = None  # transforms.Compose([ToTensor(dtype)])
        self.training_dataset = None
        self.validation_dataset = None
        self.input_dim = None
        self.output_dim = None
        self.x_min = None
        self.x_max = None
        self.wavelength = None
        self.big_dataset = big_dataset
        if big_dataset:
            self.setup = self.setup_big
        else:
            self.setup = self.setup_basic

    def setup_basic(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = SpectraHDF5Dataset(
                self.input_path,
                self.labels_to_train_on,
                self.dtype,
                self.x_transform,
                self.y_transform,
                self.iron_scale,
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
            self.wavelength = pd.read_hdf(self.input_path, "wavelength")[0].values

    def setup_big(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = SpectraHDF5BigDataset(
                self.input_path,
                self.labels_to_train_on,
                self.dtype,
                self.x_transform,
                self.y_transform,
                self.iron_scale,
            )
            n_train = int(len(dataset) * self.train_fraction)
            n_valid = int(len(dataset) - n_train)
            self.training_dataset, self.validation_dataset = random_split(
                dataset, [n_train, n_valid],
            )
            self.input_dim = dataset.labels.shape[1]
            self.output_dim = dataset.spectra.shape[1]
            self.x_min = dataset.x_min
            self.x_max = dataset.x_max
            self.wavelength = dataset.wavelength[()]

    def train_dataloader(self):
        return DataLoader(
            self.training_dataset,
            shuffle=True,
            batch_size=self.batchsize,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            shuffle=False,
            batch_size=self.batchsize,
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
