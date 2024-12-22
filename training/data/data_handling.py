# Generic imports
import h5py
import numpy as np

# Torch imports
import torch
from torch.utils.data import Dataset, random_split, DataLoader

# PyTorch Lightning
import lightning as L
from lightning.pytorch import LightningDataModule


class H5Dataset(Dataset):
    def __init__(
        self,
        data_dir: str = "",
    ) -> None:
        """
        Args:
            data_dir (str): Path to the H5 file.
        """
        # Open the file in read-only mode
        self.file = h5py.File(data_dir, "r")

        self.features = self.file["feature_matrices"]
        self.adj_matrices = self.file["adj_matrices"]
        self.mask = self.file["mask"]
        self.labels = self.file["labels"]

    def __getitem__(self, item) -> dict:
        return {
            "node_features": torch.tensor(self.features[item], dtype=torch.float32),
            "edge_features": torch.tensor(self.adj_matrices[item], dtype=torch.float32),
            "mask": torch.tensor(self.mask[item], dtype=torch.float32),
            "labels": torch.tensor(self.labels[item], dtype=torch.float32),
        }

    def __len__(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return len(self.labels)

    def close(self):
        """
        Close the HDF5 file when done.
        """
        self.file.close()


class JetTaggingDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",
    ) -> None:
        super().__init__()

        # Automatically save all parameters
        self.save_hyperparameters()

    def __len__(self) -> int:
        # Return the number of graphs in the dataset
        pass

    def __getitem__(self, item) -> dict:
        return dict()

    def setup(self, stage: str) -> None:
        pass
