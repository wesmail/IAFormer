# Generic imports
import math
import h5py
import numpy as np

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader

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
        self.sparse_adj_matrix = self.file["adj_matrices"]
        self.mask = self.file["mask"]
        self.labels = self.file["labels"]

    def __getitem__(self, item) -> dict:
        node_features = torch.tensor(self.features[item], dtype=torch.float32)
        self.max_num_particles = node_features.shape[0]

        # Reconstruct the full adj matrix
        self.reconstruct_adjacency(self.sparse_adj_matrix[item])

        return {
            "node_features": node_features,
            "edge_features": torch.tensor(self.adj_matrix, dtype=torch.float32),
            "mask": torch.tensor(self.mask[item], dtype=torch.float32),
            "labels": torch.tensor(self.labels[item], dtype=torch.float32),
        }

    def __len__(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return len(self.labels)

    def infer_num_particles_from_pairs(self, pairs) -> int:
        # Solve n * (n - 1) / 2 = pairs
        discriminant = 1 + 8 * pairs
        n = int((-1 + math.sqrt(discriminant)) / 2)

        return n + 1

    def reconstruct_adjacency(self, flat_adj_matrix):
        # Initialize an empty adjacency matrix
        self.adj_matrix = np.zeros(
            (self.max_num_particles, self.max_num_particles, flat_adj_matrix.shape[1])
        )

        # Extract the non-padded values for the current feature
        valid_values = flat_adj_matrix[np.where(flat_adj_matrix[:, 0] != -1.0)[0], :]
        num_particles = self.infer_num_particles_from_pairs(valid_values.shape[0])

        # Get the upper triangle indices
        triu_indices = np.triu_indices(num_particles, k=1)
        for feature in range(valid_values.shape[1]):
            self.adj_matrix[triu_indices[0], triu_indices[1], feature] = valid_values[
                :, feature
            ]

        self.adj_matrix[:, :, :] += self.adj_matrix[:, :, :].transpose(
            1, 0, 2
        )  # Symmetrize

    def close(self):
        """
        Close the HDF5 file when done.
        """
        self.file.close()


class JetTaggingDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for managing training, validation, and test datasets
    for jet tagging stored in HDF5 files.

    Args:
        file_list (list): List containing paths to the HDF5 files for training, validation, and testing.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
    """

    def __init__(
        self,
        data_dir: str,
        file_list: list,
        batch_size: int = 32,
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        if len(file_list) != 3:
            raise ValueError(
                "file_list must contain exactly three file paths: [train_file, val_file, test_file]"
            )

        self.train_file, self.val_file, self.test_file = file_list
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Automatically save all parameters
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """
        Set up datasets for training, validation, and testing based on the provided stage.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_dataset = H5Dataset(data_dir=self.data_dir + self.train_file)
        self.val_dataset = H5Dataset(data_dir=self.data_dir + self.val_file)
        self.test_dataset = H5Dataset(data_dir=self.data_dir + self.test_file)

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training dataset.

        Returns:
            DataLoader: PyTorch DataLoader for training.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the validation dataset.

        Returns:
            DataLoader: PyTorch DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the test dataset.

        Returns:
            DataLoader: PyTorch DataLoader for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )
