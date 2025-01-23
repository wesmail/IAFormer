# Generic imports
import math
import h5py
import numpy as np

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader

# PyTorch Lightning
from lightning.pytorch import LightningDataModule

# PyG
from torch_geometric.data import Data


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

        self.feature_matrix = self.file["feature_matrix"]
        self.adjacancy_matrix = self.file["adjacancy_matrix"]
        self.mask = self.file["mask"]
        self.labels = self.file["labels"]

    def __getitem__(self, item) -> dict:
        node_features = torch.tensor(self.feature_matrix[item], dtype=torch.float32)
        self.max_num_particles = node_features.shape[0]

        # Reconstruct the full adj matrix
        self.reconstruct_adjacency(self.adjacancy_matrix[item])

        return {
            "node_features": node_features,
            "node_mask": torch.tensor(self.mask[item], dtype=torch.int16),
            "edge_features": torch.tensor(self.adj_matrix, dtype=torch.float32),
            "edge_mask": torch.tensor(
                np.any(self.adj_matrix != 0, axis=2), dtype=torch.int16
            ),
            "labels": torch.tensor(self.labels[item], dtype=torch.float32),
        }

    def __len__(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return len(self.labels)

    @staticmethod
    def infer_num_particles_from_pairs(pairs: int) -> int:
        """
        Infer the number of particles from the number of pairs.

        Args:
            pairs (int): Number of pairs in the adjacency matrix.

        Returns:
            int: Number of particles.
        """
        # Solve n * (n - 1) / 2 = pairs
        discriminant = 1 + 8 * pairs
        n = int((-1 + math.sqrt(discriminant)) / 2)
        return n + 1

    def reconstruct_adjacency(self, flat_adj_matrix):
        """
        Reconstruct the adjacency matrix from its flattened form.

        Args:
            flat_adj_matrix (np.ndarray): Flattened adjacency matrix.
            max_num_particles (int): Maximum number of particles in the graph.

        Returns:
            np.ndarray: Reconstructed adjacency matrix.
        """
        self.adj_matrix = np.zeros(
            (self.max_num_particles, self.max_num_particles, flat_adj_matrix.shape[1])
        )

        # Extract the non-padded values for the current feature
        valid_values = flat_adj_matrix[np.where(flat_adj_matrix[:, 0] > -999.0)[0], :]
        num_particles = self.infer_num_particles_from_pairs(valid_values.shape[0])

        # Get the upper triangle indices
        triu_indices = np.triu_indices(num_particles, k=1)
        for feature in range(valid_values.shape[1]):
            self.adj_matrix[triu_indices[0], triu_indices[1], feature] = valid_values[
                :, feature
            ]

        # Symmetrize the matrix
        self.adj_matrix[:, :, :] += self.adj_matrix[:, :, :].transpose(1, 0, 2)

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
        max_num_particles: int = 100,
        batch_size: int = 32,
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        if len(file_list) != 3:
            raise ValueError(
                "file_list must contain exactly three file paths: [train_file, val_file, test_file]"
            )

        self.train_file, self.val_file, self.test_file = file_list
        self.max_num_particles = max_num_particles
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = batch_size

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
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
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
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
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
            pin_memory=True,
            drop_last=True,
        )


class LorentDataModuleDataset(LightningDataModule):
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

        self.feature_matrix = self.file["feature_matrix"]
        self.adjacancy_matrix = self.file["adjacancy_matrix"]
        self.mask = self.file["mask"]
        self.labels = self.file["labels"]

    def __getitem__(self, item) -> dict:
        node_features = torch.tensor(
            self.feature_matrix[item][:, :4], dtype=torch.float32
        )

        flat_adj_matrix = self.adjacancy_matrix[item]
        valid_values = flat_adj_matrix[np.where(flat_adj_matrix[:, 0] > -999.0)[0], :]
        num_particles = self.infer_num_particles_from_pairs(valid_values.shape[0])

        # Get the upper triangle indices
        edge_index = torch.triu_indices(num_particles, num_particles, offset=1)

        return Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor(self.labels[item], dtype=torch.float32),
        )

    def __len__(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return len(self.labels)

    @staticmethod
    def infer_num_particles_from_pairs(pairs: int) -> int:
        """
        Infer the number of particles from the number of pairs.

        Args:
            pairs (int): Number of pairs in the adjacency matrix.

        Returns:
            int: Number of particles.
        """
        # Solve n * (n - 1) / 2 = pairs
        discriminant = 1 + 8 * pairs
        n = int((-1 + math.sqrt(discriminant)) / 2)
        return n + 1
