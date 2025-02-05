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
        self.adj_matrices = self.file["adj_matrices"]
        self.mask = self.file["mask"]
        self.labels = self.file["labels"]

    def __getitem__(self, item) -> dict:
        node_features = torch.tensor(self.features[item], dtype=torch.float32)
        self.max_num_particles = node_features.shape[0]

        # Reconstruct the full adj matrix
        # self.reconstruct_adjacency(self.adj_matrices[item])

        return {
            "node_features": node_features,
            # "edge_features": torch.tensor(self.adj_matrix, dtype=torch.float32),
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
        valid_values = flat_adj_matrix[np.where(flat_adj_matrix[:, 0] > -999.0)[0], :]
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


class MemMapDataset(Dataset):
    def __init__(self, data_dir: str, max_num_particles: int):
        """
        Args:
            data_dir (str): Path to the H5 file.
        """
        self.data_dir = data_dir
        self.max_num_particles = max_num_particles
        self.label_memmap = np.memmap(data_dir + "_labels.npy", dtype="int16", mode="r")
        self.size = self.label_memmap.shape[0]

        self.feature_matrix_memmap = np.memmap(
            data_dir + "_feature_matrix.npy",
            dtype="float32",
            mode="r",
            shape=(self.size, max_num_particles, 11),
        )
        self._adj_matrix_memmap = np.memmap(
            data_dir + "_adj_matrix.npy",
            dtype="float32",
            mode="r",
            shape=(self.size, max_num_particles * (max_num_particles - 1) // 2, 4),
        )
        self.mask_memmap = np.memmap(
            data_dir + "_mask.npy",
            dtype="int16",
            mode="r",
            shape=(self.size, max_num_particles),
        )

    def __len__(self) -> int:
        """
        Total number of samples in the dataset.
        """
        return self.size

    def __getitem__(self, item) -> dict:
        self.reconstruct_adjacency(self._adj_matrix_memmap[item])
        return {
            "node_features": torch.tensor(
                self.feature_matrix_memmap[item], dtype=torch.float32
            ),
            "edge_features": torch.tensor(self.adj_matrix, dtype=torch.float32),
            "mask": torch.tensor(self.mask_memmap[item], dtype=torch.float32),
            "labels": torch.tensor(self.label_memmap[item], dtype=torch.float32),
        }
        return {"node_features": torch.tensor(self.feature_matrix_memmap[item])}

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

        # Automatically save all parameters
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        """
        Set up datasets for training, validation, and testing based on the provided stage.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        self.train_dataset = MemMapDataset(
            data_dir=self.data_dir + self.train_file,
            max_num_particles=self.max_num_particles,
        )
        self.val_dataset = MemMapDataset(
            data_dir=self.data_dir + self.val_file,
            max_num_particles=self.max_num_particles,
        )
        self.test_dataset = MemMapDataset(
            data_dir=self.data_dir + self.test_file,
            max_num_particles=self.max_num_particles,
        )

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
            drop_last=True,
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
            drop_last=True,
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


class LorentzDataset(Dataset):
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
        idx = np.where(self.mask[item] == 1)[0]
        node_features = torch.tensor(
            self.feature_matrix[item][idx, :4], dtype=torch.float32
        )
        self.max_num_particles = self.feature_matrix[item].shape[0]

        flat_adj_matrix = self.adjacancy_matrix[item]
        self.reconstruct_adjacency(flat_adj_matrix)

        return (
            Data(
                x=node_features,
                edge_index=self.edge_index,
                y=torch.tensor(self.labels[item], dtype=torch.float32),
            ),
            torch.tensor(self.adj_matrix, dtype=torch.float32),
            torch.tensor(self.mask[item], dtype=torch.int16),
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
        self.edge_index = torch.triu_indices(num_particles, num_particles, offset=1)

        # Get the upper triangle indices
        triu_indices = np.triu_indices(num_particles, k=1)
        for feature in range(valid_values.shape[1]):
            self.adj_matrix[triu_indices[0], triu_indices[1], feature] = valid_values[
                :, feature
            ]

        # Symmetrize the matrix
        self.adj_matrix[:, :, :] += self.adj_matrix[:, :, :].transpose(1, 0, 2)


class LorentzDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for managing training, validation, and test datasets
    for jet tagging stored in HDF5 files. Adapted fro LorentzNet

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
        self.train_dataset = LorentzDataset(data_dir=self.data_dir + self.train_file)
        self.val_dataset = LorentzDataset(data_dir=self.data_dir + self.val_file)
        self.test_dataset = LorentzDataset(data_dir=self.data_dir + self.test_file)

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training dataset.

        Returns:
            DataLoader: PyTorch DataLoader for training.
        """

        return PyGDataLoader(
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

        return PyGDataLoader(
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
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )