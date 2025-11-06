import torch
import math
import h5py
from typing import Optional

from lightning.pytorch import LightningDataModule

# PyG imports
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------
# Dataset for a single JetClass HDF5 file
# ---------------------------------------------------------------------
class JetClassDenseDataset(Dataset):
    """
    Dataset for a single JetClass HDF5 file.
    Returns a dict:
      {
        'node_features': torch.FloatTensor [max_particles, n_node_features],
        'edge_features': torch.FloatTensor [max_particles, max_particles, n_edge_features],
        'labels': torch.LongTensor []
      }
    """

    def __init__(self, h5_path: str):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r", swmr=True)
        self.n_particles = self.h5["n_particles"]
        self.x = self.h5["feature_matrix"]
        self.edge = self.h5["adjacency_matrix"]
        self.y = self.h5["labels"]

    # Convert edge indices (i, j) to upper triangular linear index
    # For upper triangle (i < j): index = i * n - i*(i+1)/2 + (j-i-1)
    @staticmethod
    def edge_indices_to_triu_index(i, j, n):
        """
        Convert edge indices (i, j) where i < j to linear index in upper triangular storage
        """
        return i * n - i * (i + 1) // 2 + j - i - 1            

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item: int):
        node_features = torch.as_tensor(self.x[item], dtype=torch.float32)
        edge_features = torch.as_tensor(self.edge[item], dtype=torch.float32)

        # Indices for upper triangle (excluding diagonal), in canonical order
        i, j = torch.triu_indices(self.n_particles[item], self.n_particles[item], offset=1)  # each of shape [E]
        edge_index = torch.stack([i, j], dim=0)

        # Extract edge attributes
        triu_indices = self.edge_indices_to_triu_index(edge_index[0], edge_index[1], self.n_particles[item])
        edge_attr = edge_features[triu_indices]        

        label = torch.tensor(int(self.y[item]), dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
        
        return data


# ---------------------------------------------------------------------
# Lightning DataModule (single HDF5 file per split)
# ---------------------------------------------------------------------
class JetClassLightningDataModule(LightningDataModule):
    """
    Optimized Lightning DataModule for JetClass single-file datasets.
    """

    def __init__(
        self,
        train_files: str,
        val_files: str,
        test_files: str,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
    ):
        super().__init__()
        self.train_file = train_files
        self.val_file = val_files
        self.test_file = test_files

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.save_hyperparameters()

    # -----------------------------------------------------------------
    # Setup datasets
    # -----------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = JetClassDenseDataset(
                self.train_file
            )
            self.val_dataset = JetClassDenseDataset(
                self.val_file
            )

        if stage == "test" or stage is None:
            self.test_dataset = JetClassDenseDataset(
                self.test_file
            )

    # -----------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

