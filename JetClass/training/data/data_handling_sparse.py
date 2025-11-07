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
    Returns a PyG Data object with the following attributes:
      - x: torch.FloatTensor [n_particles, n_node_features]
      - edge_index: torch.LongTensor [2, E]
      - edge_attr: torch.FloatTensor [E, n_edge_features]
      - y: torch.LongTensor [1]
    """

    def __init__(self, h5_path: str, filter_zero_edges: bool = True):
        """
        Args:
            h5_path: Path to HDF5 file
            filter_zero_edges: If True, remove edges with all-zero features
        """
        super().__init__()
        self.h5 = h5py.File(h5_path, "r", swmr=True)
        self.n_particles = self.h5["n_particles"]
        self.x = self.h5["feature_matrix"]
        self.edge = self.h5["adjacency_matrix"]
        self.y = self.h5["labels"]
        self.filter_zero_edges = filter_zero_edges

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item: int):
        node_features = torch.as_tensor(self.x[item], dtype=torch.float32)
        edge_features = torch.as_tensor(self.edge[item], dtype=torch.float32)
        n_particles = int(self.n_particles[item])

        # Generate all possible upper triangular indices
        i, j = torch.triu_indices(n_particles, n_particles, offset=1)
        
        # Create edge_index
        edge_index = torch.stack([i, j], dim=0)  # [2, num_possible_edges]
        
        # CRITICAL FIX: The number of edges stored might not match n_particles*(n_particles-1)/2
        # We need to handle the actual number of edges in the HDF5 file
        num_possible_edges = edge_index.shape[1]
        num_stored_edges = edge_features.shape[0]
        
        if num_stored_edges < num_possible_edges:
            # Pad edge features with zeros if fewer edges are stored
            padding = torch.zeros(num_possible_edges - num_stored_edges, edge_features.shape[1])
            edge_attr = torch.cat([edge_features, padding], dim=0)
        else:
            # Take only the first num_possible_edges
            edge_attr = edge_features[:num_possible_edges]
        
        # Optional: Filter out edges with all-zero features
        if self.filter_zero_edges:
            # Keep only edges where at least one feature is non-zero
            edge_mask = (edge_attr.abs().sum(dim=1) > 1e-6)
            edge_index = edge_index[:, edge_mask]
            edge_attr = edge_attr[edge_mask]
        
        label = torch.tensor(int(self.y[item]), dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
        
        return data


# Alternative implementation if your HDF5 stores edges differently
class JetClassSparseDataset(Dataset):
    """
    Alternative dataset that assumes edges are stored in a specific format.
    Use this if you know the exact storage format of your HDF5 files.
    """

    def __init__(self, h5_path: str, filter_zero_edges: bool = True, edge_threshold: float = 1e-6):
        """
        Args:
            h5_path: Path to HDF5 file
            filter_zero_edges: If True, remove edges with magnitude below threshold
            edge_threshold: Minimum feature magnitude to consider an edge valid
        """
        super().__init__()
        self.h5 = h5py.File(h5_path, "r", swmr=True)
        self.n_particles = self.h5["n_particles"]
        self.x = self.h5["feature_matrix"]
        self.edge = self.h5["adjacency_matrix"]
        self.y = self.h5["labels"]
        self.filter_zero_edges = filter_zero_edges
        self.edge_threshold = edge_threshold if filter_zero_edges else 0.0

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item: int):
        node_features = torch.as_tensor(self.x[item], dtype=torch.float32)
        edge_features = torch.as_tensor(self.edge[item], dtype=torch.float32)
        n_particles = int(self.n_particles[item])

        # Calculate expected number of edges for upper triangle
        num_expected_edges = n_particles * (n_particles - 1) // 2
        
        # Get stored edges (might be padded with zeros)
        stored_edges = edge_features.shape[0]
        
        # Create edge indices for the actual number of stored edges
        if stored_edges >= num_expected_edges:
            # Standard case: use all expected edges
            i, j = torch.triu_indices(n_particles, n_particles, offset=1)
            edge_index = torch.stack([i, j], dim=0)
            edge_attr = edge_features[:num_expected_edges]
        else:
            # Edge case: fewer edges stored than expected
            # This shouldn't happen if data is formatted correctly
            i, j = torch.triu_indices(n_particles, n_particles, offset=1)
            edge_index = torch.stack([i, j], dim=0)[:, :stored_edges]
            edge_attr = edge_features
        
        # Filter out edges with near-zero features
        if self.edge_threshold > 0:
            edge_magnitude = edge_attr.abs().sum(dim=1)
            valid_edges = edge_magnitude > self.edge_threshold
            edge_index = edge_index[:, valid_edges]
            edge_attr = edge_attr[valid_edges]
        
        label = torch.tensor(int(self.y[item]), dtype=torch.long)
        data = Data(
            x=node_features[:n_particles],  # Trim to actual particles
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label
        )
        
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
        filter_zero_edges: bool = True,  # NEW: filter zero edges
        dataset_class: str = "dense",  # "dense" or "sparse"
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
        self.filter_zero_edges = filter_zero_edges
        
        # Choose dataset implementation
        self.dataset_cls = JetClassDenseDataset if dataset_class == "dense" else JetClassSparseDataset

        self.save_hyperparameters()

    # -----------------------------------------------------------------
    # Setup datasets
    # -----------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                self.train_file,
                filter_zero_edges=self.filter_zero_edges
            )
            self.val_dataset = self.dataset_cls(
                self.val_file,
                filter_zero_edges=self.filter_zero_edges
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(
                self.test_file,
                filter_zero_edges=self.filter_zero_edges
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