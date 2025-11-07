import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from lightning.pytorch import LightningDataModule

# ---------------------------------------------------------------------
# Helper: Convert upper-triangle flat edge list → dense symmetric matrix
# ---------------------------------------------------------------------
def upper_to_dense(E_flat, max_particles, n_edge_features):
    i, j = torch.triu_indices(max_particles, max_particles, offset=1)
    dense = torch.zeros((max_particles, max_particles, n_edge_features),
                        dtype=E_flat.dtype, device=E_flat.device)
    dense[i, j] = E_flat
    dense[j, i] = E_flat
    return dense


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

    def __init__(self, h5_path: str, reconstruct_full_adjacency: bool = True):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r", swmr=True)
        self.X = self.h5["feature_matrix"]
        self.E = self.h5["adjacency_matrix"]
        self.y = self.h5["labels"]

        self.max_particles = int(self.h5.attrs["max_particles"])
        self.n_node_features = self.X.shape[2]
        self.n_edge_features = self.E.shape[2]
        self.reconstruct_full_adjacency = reconstruct_full_adjacency

        # Detect adjacency type
        if "adjacency_type" in self.h5.attrs:
            self.adj_type = self.h5.attrs["adjacency_type"]
        else:
            M = self.E.shape[1]
            self.adj_type = (
                "full"
                if M == self.max_particles * (self.max_particles - 1)
                else "upper"
            )

        # Precompute indices once (reused for all samples)
        self.triu_i, self.triu_j = torch.triu_indices(
            self.max_particles, self.max_particles, offset=1
        )

        if self.adj_type == "full":
            i = torch.arange(self.max_particles)
            I, J = torch.meshgrid(i, i, indexing='ij')
            mask = I != J
            self.src, self.dst = I[mask], J[mask]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        node_features = torch.as_tensor(self.X[idx], dtype=torch.float32)
        E_flat = torch.as_tensor(self.E[idx], dtype=torch.float32)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)

        # Reconstruct dense adjacency
        if self.adj_type == "upper":
            edge_features = torch.zeros(
                (self.max_particles, self.max_particles, self.n_edge_features),
                dtype=torch.float32,
            )
            edge_features[self.triu_i, self.triu_j] = E_flat
            edge_features[self.triu_j, self.triu_i] = E_flat
        else:
            edge_features = torch.zeros(
                (self.max_particles, self.max_particles, self.n_edge_features),
                dtype=torch.float32,
            )
            edge_features[self.src, self.dst] = E_flat
            if self.reconstruct_full_adjacency:
                edge_features = torch.maximum(edge_features, edge_features.transpose(0, 1))

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'labels': label,
        }


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
        reconstruct_full_adjacency: bool = True,
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
        self.reconstruct_full_adjacency = reconstruct_full_adjacency

        self.save_hyperparameters()

    # -----------------------------------------------------------------
    # Setup datasets
    # -----------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = JetClassDenseDataset(
                self.train_file, reconstruct_full_adjacency=self.reconstruct_full_adjacency
            )
            self.val_dataset = JetClassDenseDataset(
                self.val_file, reconstruct_full_adjacency=self.reconstruct_full_adjacency
            )

        if stage == "test" or stage is None:
            self.test_dataset = JetClassDenseDataset(
                self.test_file, reconstruct_full_adjacency=self.reconstruct_full_adjacency
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

