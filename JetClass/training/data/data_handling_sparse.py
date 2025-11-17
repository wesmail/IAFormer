import torch
import h5py
from typing import Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from functools import lru_cache


class JetClassOptimizedDataset(Dataset):
    """
    Optimized dataset that:
    1. Opens ONE file at a time per worker (closes after reading)
    2. Caches edge index generation (triu_indices)
    3. Minimizes memory allocations
    4. No file handle accumulation
    """

    def __init__(
        self,
        h5_paths: List[str],
        cache_file_sizes: bool = True,
        max_cache_size: int = 256,  # Cache edge indices for common sizes
    ):
        super().__init__()
        self.h5_paths = sorted(h5_paths)
        self.max_cache_size = max_cache_size
        
        # Pre-compute cumulative indices
        self._setup_file_boundaries(cache_file_sizes)
        
        # Cache for edge indices (per n_nodes)
        self._edge_index_cache = {}

    def _setup_file_boundaries(self, cache_file_sizes: bool):
        """Pre-compute file boundaries."""
        cache_path = self._get_cache_path()

        if cache_file_sizes and cache_path.exists():
            cache = np.load(cache_path)
            self.file_sizes = cache["file_sizes"].tolist()
            self.cumulative_sizes = cache["cumulative_sizes"].tolist()
            print(f"✅ Loaded cached file boundaries from {cache_path}")
        else:
            print("Computing file boundaries (one-time)...")
            self.file_sizes = []
            for h5_path in self.h5_paths:
                with h5py.File(h5_path, "r") as f:
                    self.file_sizes.append(len(f["labels"]))

            self.cumulative_sizes = np.cumsum([0] + self.file_sizes).tolist()

            if cache_file_sizes:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    file_sizes=np.array(self.file_sizes),
                    cumulative_sizes=np.array(self.cumulative_sizes),
                )
                print(f"✅ Cached file boundaries to {cache_path}")

        self.total_size = self.cumulative_sizes[-1]
        print(f"📊 Total samples: {self.total_size:,} across {len(self.h5_paths)} files")

    def _get_cache_path(self) -> Path:
        """Generate cache path based on dataset files."""
        import hashlib
        paths_str = "".join(sorted(self.h5_paths))
        hash_str = hashlib.md5(paths_str.encode()).hexdigest()[:8]
        return Path(f"/tmp/jetclass_opt_cache_{hash_str}.npz")

    def _find_file_and_local_idx(self, global_idx: int) -> Tuple[int, int]:
        """Binary search for file containing global_idx."""
        left, right = 0, len(self.cumulative_sizes) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if self.cumulative_sizes[mid] <= global_idx:
                left = mid
            else:
                right = mid

        file_idx = left
        local_idx = global_idx - self.cumulative_sizes[file_idx]
        return file_idx, local_idx

    @staticmethod
    def infer_num_particles_from_pairs(pairs: int) -> int:
        """Infer number of particles from number of pairs."""
        import math
        discriminant = 1 + 8 * pairs
        n = int((-1 + math.sqrt(discriminant)) / 2)
        return n + 1

    def _get_edge_index_upper(self, n_nodes: int) -> torch.Tensor:
        """
        Get upper triangular edge indices with caching.
        This eliminates the torch.triu_indices bottleneck!
        """
        if n_nodes in self._edge_index_cache:
            return self._edge_index_cache[n_nodes].clone()
        
        # Only cache if within reasonable size
        if len(self._edge_index_cache) < self.max_cache_size:
            i, j = torch.triu_indices(n_nodes, n_nodes, offset=1)
            edge_index = torch.stack([i, j], dim=0)
            self._edge_index_cache[n_nodes] = edge_index
            return edge_index.clone()
        else:
            # Don't cache if too many different sizes
            i, j = torch.triu_indices(n_nodes, n_nodes, offset=1)
            return torch.stack([i, j], dim=0)

    def __len__(self):
        return self.total_size

    def __getitem__(self, global_idx: int) -> Data:
        """
        Load sample with optimizations:
        - Open file, read, close immediately
        - Use cached edge indices
        - Minimal allocations
        """
        # Find which file
        file_idx, local_idx = self._find_file_and_local_idx(global_idx)

        # 🔥 KEY CHANGE: Open file, read, close immediately
        # This prevents file handle accumulation!
        with h5py.File(self.h5_paths[file_idx], "r", rdcc_nbytes=1024**2*100) as h5_file:
            # Load raw data
            node_features = torch.as_tensor(
                h5_file["node_features"][local_idx],
                dtype=torch.float32,
            )
            edge_features = torch.as_tensor(
                h5_file["adjacency_matrix"][local_idx],
                dtype=torch.float32,
            )
            n_particles_raw = int(h5_file["n_particles"][local_idx])
            label = torch.tensor(int(h5_file["labels"][local_idx]), dtype=torch.long)

        # Infer n_nodes
        max_pairs = edge_features.shape[0]
        max_particles_from_pairs = self.infer_num_particles_from_pairs(max_pairs)
        n_nodes = min(n_particles_raw, node_features.shape[0], max_particles_from_pairs)

        if n_nodes <= 1:
            # Degenerate case
            return Data(
                x=node_features[:1],
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, edge_features.shape[1]), dtype=torch.float32),
                y=label,
            )

        # Trim nodes
        node_features = node_features[:n_nodes]

        # 🔥 OPTIMIZED: Use cached edge indices
        num_expected_edges = n_nodes * (n_nodes - 1) // 2
        edge_index_upper = self._get_edge_index_upper(n_nodes)
        edge_attr_upper = edge_features[:num_expected_edges]

        # Make undirected
        edge_index = torch.cat([edge_index_upper, edge_index_upper.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr_upper, edge_attr_upper], dim=0)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label,
        )


class JetClassLightningDataModule(LightningDataModule):
    """
    Optimized Lightning DataModule with:
    - One file at a time access
    - Aggressive file handle management
    - Memory-efficient settings
    """

    def __init__(
        self,
        train_files: Union[List[str], str],
        val_files: Union[List[str], str],
        test_files: Optional[Union[List[str], str]] = None,
        batch_size: int = 256,
        num_workers: int = 4,  # Reduced default
        pin_memory: bool = True,
        persistent_workers: bool = False,  # 🔥 CHANGED: Disable to allow worker cleanup
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.train_files = train_files if isinstance(train_files, list) else [train_files]
        self.val_files = val_files if isinstance(val_files, list) else [val_files]
        self.test_files = test_files if test_files is None or isinstance(test_files, list) else [test_files]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            print("🔧 Setting up optimized training dataset...")
            self.train_dataset = JetClassOptimizedDataset(self.train_files)
            print("🔧 Setting up optimized validation dataset...")
            self.val_dataset = JetClassOptimizedDataset(self.val_files)

        if stage == "test" and self.test_files is not None:
            self.test_dataset = JetClassOptimizedDataset(self.test_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,  # Helps with consistency
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
