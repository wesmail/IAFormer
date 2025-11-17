import torch
import math
import h5py
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from lightning.pytorch import LightningDataModule
import logging
import os
import hashlib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Efficient Single-File Dataset with Lazy Loading (dense adjacency)
# ---------------------------------------------------------------------
class JetClassH5Dataset(Dataset):
    """
    Efficient single HDF5 file dataset with lazy loading.
    Opens file only when needed and keeps it open during iteration.
    """

    def __init__(
        self,
        h5_path: str,
        reconstruct_full_adjacency: bool = True,
        cache_metadata: bool = True,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.reconstruct_full_adjacency = reconstruct_full_adjacency
        self.h5 = None
        self._length = None
        self._metadata = {}

        # Cache metadata on initialization (lightweight)
        if cache_metadata:
            self._load_metadata()

    def _load_metadata(self):
        """Load metadata without keeping file open."""
        with h5py.File(self.h5_path, "r") as f:
            self._length = f["feature_matrix"].shape[0]
            self._metadata["max_particles"] = int(f.attrs["max_particles"])
            self._metadata["n_node_features"] = f["feature_matrix"].shape[2]
            self._metadata["n_edge_features"] = f["adjacency_matrix"].shape[2]

            # Detect adjacency type
            if "adjacency_type" in f.attrs:
                self._metadata["adj_type"] = f.attrs["adjacency_type"]
            else:
                M = f["adjacency_matrix"].shape[1]
                max_p = self._metadata["max_particles"]
                self._metadata["adj_type"] = (
                    "full" if M == max_p * (max_p - 1) else "upper"
                )

        # Precompute indices
        max_p = self._metadata["max_particles"]
        self.triu_i, self.triu_j = torch.triu_indices(max_p, max_p, offset=1)

        if self._metadata["adj_type"] == "full":
            i = torch.arange(max_p)
            I, J = torch.meshgrid(i, i, indexing='ij')
            mask = I != J
            self.src, self.dst = I[mask], J[mask]

    def _open_file(self):
        """Open HDF5 file if not already open."""
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r", swmr=True, libver='latest')
            self.X = self.h5["feature_matrix"]
            self.E = self.h5["adjacency_matrix"]
            self.y = self.h5["labels"]

    def __len__(self):
        if self._length is None:
            self._load_metadata()
        return self._length

    def __getitem__(self, idx):
        self._open_file()

        # Read data
        node_features = torch.as_tensor(self.X[idx], dtype=torch.float32)
        E_flat = torch.as_tensor(self.E[idx], dtype=torch.float32)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)

        # Reconstruct dense adjacency
        max_p = self._metadata["max_particles"]
        n_edge = self._metadata["n_edge_features"]

        # (max_p, max_p, n_edge)
        edge_features = torch.zeros((max_p, max_p, n_edge), dtype=torch.float32)

        if self._metadata["adj_type"] == "upper":
            # E_flat: (#upper_edges, n_edge)
            edge_features[self.triu_i, self.triu_j] = E_flat
            edge_features[self.triu_j, self.triu_i] = E_flat
        else:
            # E_flat: (#off_diagonal_edges, n_edge) for all ordered pairs i != j
            edge_features[self.src, self.dst] = E_flat
            if self.reconstruct_full_adjacency:
                # Make symmetric by taking max of (i,j) and (j,i)
                edge_features = torch.maximum(
                    edge_features, edge_features.transpose(0, 1)
                )

        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'labels': label,
        }

    def __del__(self):
        """Clean up file handle."""
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            # Avoid errors during interpreter shutdown
            pass


# ---------------------------------------------------------------------
# Multi-File Dataset with Cached File Sizes + Lazy Per-File Datasets
# ---------------------------------------------------------------------
class MultiFileJetClassDataset(Dataset):
    """
    Efficient multi-file dataset that concatenates multiple HDF5 files.

    Improvements:
    - Lazy loading: files/datasets opened only when needed
    - Cached file sizes + cumulative boundaries in /tmp
    - O(log N_files) mapping from global index to (file_idx, local_idx)
    """

    def __init__(
        self,
        file_paths: Union[List[str], str],
        reconstruct_full_adjacency: bool = True,
        cache_metadata: bool = True,
        cache_file_sizes: bool = True,
    ):
        super().__init__()

        # Handle single file or list of files
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Expand glob patterns if provided
        expanded_paths = []
        for path_pattern in file_paths:
            path = Path(path_pattern)
            if '*' in str(path):
                # Glob pattern
                expanded_paths.extend(sorted(path.parent.glob(path.name)))
            else:
                expanded_paths.append(path)

        self.file_paths = [str(p) for p in expanded_paths]

        if len(self.file_paths) == 0:
            raise ValueError(f"No files found matching patterns: {file_paths}")

        logger.info(f"Using {len(self.file_paths)} HDF5 files...")

        self.reconstruct_full_adjacency = reconstruct_full_adjacency
        self.cache_metadata = cache_metadata
        self.cache_file_sizes = cache_file_sizes

        # Cached per-file datasets (created lazily)
        self.datasets: List[Optional[JetClassH5Dataset]] = [None] * len(self.file_paths)

        # Precompute file sizes + cumulative boundaries (with disk cache)
        self._setup_file_boundaries()

        logger.info(f"Total samples: {self.total_length:,}")
        logger.info(f"Samples per file: {self.file_sizes}")

    # ---------- cache helpers ----------
    def _get_cache_path(self) -> Path:
        """Generate cache path based on the list of dataset files."""
        paths_str = "".join(sorted(self.file_paths))
        hash_str = hashlib.md5(paths_str.encode()).hexdigest()[:8]
        return Path(f"/tmp/jetclass_multifile_{hash_str}.npz")

    def _setup_file_boundaries(self):
        """
        Pre-compute file_sizes and cumulative_lengths.
        Cached to /tmp so repeated runs don't reopen all HDF5 files.
        """
        cache_path = self._get_cache_path()

        if self.cache_file_sizes and cache_path.exists():
            logger.info(f"Loading cached file boundaries from {cache_path}")
            cache = np.load(cache_path)
            self.file_sizes = cache["file_sizes"].tolist()
            self.cumulative_lengths = cache["cumulative_lengths"].tolist()
        else:
            logger.info("Computing file boundaries (this may take a moment)...")
            self.file_sizes = []
            for fp in self.file_paths:
                with h5py.File(fp, "r") as f:
                    # Number of samples = length of feature_matrix / labels
                    self.file_sizes.append(f["feature_matrix"].shape[0])

            # cumulative_lengths[i] = total samples up to and including file i
            self.cumulative_lengths = []
            cumsum = 0
            for size in self.file_sizes:
                cumsum += size
                self.cumulative_lengths.append(cumsum)

            if self.cache_file_sizes:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_path,
                    file_sizes=np.array(self.file_sizes, dtype=np.int64),
                    cumulative_lengths=np.array(self.cumulative_lengths, dtype=np.int64),
                )
                logger.info(f"Cached file boundaries to {cache_path}")

        self.total_length = int(self.cumulative_lengths[-1])

    # ---------- dataset access ----------
    def _get_dataset(self, file_idx: int) -> JetClassH5Dataset:
        """
        Lazily create JetClassH5Dataset for a given file index.
        Ensures each worker process has its own dataset instances.
        """
        ds = self.datasets[file_idx]
        if ds is None:
            ds = JetClassH5Dataset(
                self.file_paths[file_idx],
                reconstruct_full_adjacency=self.reconstruct_full_adjacency,
                cache_metadata=self.cache_metadata,
            )
            self.datasets[file_idx] = ds
        return ds

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

        # Binary search to find which file contains this index
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))

        # Compute local index within that file
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[file_idx - 1]

        ds = self._get_dataset(file_idx)
        return ds[local_idx]

    def get_file_idx(self, global_idx):
        """Get which file and local index for a global index."""
        file_idx = int(np.searchsorted(self.cumulative_lengths, global_idx, side='right'))
        if file_idx == 0:
            local_idx = global_idx
        else:
            local_idx = global_idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, local_idx


# ---------------------------------------------------------------------
# Optimized DataLoader Worker Init (for multi-processing)
# ---------------------------------------------------------------------
def worker_init_fn(worker_id):
    """
    Initialize worker to ensure each worker has its own HDF5 file handles.
    This prevents threading issues with HDF5.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Each worker gets a unique seed
        np.random.seed(worker_info.seed % (2**32))


class VariableLengthBatchSampler(Sampler):
    """
    Batch sampler that groups samples with similar particle counts
    to minimize padding inside a batch.

    Assumes underlying dataset is MultiFileJetClassDataset and that
    each HDF5 file has either:
      - a 'n_particles' dataset, or
      - padded 'feature_matrix' where zero rows = no particle.
    """

    def __init__(
        self,
        dataset: MultiFileJetClassDataset,
        batch_size: int,
        shuffle_batches: bool = True,
        cache_counts: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.cache_counts = cache_counts

        # ------------------------------------------------------------------
        # 1) Try to load cached counts
        # ------------------------------------------------------------------
        self.counts = self._load_or_compute_counts()

        # ------------------------------------------------------------------
        # 2) Sort indices by particle count
        # ------------------------------------------------------------------
        # counts[i] = n_particles for global index i
        indexed_counts = list(enumerate(self.counts))
        indexed_counts.sort(key=lambda x: x[1])  # sort by n_particles

        # ------------------------------------------------------------------
        # 3) Build batches of global indices
        # ------------------------------------------------------------------
        self.batches = []
        for i in range(0, len(indexed_counts), batch_size):
            batch = [idx for idx, _ in indexed_counts[i : i + batch_size]]
            self.batches.append(batch)

    # ---------- caching helpers ----------
    def _counts_cache_path(self) -> Path:
        """Use same file list hash as dataset, but different suffix."""
        paths_str = "".join(sorted(self.dataset.file_paths))
        hash_str = hashlib.md5(paths_str.encode()).hexdigest()[:8]
        return Path(f"/tmp/jetclass_counts_{hash_str}.npz")

    def _load_or_compute_counts(self):
        cache_path = self._counts_cache_path()
        if self.cache_counts and cache_path.exists():
            logger.info(f"Loading cached n_particles counts from {cache_path}")
            data = np.load(cache_path)
            counts = data["counts"].astype(np.int32)
            return counts

        logger.info("Computing n_particles for all samples (one-time cost)...")

        counts = np.zeros(self.dataset.total_length, dtype=np.int32)
        global_offset = 0

        # We assume MultiFileJetClassDataset has file_paths and file_sizes
        for fp, size in zip(self.dataset.file_paths, self.dataset.file_sizes):
            with h5py.File(fp, "r") as f:
                if "n_particles" in f:
                    # Fast path: directly read n_particles
                    n_parts = f["n_particles"][:]  # shape (size,)
                    counts[global_offset : global_offset + size] = n_parts.astype(np.int32)
                else:
                    # Fallback: infer from padded feature_matrix
                    X = f["feature_matrix"]  # (N, max_p, n_feat)
                    max_p = X.shape[1]

                    # process in chunks to avoid huge memory spikes
                    chunk = 1024
                    for start in range(0, size, chunk):
                        end = min(start + chunk, size)
                        X_chunk = X[start:end]  # (chunk, max_p, n_feat)
                        # non-zero rows = real particles
                        # sum over feature dim; >0 => real
                        mask = np.abs(X_chunk).sum(axis=-1) > 0  # (chunk, max_p)
                        cnt = mask.sum(axis=-1)                  # (chunk,)
                        counts[global_offset + start : global_offset + end] = cnt.astype(np.int32)

            global_offset += size

        if self.cache_counts:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, counts=counts)
            logger.info(f"Cached n_particles counts to {cache_path}")

        return counts

    # ---------- Sampler interface ----------
    def __iter__(self):
        # Optional: shuffle batch order each epoch
        if self.shuffle_batches:
            order = np.random.permutation(len(self.batches))
            for i in order:
                yield self.batches[i]
        else:
            for batch in self.batches:
                yield batch

    def __len__(self):
        return len(self.batches)

class MultiFileJetClassDataModule(LightningDataModule):
    def __init__(
        self,
        train_files: Union[List[str], str],
        val_files: Union[List[str], str],
        test_files: Optional[Union[List[str], str]] = None,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        reconstruct_full_adjacency: bool = True,
        cache_metadata: bool = True,
        cache_file_sizes: bool = True,
        use_variable_length_batches: bool = True,        # <--- NEW
        shuffle_vlbatches: bool = True,                  # <--- optional
    ):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        self.reconstruct_full_adjacency = reconstruct_full_adjacency
        self.cache_metadata = cache_metadata
        self.cache_file_sizes = cache_file_sizes
        self.use_variable_length_batches = use_variable_length_batches
        self.shuffle_vlbatches = shuffle_vlbatches

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            logger.info("Setting up training dataset...")
            self.train_dataset = MultiFileJetClassDataset(
                self.train_files,
                reconstruct_full_adjacency=self.reconstruct_full_adjacency,
                cache_metadata=self.cache_metadata,
                cache_file_sizes=self.cache_file_sizes,
            )

            logger.info("Setting up validation dataset...")
            self.val_dataset = MultiFileJetClassDataset(
                self.val_files,
                reconstruct_full_adjacency=self.reconstruct_full_adjacency,
                cache_metadata=self.cache_metadata,
                cache_file_sizes=self.cache_file_sizes,
            )

        if stage == "test" or stage is None:
            if self.test_files is not None:
                logger.info("Setting up test dataset...")
                self.test_dataset = MultiFileJetClassDataset(
                    self.test_files,
                    reconstruct_full_adjacency=self.reconstruct_full_adjacency,
                    cache_metadata=self.cache_metadata,
                    cache_file_sizes=self.cache_file_sizes,
                )

    def train_dataloader(self):
        if self.use_variable_length_batches:
            batch_sampler = VariableLengthBatchSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle_batches=self.shuffle_vlbatches,
                cache_counts=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
                drop_last=True,  # For stable batch norm
            )

    def val_dataloader(self):
        if self.use_variable_length_batches:
            # usually no shuffle for validation
            batch_sampler = VariableLengthBatchSampler(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle_batches=False,
                cache_counts=True,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=worker_init_fn,
            )                
