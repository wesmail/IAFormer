#!/usr/bin/env python3
"""
Optimized Multi-file JetClass ROOT to HDF5 converter.
Optimizations: vectorized operations, efficient chunking, reduced copies.
"""

import sys
import math
import glob
import h5py
import numpy as np
import uproot
import awkward as ak
import vector
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class JetClassMultiFileGraphBuilder:
    """Convert multiple JetClass ROOT files to a single HDF5 file with optimized I/O."""
    
    def __init__(
        self,
        root_paths: List[str],
        tree_name: str = "tree",
        max_particles: int = 100,
        epsilon: float = 1e-5,
        label_order: Optional[List[str]] = None,
        compression: str = "lzf",
        chunk_size: int = 16384,  # Increased default
    ):
        # Expand globs and sort for deterministic processing
        self.root_paths = []
        for path in root_paths:
            if '*' in path or '?' in path:
                self.root_paths.extend(glob.glob(path))
            else:
                self.root_paths.append(path)
        
        self.root_paths = sorted(self.root_paths)
        
        if not self.root_paths:
            raise ValueError("No ROOT files found matching the provided paths")
        
        self.tree_name = tree_name
        self.max_particles = int(max_particles)
        self.epsilon = np.float32(epsilon)
        self.compression = compression
        self.chunk_size = chunk_size
        
        # Default label order
        default_label_order = [
            "QCD", "Hbb", "Hcc", "Hgg", "H4q",
            "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
        ]
        self.label_order = label_order or default_label_order
        
        # Precompute number of upper-triangle pairs
        self.pad_max_pairs = self.max_particles * (self.max_particles - 1) // 2
        
        # Feature names for metadata
        self.feature_names = [
            "E", "px", "py", "pz", 
            "log_pt", "log_E", "log_pt_rel", "log_E_rel",
            "d_eta", "d_phi", "d_r"
        ]
        self.edge_feature_names = [
            "log_deltaR", "log_kT", "log_z", 
            "log_m2", "log_pt_sum_rel", "log_energy_sum_rel"
        ]
    
    def _safe_log(self, x):
        """Safe logarithm with epsilon for numerical stability."""
        return np.log(ak.where(x <= 0, self.epsilon, x))
    
    def _to_numpy_padded(self, arr, target, fill=0.0, dtype=np.float32):
        """Convert jagged array to padded numpy array - optimized version."""
        padded = ak.pad_none(arr, target=target, axis=1, clip=True)
        filled = ak.fill_none(padded, fill)
        return ak.to_numpy(filled, allow_missing=False).astype(dtype, copy=False)
    
    def _count_total_events(self) -> Tuple[int, List[Tuple[str, int]]]:
        """Count total events across all ROOT files."""
        total = 0
        file_info = []
        
        print(f"Counting events in {len(self.root_paths)} file(s)...")
        for path in tqdm(self.root_paths, desc="Scanning files", unit="file"):
            try:
                with uproot.open(path) as f:
                    n = f[self.tree_name].num_entries
                    total += n
                    file_info.append((path, n))
            except Exception as e:
                print(f"\nWarning: Could not read {path}: {e}", file=sys.stderr)
                continue
        
        print(f"Total: {total:,} events across {len(file_info)} readable file(s)")
        return total, file_info
    
    def _create_datasets(self, h5file: h5py.File, n_events: int) -> dict:
        """Create HDF5 datasets with optimal chunking and compression."""
        datasets = {}
        
        print(f"Pre-allocating datasets for {n_events:,} events")
        
        # Optimize chunk size for I/O performance
        # Rule of thumb: chunk size should be 10KB - 1MB in each dimension
        opt_chunk_size = min(self.chunk_size, max(1024, n_events // 10))
        
        # Feature matrix: (N, max_particles, 11)
        datasets['X'] = h5file.create_dataset(
            "feature_matrix",
            shape=(n_events, self.max_particles, 11),
            dtype="float32",
            chunks=(opt_chunk_size, self.max_particles, 11),
            compression=self.compression if self.compression != "none" else None,
            shuffle=True if self.compression == "gzip" else False,  # Better compression
        )
        
        # Adjacency matrix: (N, max_pairs, 6)
        datasets['E'] = h5file.create_dataset(
            "adjacency_matrix",
            shape=(n_events, self.pad_max_pairs, 6),
            dtype="float32",
            chunks=(opt_chunk_size, self.pad_max_pairs, 6),
            compression=self.compression if self.compression != "none" else None,
            shuffle=True if self.compression == "gzip" else False,
        )
        
        # Labels: (N,)
        datasets['y'] = h5file.create_dataset(
            "labels",
            shape=(n_events,),
            dtype="int16",
            chunks=(opt_chunk_size,),
            compression=self.compression if self.compression != "none" else None,
        )
        
        # Number of particles per jet: (N,)
        datasets['n_particles'] = h5file.create_dataset(
            "n_particles",
            shape=(n_events,),
            dtype="int16",
            chunks=(opt_chunk_size,),
            compression=self.compression if self.compression != "none" else None,
        )
        
        # Optional index for PyTorch shuffling: (N,)
        datasets['index'] = h5file.create_dataset(
            "index",
            shape=(n_events,),
            dtype="int64",
            chunks=(opt_chunk_size,),
            compression=self.compression if self.compression != "none" else None,
        )
        
        return datasets
    
    def _add_metadata(self, h5file: h5py.File, file_info: List[Tuple[str, int]]):
        """Add metadata as file-level attributes."""
        h5file.attrs['max_particles'] = self.max_particles
        h5file.attrs['pad_max_pairs'] = self.pad_max_pairs
        h5file.attrs['label_order'] = self.label_order
        h5file.attrs['feature_names'] = self.feature_names
        h5file.attrs['edge_feature_names'] = self.edge_feature_names
        h5file.attrs['source_files'] = [str(Path(f).name) for f, _ in file_info]
        h5file.attrs['compression'] = self.compression
        h5file.attrs['chunk_size'] = self.chunk_size
        h5file.attrs['events_per_file'] = [n for _, n in file_info]
    
    def _process_chunk(self, arrays) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process a chunk of events - optimized with fewer copies."""
        # Build 4-vectors
        p4 = vector.zip({
            "px": arrays["part_px"],
            "py": arrays["part_py"],
            "pz": arrays["part_pz"],
            "E": arrays["part_energy"],
        })
        
        # Get number of particles per jet
        n_particles = ak.num(p4, axis=1)
        
        # Jet variables (broadcast once)
        jet_pt = arrays["jet_pt"]
        jet_eta = arrays["jet_eta"]
        jet_phi = arrays["jet_phi"]
        jet_E = arrays["jet_energy"]
        
        # Node features - compute in bulk
        d_eta = p4.eta - jet_eta
        d_phi = p4.phi - jet_phi
        d_r = np.sqrt(d_eta**2 + d_phi**2)
        
        log_pt = self._safe_log(p4.pt)
        log_E = self._safe_log(p4.E)
        log_pt_rel = self._safe_log(p4.pt / jet_pt)
        log_E_rel = self._safe_log(p4.E / jet_E)
        
        # Pad and stack in one go - reduced function calls
        X = np.stack([
            self._to_numpy_padded(p4.E, self.max_particles),
            self._to_numpy_padded(p4.px, self.max_particles),
            self._to_numpy_padded(p4.py, self.max_particles),
            self._to_numpy_padded(p4.pz, self.max_particles),
            self._to_numpy_padded(log_pt, self.max_particles),
            self._to_numpy_padded(log_E, self.max_particles),
            self._to_numpy_padded(log_pt_rel, self.max_particles),
            self._to_numpy_padded(log_E_rel, self.max_particles),
            self._to_numpy_padded(d_eta, self.max_particles),
            self._to_numpy_padded(d_phi, self.max_particles),
            self._to_numpy_padded(d_r, self.max_particles),
        ], axis=-1)
        
        # Edge features - compute combinations once
        p1, p2 = ak.unzip(ak.combinations(p4, 2, axis=1))
        
        # Precompute reused values
        dR = p1.deltaR(p2)
        p1_pt = p1.pt
        p2_pt = p2.pt
        pt_min = ak.where(p1_pt < p2_pt, p1_pt, p2_pt)  # Faster than ak.min
        pt_sum = p1_pt + p2_pt
        E_sum = p1.E + p2.E
        p_sum = p1 + p2
        p4_sum = ak.sum(p4, axis=1)  # Compute once
        
        # Edge features - vectorized
        deltaR = self._safe_log(dR)
        kT = self._safe_log(pt_min * dR)
        z = self._safe_log(pt_min / pt_sum)
        m2 = self._safe_log(E_sum**2 - p_sum.mag2)
        pt_sum_rel = self._safe_log(pt_sum / p4_sum.pt)
        energy_sum_rel = self._safe_log(E_sum / p4_sum.energy)
        
        # Stack edge features
        E = np.stack([
            self._to_numpy_padded(deltaR, self.pad_max_pairs, fill=-1000.0),
            self._to_numpy_padded(kT, self.pad_max_pairs, fill=-1000.0),
            self._to_numpy_padded(z, self.pad_max_pairs, fill=-1000.0),
            self._to_numpy_padded(m2, self.pad_max_pairs, fill=-1000.0),
            self._to_numpy_padded(pt_sum_rel, self.pad_max_pairs, fill=-1000.0),
            self._to_numpy_padded(energy_sum_rel, self.pad_max_pairs, fill=-1000.0),
        ], axis=-1)
        
        # Labels - direct argmax
        label_mat = np.stack([ak.to_numpy(arrays[f"label_{k}"]) for k in self.label_order], axis=1)
        y = np.argmax(label_mat, axis=1).astype(np.int16)
        
        # Number of particles
        n_parts = ak.to_numpy(n_particles).astype(np.int16)
        
        return X, E, y, n_parts
    
    def convert(self, out_h5: str):
        """Convert multiple ROOT files to single HDF5 with optimal I/O."""
        # Count total events
        try:
            n_total, file_info = self._count_total_events()
        except Exception as e:
            print(f"Error counting events: {e}", file=sys.stderr)
            raise
        
        if n_total == 0:
            raise ValueError("No events found in input files")
        
        # Prepare output
        out_path = Path(out_h5)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Branches to read
        branches = [
            "part_px", "part_py", "part_pz", "part_energy",
            "jet_pt", "jet_eta", "jet_phi", "jet_energy",
        ] + [f"label_{k}" for k in self.label_order]
        
        # Process files
        with h5py.File(out_path, "w", libver='latest') as h5:  # libver for better performance
            # Create datasets
            datasets = self._create_datasets(h5, n_total)
            
            # Add metadata
            self._add_metadata(h5, file_info)
            
            write_ptr = 0
            event_idx = 0
            
            # Overall progress bar
            overall_pbar = tqdm(
                total=n_total, 
                desc="Overall progress", 
                unit="events",
                position=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            # Process each file
            for file_idx, root_path in enumerate(self.root_paths):
                file_name = Path(root_path).name
                expected_events = next((n for p, n in file_info if p == root_path), 0)
                
                file_pbar = tqdm(
                    total=expected_events,
                    desc=f"File [{file_idx+1}/{len(self.root_paths)}]: {file_name}",
                    unit="events",
                    position=1,
                    leave=False,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
                )
                
                try:
                    # Iterate in chunks with uproot optimization
                    for arrays in uproot.iterate(
                        f"{root_path}:{self.tree_name}",
                        step_size=self.chunk_size,
                        filter_name=branches,
                        library="ak",
                        ak_add_doc=False,  # Skip docstrings for speed
                    ):
                        # Process chunk
                        X, E, y, n_parts = self._process_chunk(arrays)
                        n = X.shape[0]
                        
                        # Write data in one operation per dataset
                        end_ptr = write_ptr + n
                        datasets['X'][write_ptr:end_ptr] = X
                        datasets['E'][write_ptr:end_ptr] = E
                        datasets['y'][write_ptr:end_ptr] = y
                        datasets['n_particles'][write_ptr:end_ptr] = n_parts
                        datasets['index'][write_ptr:end_ptr] = np.arange(event_idx, event_idx + n, dtype=np.int64)
                        
                        write_ptr = end_ptr
                        event_idx += n
                        
                        # Update progress bars
                        file_pbar.update(n)
                        overall_pbar.update(n)
                
                except Exception as e:
                    tqdm.write(f"Warning: Error processing {root_path}: {e}", file=sys.stderr)
                    continue
                finally:
                    file_pbar.close()
            
            overall_pbar.close()
            
            # Update total events in metadata
            h5.attrs['total_events'] = write_ptr
            
            # Force flush to disk
            h5.flush()
        
        print(f"\n✓ Successfully wrote {write_ptr:,} events → {out_path}")
        print(f"  File size: {out_path.stat().st_size / 1e9:.2f} GB")
        print(f"  Compression: {self.compression}")
        print(f"  Chunk size: {self.chunk_size:,} events")


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple JetClass ROOT files to single HDF5 (optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--roots", 
        nargs='+', 
        required=True,
        help="Paths to ROOT files (supports globs like '*.root')"
    )
    parser.add_argument(
        "--out", 
        required=True,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--tree", 
        default="tree",
        help="Name of the tree in ROOT files"
    )
    parser.add_argument(
        "--max_particles", 
        type=int, 
        default=100,
        help="Maximum number of particles per jet"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=16384,  # Increased default
        help="Number of events to process per chunk"
    )
    parser.add_argument(
        "--compression", 
        default="lzf",
        choices=["lzf", "gzip", "szip", "none"],
        help="HDF5 compression algorithm"
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = JetClassMultiFileGraphBuilder(
        root_paths=args.roots,
        tree_name=args.tree,
        max_particles=args.max_particles,
        compression=args.compression,
        chunk_size=args.chunk_size,
    )
    
    # Convert files
    converter.convert(args.out)


if __name__ == "__main__":
    main()
