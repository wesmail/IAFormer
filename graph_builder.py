import sys
import h5py
import logging
import vector
import numpy as np
import pandas as pd
import awkward as ak

# from dask import delayed, compute

from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParticleGraphBuilder:
    def __init__(
        self,
        file_path: str,
        key: str = "table",
        max_particles: int = 200,
        chunk_size: int = 1000,
        max_num_chunks: int = -1,
    ):
        self.file_path = file_path
        self.key = key
        self.max_particles = max_particles
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        logging.info("Initialized ParticleGraphBuilder with file: %s", file_path)

    def _col_list(self, prefix):
        """Generate column names for a given prefix."""
        return [f"{prefix}_{i}" for i in range(self.max_particles)]

    def process_chunk(self, start, end):
        """Process a range of data to compute the 4-vectors, adjacency matrix, mask, and labels."""
        # logging.info("Processing chunk from row %d to row %d", start, end)

        df_chunk = pd.read_hdf(self.file_path, key=self.key, start=start, stop=end)

        # Process columns for the current chunk
        _px = df_chunk[self._col_list("PX")].values
        _py = df_chunk[self._col_list("PY")].values
        _pz = df_chunk[self._col_list("PZ")].values
        _e = df_chunk[self._col_list("E")].values

        mask = _e > 0
        n_particles = np.sum(mask, axis=1)
        max_particles_chunk = np.max(n_particles)
        max_particles = min(self.max_particles, max_particles_chunk)

        px = ak.unflatten(_px[mask], n_particles)
        py = ak.unflatten(_py[mask], n_particles)
        pz = ak.unflatten(_pz[mask], n_particles)
        e = ak.unflatten(_e[mask], n_particles)

        p4_chunk = vector.zip({"px": px, "py": py, "pz": pz, "energy": e})

        # Calculate Particle Interactions
        # Step 1: Compute pairwise deltaR using ak.combinations
        p1, p2 = ak.unzip(ak.combinations(p4_chunk, 2, axis=1))
        # Step 2: Calculate pairwise ∆R = sqrt((Δy)^2 + (Δφ)^2)
        delta = p1.deltaR(p2)
        # Step 3: Calculate k_T # min(p_T,a, p_T,b)
        pt_min = ak.min([p1.pt, p2.pt], axis=0)
        k_T = pt_min * delta
        pt_sum = p1.pt + p2.pt
        # Step 4: Calculate z
        z = pt_min / pt_sum
        E_sum = p1.E + p2.E
        p_sum = p1 + p2
        # Step 5: Calculate m^2 # m^2 = (E_sum)^2 - |p_sum|^2
        m_squared = E_sum**2 - p_sum.mag2

        # Compute logarithms
        epsilon = 1e-5  # Small positive value to replace zeros
        delta = ak.where(delta <= 0, epsilon, delta)
        k_T = ak.where(k_T <= 0, epsilon, k_T)
        z = ak.where(z <= 0, epsilon, z)
        m_squared = ak.where(m_squared <= 0, epsilon, m_squared)
        delta = np.log(delta)
        k_T = np.log(k_T)
        z = np.log(z)
        m_squared = np.log(m_squared)

        # Number of particles per event
        num_particles = ak.num(p4_chunk, axis=1)
        # Maximum padding for the sparse-like adjacency matrix
        pad_max = self.max_particles * (self.max_particles - 1) // 2

        # Sparse-like adjacency matrix
        sparse_adj_matrix = np.stack(
            [
                ak.to_numpy(ak.fill_none(ak.pad_none(delta, target=pad_max), -1)),
                ak.to_numpy(ak.fill_none(ak.pad_none(k_T, target=pad_max), -1)),
                ak.to_numpy(ak.fill_none(ak.pad_none(z, target=pad_max), -1)),
                ak.to_numpy(ak.fill_none(ak.pad_none(m_squared, target=pad_max), -1)),
            ],
            axis=-1,
        )

        # Create the 4-momentum array
        energy_padded = ak.fill_none(
            ak.pad_none(p4_chunk.energy, target=self.max_particles), 0
        )
        px_padded = ak.fill_none(ak.pad_none(p4_chunk.px, target=self.max_particles), 0)
        py_padded = ak.fill_none(ak.pad_none(p4_chunk.py, target=self.max_particles), 0)
        pz_padded = ak.fill_none(ak.pad_none(p4_chunk.pz, target=self.max_particles), 0)

        # Calculate jet-level properties
        jet_p4 = ak.sum(p4_chunk, axis=1)
        jet_pt = jet_p4.pt
        jet_eta = jet_p4.eta
        jet_phi = jet_p4.phi
        jet_energy = jet_p4.energy

        # Particle-level properties
        particle_pt = p4_chunk.pt
        particle_eta = p4_chunk.eta
        particle_phi = p4_chunk.phi
        particle_energy = p4_chunk.energy

        # Calculate Δη and Δφ
        delta_eta = particle_eta - jet_eta
        delta_phi = particle_phi - jet_phi

        # Calculate log(pT), log(E)
        log_pt = np.log(particle_pt)
        log_energy = np.log(particle_energy)

        # Calculate log(pT/pT_jet) and log(E/E_jet)
        pt_rel = particle_pt / jet_pt
        log_pt_rel = np.log(pt_rel)
        energy_rel = particle_energy / jet_energy
        log_energy_rel = np.log(energy_rel)

        # Calculate ΔR
        delta_r = np.hypot(delta_eta, delta_phi)

        # Pad kinematic variables
        log_pt_padded = ak.fill_none(ak.pad_none(log_pt, target=self.max_particles), 0)
        log_energy_padded = ak.fill_none(
            ak.pad_none(log_energy, target=self.max_particles), 0
        )
        log_pt_rel_padded = ak.fill_none(
            ak.pad_none(log_pt_rel, target=self.max_particles), 0
        )
        log_energy_rel_padded = ak.fill_none(
            ak.pad_none(log_energy_rel, target=self.max_particles), 0
        )
        delta_eta_padded = ak.fill_none(
            ak.pad_none(delta_eta, target=self.max_particles), 0
        )
        delta_phi_padded = ak.fill_none(
            ak.pad_none(delta_phi, target=self.max_particles), 0
        )
        delta_r_padded = ak.fill_none(
            ak.pad_none(delta_r, target=self.max_particles), 0
        )

        # Stack all features into a single array
        feature_matrix = np.stack(
            [
                ak.to_numpy(energy_padded),
                ak.to_numpy(px_padded),
                ak.to_numpy(py_padded),
                ak.to_numpy(pz_padded),
                ak.to_numpy(log_pt_padded),
                ak.to_numpy(log_energy_padded),
                ak.to_numpy(log_pt_rel_padded),
                ak.to_numpy(log_energy_rel_padded),
                ak.to_numpy(delta_eta_padded),
                ak.to_numpy(delta_phi_padded),
                ak.to_numpy(delta_r_padded),
            ],
            axis=-1,
        )

        # Create the mask
        particles = ak.to_numpy(num_particles)
        row_indices = np.arange(len(particles)).reshape(-1, 1)
        column_indices = np.arange(self.max_particles)
        mask = column_indices < particles[row_indices]

        # Extract labels
        labels = df_chunk["is_signal_new"].values

        return (
            feature_matrix.astype(np.float32),
            sparse_adj_matrix.astype(np.float32),
            mask.astype(np.int16),
            labels.astype(np.int16),
        )

    def generate_graphs(self, output_file="output.h5", n_jobs=1):
        """Load data from the HDF5 file and read the 4-momentum.
        Then generate node and edge features and store them in
        an output HDF5 file."""
        logging.info("Loading data from HDF5 file: %s", self.file_path)

        with h5py.File(output_file, "w") as h5f:
            lorentz_dset, adj_dset, mask_dset, label_dset = None, None, None, None

            # Determine the total number of rows
            with pd.HDFStore(self.file_path, mode="r") as store:
                total_rows = store.get_storer(self.key).nrows

            # Adjust total rows based on max_num_chunks
            if (
                self.max_num_chunks != -1
                and self.max_num_chunks < total_rows // self.chunk_size
            ):
                total_rows = int(self.max_num_chunks * self.chunk_size)

            # Create chunk ranges
            chunk_ranges = [
                (start, min(start + self.chunk_size, total_rows))
                for start in range(0, total_rows, self.chunk_size)
            ]

            total_chunks = len(chunk_ranges)
            processed_chunks = 0

            # Progress bar function
            def update_progress_bar(current, total):
                bar_length = 100  # Length of the progress bar
                progress = current / total
                block = int(bar_length * progress)
                progress_bar = f"[{'#' * block}{'-' * (bar_length - block)}] {current}/{total} chunks"
                sys.stdout.write(f"\r{progress_bar}")
                sys.stdout.flush()

            # Parallel processing with batching
            batch_size = n_jobs
            for i in range(0, len(chunk_ranges), batch_size):
                # Process a batch of chunks in parallel
                batch_ranges = chunk_ranges[i : i + batch_size]
                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.process_chunk)(start, end)
                    for start, end in batch_ranges
                )

                # Write processed data sequentially
                for features, adj_matrices, mask, labels in results:
                    processed_chunks += 1
                    update_progress_bar(processed_chunks, total_chunks)

                    if lorentz_dset is None:
                        # Create datasets on the first batch
                        lorentz_dset = h5f.create_dataset(
                            "feature_matrices",
                            data=features,
                            maxshape=(None, *features.shape[1:]),
                            chunks=True,
                            dtype=features.dtype,
                        )
                        adj_dset = h5f.create_dataset(
                            "adj_matrices",
                            data=adj_matrices,
                            maxshape=(None, *adj_matrices.shape[1:]),
                            chunks=True,
                            dtype=adj_matrices.dtype,
                        )
                        mask_dset = h5f.create_dataset(
                            "mask",
                            data=mask,
                            maxshape=(None, *mask.shape[1:]),
                            chunks=True,
                            dtype=mask.dtype,
                        )
                        label_dset = h5f.create_dataset(
                            "labels",
                            data=labels,
                            maxshape=(None,),
                            chunks=True,
                            dtype=labels.dtype,
                        )
                    else:
                        # Resize and write data for subsequent batches
                        lorentz_dset.resize(
                            lorentz_dset.shape[0] + features.shape[0], axis=0
                        )
                        lorentz_dset[-features.shape[0] :] = features
                        h5f.flush()

                        adj_dset.resize(
                            adj_dset.shape[0] + adj_matrices.shape[0], axis=0
                        )
                        adj_dset[-adj_matrices.shape[0] :] = adj_matrices
                        h5f.flush()

                        mask_dset.resize(mask_dset.shape[0] + mask.shape[0], axis=0)
                        mask_dset[-mask.shape[0] :] = mask
                        h5f.flush()

                        label_dset.resize(label_dset.shape[0] + labels.shape[0], axis=0)
                        label_dset[-labels.shape[0] :] = labels
                        h5f.flush()

        sys.stdout.write(
            "\n"
        )  # Move to the next line after the progress bar is complete
        logging.info("Data successfully saved to %s", output_file)


# Example usage:
# builder = ParticleGraphBuilder(file_path="val.h5", key="table", max_particles=100, chunk_size=1000, max_num_chunks=-1)
# builder.generate_graphs(output_file="val-graphs.h5", n_jobs=16)
