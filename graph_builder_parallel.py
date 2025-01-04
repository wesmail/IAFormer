import h5py
import logging
import vector
import numpy as np
import pandas as pd
import awkward as ak

from scipy.sparse import coo_matrix
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParticleGraphBuilder:
    def __init__(self, file_path, key="table", max_particles=200):
        self.file_path = file_path
        self.key = key
        self._max_particles = max_particles
        logging.info("Initialized ParticleGraphBuilder with file: %s", file_path)

    def _col_list(self, prefix):
        """Generate column names for a given prefix."""
        return [f"{prefix}_{i}" for i in range(self._max_particles)]

    def load_data(self):
        """Load data from the HDF5 file and read the 4-momentum."""
        logging.info("Loading data from HDF5 file: %s", self.file_path)
        self.df = pd.read_hdf(self.file_path, key=self.key)
        # Shuffle the dataset
        self.df = self.df.sample(frac=1)

        _px = self.df[self._col_list("PX")].values
        _py = self.df[self._col_list("PY")].values
        _pz = self.df[self._col_list("PZ")].values
        _e = self.df[self._col_list("E")].values

        mask = _e > 0
        n_particles = np.sum(mask, axis=1)
        self.max_particles = np.max(n_particles)
        if self._max_particles < self.max_particles:
            self.max_particles = self._max_particles

        # Unflatten the valid arrays using the particle counts
        self.px = ak.unflatten(_px[mask], n_particles)
        self.py = ak.unflatten(_py[mask], n_particles)
        self.pz = ak.unflatten(_pz[mask], n_particles)
        self.e = ak.unflatten(_e[mask], n_particles)

        # Create the 4-vectors
        self.p4 = vector.zip(
            {"px": self.px, "py": self.py, "pz": self.pz, "energy": self.e}
        )

    def process_chunk(self, start, end):
        """Process a range of data to compute the 4-vectors, adjacency matrix, mask, and labels."""
        logging.info("Processing chunk from row %d to row %d", start, end)

        # Extract the relevant slice of 4-vectors
        p4_chunk = self.p4[start:end]

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

        num_particles = ak.num(p4_chunk, axis=1)

        # Create a placeholder 4D matrix (N_events, N_particles, N_particles, 4)
        adj_matrices = np.zeros(
            (len(p4_chunk), self.max_particles, self.max_particles, 4)
        )

        # Step 6: Fill a adjacency 3D matrix (N_events, N_particles, 4)
        for i, n in enumerate(num_particles):
            triu_indices = np.triu_indices(n, k=1)
            adj_matrices[i, triu_indices[0], triu_indices[1], 0] = ak.to_numpy(delta[i])
            adj_matrices[i, triu_indices[0], triu_indices[1], 1] = ak.to_numpy(k_T[i])
            adj_matrices[i, triu_indices[0], triu_indices[1], 2] = ak.to_numpy(z[i])
            adj_matrices[i, triu_indices[0], triu_indices[1], 3] = ak.to_numpy(
                m_squared[i]
            )
            adj_matrices[i, :, :, :] += adj_matrices[i, :, :, :].transpose(1, 0, 2)

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
        labels = self.df["is_signal_new"].iloc[start:end].values

        return (
            feature_matrix.astype(np.float32),
            adj_matrices.astype(np.float32),
            mask.astype(np.int16),
            labels.astype(np.int16),
        )

    def save_to_hdf5(self, output_file, chunk_size=100, max_num_chunks=-1, n_jobs=4):
        """Save feature matrix, edge feature matrix, mask, and labels to an HDF5 file in chunks."""
        logging.info("Saving data to HDF5 file: %s", output_file)

        with h5py.File(output_file, "w") as h5f:
            lorentz_dset, adj_dset, mask_dset, label_dset = None, None, None, None

            num_rows = len(self.p4)
            if max_num_chunks != -1 and max_num_chunks < num_rows // chunk_size:
                num_rows = int(max_num_chunks * chunk_size)

            # Create list of chunk ranges
            chunk_ranges = [
                (start, min(start + chunk_size, num_rows))
                for start in range(0, num_rows, chunk_size)
            ]

            # Process chunks in parallel using joblib
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_chunk)(start, end) for start, end in chunk_ranges
            )

            # Collect and save results
            for features, adj_matrices, mask, labels in results:
                if lorentz_dset is None:
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
                    lorentz_dset.resize(
                        lorentz_dset.shape[0] + features.shape[0], axis=0
                    )
                    lorentz_dset[-features.shape[0] :] = features

                    adj_dset.resize(adj_dset.shape[0] + adj_matrices.shape[0], axis=0)
                    adj_dset[-adj_matrices.shape[0] :] = adj_matrices

                    mask_dset.resize(mask_dset.shape[0] + mask.shape[0], axis=0)
                    mask_dset[-mask.shape[0] :] = mask

                    label_dset.resize(label_dset.shape[0] + labels.shape[0], axis=0)
                    label_dset[-labels.shape[0] :] = labels

        logging.info("Data successfully saved to %s", output_file)


class SparseParticleGraphBuilder:
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
        logging.info("Processing chunk from row %d to row %d", start, end)

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
        an output HDF5 file"""
        logging.info("Loading data from HDF5 file: %s", self.file_path)

        with h5py.File(output_file, "w") as h5f:
            lorentz_dset, adj_dset, mask_dset, label_dset = None, None, None, None

            with pd.HDFStore(self.file_path, mode="r") as store:
                total_rows = store.get_storer(self.key).nrows

            if (
                self.max_num_chunks != -1
                and self.max_num_chunks < total_rows // self.chunk_size
            ):
                total_rows = int(self.max_num_chunks * self.chunk_size)

            # Create list of chunk ranges
            chunk_ranges = [
                (start, min(start + self.chunk_size, total_rows))
                for start in range(0, total_rows, self.chunk_size)
            ]

            # Process chunks in parallel using joblib
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_chunk)(start, end) for start, end in chunk_ranges
            )

            # Collect and save results
            for features, adj_matrices, mask, labels in results:
                if lorentz_dset is None:
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
                    lorentz_dset.resize(
                        lorentz_dset.shape[0] + features.shape[0], axis=0
                    )
                    lorentz_dset[-features.shape[0] :] = features

                    adj_dset.resize(adj_dset.shape[0] + adj_matrices.shape[0], axis=0)
                    adj_dset[-adj_matrices.shape[0] :] = adj_matrices

                    mask_dset.resize(mask_dset.shape[0] + mask.shape[0], axis=0)
                    mask_dset[-mask.shape[0] :] = mask

                    label_dset.resize(label_dset.shape[0] + labels.shape[0], axis=0)
                    label_dset[-labels.shape[0] :] = labels

        logging.info("Data successfully saved to %s", output_file)


# Example usage:
# builder = ParticleGraphBuilder("val.h5")
# builder.load_data()
# builder.save_to_hdf5("output.h5", chunk_size=5000, max_num_chunks=-1, n_jobs=16)

# Alternative use
# builder = SparseParticleGraphBuilder(file_path="val.h5", key="table", max_particles=100, chunk_size=1000, max_num_chunks=-1)
# builder.generate_graphs(output_file="val-graphs.h5", n_jobs=16)
