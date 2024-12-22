import pandas as pd
import numpy as np
import awkward as ak
import vector
import h5py
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParticleGraphBuilder:
    def __init__(self, file_path, key="table", max_particles=200):
        self.file_path = file_path
        self.key = key
        self.max_particles = max_particles
        logging.info(
            "Initialized ParticleInteractionCalculator with file: %s", file_path
        )

    def _col_list(self, prefix):
        """Generate column names for a given prefix."""
        return [f"{prefix}_{i}" for i in range(self.max_particles)]

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

        num_particles = ak.num(p4_chunk, axis=1)

        # Create a placeholder 4D matrix (N_events, N_particles, N_particles, 4)
        adj_matrices = np.zeros(
            (len(p4_chunk), self.max_particles, self.max_particles, 4)
        )

        # Step 6: Fill the adjacency matrices
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

        p4 = np.stack(
            [
                ak.to_numpy(energy_padded),
                ak.to_numpy(px_padded),
                ak.to_numpy(py_padded),
                ak.to_numpy(pz_padded),
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

        return p4, adj_matrices, mask, labels

    def save_to_hdf5(self, output_file, chunk_size=100, max_num_chunks=-1):
        """Save feature matrix, edge feature matrix, mask, and labels to an HDF5 file in chunks."""
        logging.info("Saving data to HDF5 file: %s", output_file)

        with h5py.File(output_file, "w") as h5f:
            lorentz_dset = None
            adj_dset = None
            mask_dset = None
            label_dset = None

            num_rows = len(self.p4)
            if max_num_chunks != -1 or max_num_chunks < num_rows // chunk_size:
                num_rows = int(max_num_chunks * chunk_size)
            for start in range(0, num_rows, chunk_size):
                end = min(start + chunk_size, num_rows)
                lorentz_array, adj_matrices, mask, labels = self.process_chunk(
                    start, end
                )

                if lorentz_dset is None:
                    lorentz_dset = h5f.create_dataset(
                        "feature_matrices",
                        data=lorentz_array,
                        maxshape=(None, *lorentz_array.shape[1:]),
                        chunks=True,
                        dtype=lorentz_array.dtype,
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
                        lorentz_dset.shape[0] + lorentz_array.shape[0], axis=0
                    )
                    lorentz_dset[-lorentz_array.shape[0] :] = lorentz_array

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
# builder.save_to_hdf5("output.h5", chunk_size=5000)
