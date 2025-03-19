import sys
import h5py
import logging
import vector
import numpy as np
import pandas as pd
import awkward as ak
from particle import Particle

from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TopParticleGraphBuilder:
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

        # more features in the U matrix
        pt_sum_rel = pt_sum / ak.sum(p4_chunk, axis=1).pt
        energy_sum_rel = E_sum / ak.sum(p4_chunk, axis=1).energy

        # Compute logarithms
        epsilon = 1e-5  # Small positive value to replace zeros
        delta = ak.where(delta <= 0, epsilon, delta)
        k_T = ak.where(k_T <= 0, epsilon, k_T)
        z = ak.where(z <= 0, epsilon, z)
        m_squared = ak.where(m_squared <= 0, epsilon, m_squared)
        pt_sum_rel = ak.where(pt_sum_rel <= 0, epsilon, pt_sum_rel)
        energy_sum_rel = ak.where(energy_sum_rel <= 0, epsilon, energy_sum_rel)

        # logarithm
        delta = np.log(delta)
        k_T = np.log(k_T)
        z = np.log(z)
        m_squared = np.log(m_squared)
        pt_sum_rel = np.log(pt_sum_rel)
        energy_sum_rel = np.log(energy_sum_rel)

        # Number of particles per event
        num_particles = ak.num(p4_chunk, axis=1)
        # Maximum padding for the sparse-like adjacency matrix
        self.pad_max = self.max_particles * (self.max_particles - 1) // 2

        # Sparse-like adjacency matrix
        sparse_adj_matrix = np.stack(
            [
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(delta, target=self.pad_max), -1000)
                ),
                ak.to_numpy(ak.fill_none(ak.pad_none(k_T, target=self.pad_max), -1000)),
                ak.to_numpy(ak.fill_none(ak.pad_none(z, target=self.pad_max), -1000)),
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(m_squared, target=self.pad_max), -1000)
                ),
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(pt_sum_rel, target=self.pad_max), -1000)
                ),
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(energy_sum_rel, target=self.pad_max), -1000
                    )
                ),
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
            mask.astype(np.float32),
            labels.astype(np.float32),
        )

    def generate_graphs(self, output_file="output", n_jobs=1):
        """Load data and generate node and edge features, storing them as NumPy memmap arrays."""
        logging.info("Loading data from HDF5 file: %s", self.file_path)

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
            progress_bar = (
                f"[{'=' * block}{'-' * (bar_length - block)}] {current}/{total} chunks"
            )
            sys.stdout.write(f"\r{progress_bar}")
            sys.stdout.flush()

        batch_size = n_jobs

        # Initialize H5 datasets for writing (will be set on the first batch)
        outfile = h5py.File(f"{output_file}", "w")
        _feature_matrix = outfile.create_dataset(
            "feature_matrix",
            shape=(total_rows, self.max_particles, 11),
            dtype="float32",
            chunks=(self.chunk_size, self.max_particles, 11),
        )

        _adjacancy_matrix = outfile.create_dataset(
            "adjacancy_matrix",
            shape=(total_rows, self.max_particles * (self.max_particles - 1) // 2, 6),
            dtype="float32",
            chunks=(
                self.chunk_size,
                self.max_particles * (self.max_particles - 1) // 2,
                6,
            ),
        )
        _mask = outfile.create_dataset(
            "mask",
            shape=(total_rows, self.max_particles),
            dtype="float32",
            chunks=(self.chunk_size, self.max_particles),
        )
        _labels = outfile.create_dataset(
            "labels", shape=(total_rows,), dtype="float32", chunks=(self.chunk_size,)
        )

        for i in range(0, len(chunk_ranges), batch_size):
            # Process a batch of chunks in parallel
            batch_ranges = chunk_ranges[i : i + batch_size]
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_chunk)(start, end) for (start, end) in batch_ranges
            )

            # Loop over results from each chunk in this batch
            for j, (start, end) in enumerate(batch_ranges):
                # Unpack the returned arrays from process_chunk
                features, adj_matrices, mask, labels = results[j]

                # Write directly to the [start:end] slice to keep indexes aligned
                _feature_matrix[start:end] = features
                _adjacancy_matrix[start:end] = adj_matrices
                _mask[start:end] = mask
                _labels[start:end] = labels

                # Update progress
                processed_chunks += 1
                update_progress_bar(processed_chunks, total_chunks)

        sys.stdout.write(
            "\n"
        )  # Move to the next line after the progress bar is complete
        outfile.flush()
        outfile.close()
        logging.info("Data successfully saved.")


class QGParticleGraphBuilder:
    def __init__(
        self,
        file_paths: list,
        max_particles: int = 200,
        chunk_size: int = 1000,
        max_num_chunks: int = -1,
    ):
        self.file_paths = file_paths
        self.max_particles = max_particles
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        logging.info("Initialized ParticleGraphBuilder with files: %s", file_paths)

    def p4_from_ptetaphimass(self, pt, eta, phi, pids):
        # Function to get mass for each PID
        def get_mass(pid):
            try:
                return Particle.from_pdgid(int(pid)).mass / 1000.0  # Convert MeV to GeV
            except Exception:
                return 0.0  # Return 0 if PID is invalid

        # Vectorized mass extraction
        mass = ak.Array([[get_mass(pid) for pid in event] for event in pids])

        return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})

    def process_chunk(self, arr, labels):
        """Process a range of data to compute the 4-vectors, adjacency matrix, mask, and labels."""

        _pt = arr[:, :, 0]
        _eta = arr[:, :, 1]
        _phi = arr[:, :, 2]
        _pid = arr[:, :, 3]

        # Create mask for non-zero pT particles and count particles per jet
        mask = _pt > 0
        n_particles = np.sum(mask, axis=1)
        max_particles_chunk = np.max(n_particles)
        max_particles = min(self.max_particles, max_particles_chunk)

        # Convert to awkward arrays
        pt = ak.unflatten(_pt[mask], n_particles)
        eta = ak.unflatten(_eta[mask], n_particles)
        phi = ak.unflatten(_phi[mask], n_particles)
        pids = ak.unflatten(_pid[mask], n_particles)

        p4_chunk = self.p4_from_ptetaphimass(pt, eta, phi, pids)
        p4_chunk = p4_chunk[:, :max_particles]

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

        # more features in the U matrix
        pt_sum_rel = pt_sum / ak.sum(p4_chunk, axis=1).pt
        energy_sum_rel = E_sum / ak.sum(p4_chunk, axis=1).energy

        # Compute logarithms
        epsilon = 1e-5  # Small positive value to replace zeros
        delta = ak.where(delta <= 0, epsilon, delta)
        k_T = ak.where(k_T <= 0, epsilon, k_T)
        z = ak.where(z <= 0, epsilon, z)
        m_squared = ak.where(m_squared <= 0, epsilon, m_squared)
        pt_sum_rel = ak.where(pt_sum_rel <= 0, epsilon, pt_sum_rel)
        energy_sum_rel = ak.where(energy_sum_rel <= 0, epsilon, energy_sum_rel)

        # logarithm
        delta = np.log(delta)
        k_T = np.log(k_T)
        z = np.log(z)
        m_squared = np.log(m_squared)
        pt_sum_rel = np.log(pt_sum_rel)
        energy_sum_rel = np.log(energy_sum_rel)

        # Number of particles per event
        num_particles = ak.num(p4_chunk, axis=1)
        # Maximum padding for the sparse-like adjacency matrix
        self.pad_max = self.max_particles * (self.max_particles - 1) // 2

        # Sparse-like adjacency matrix
        sparse_adj_matrix = np.stack(
            [
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(delta, target=self.pad_max), -1000)
                ),
                ak.to_numpy(ak.fill_none(ak.pad_none(k_T, target=self.pad_max), -1000)),
                ak.to_numpy(ak.fill_none(ak.pad_none(z, target=self.pad_max), -1000)),
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(m_squared, target=self.pad_max), -1000)
                ),
                ak.to_numpy(
                    ak.fill_none(ak.pad_none(pt_sum_rel, target=self.pad_max), -1000)
                ),
                ak.to_numpy(
                    ak.fill_none(
                        ak.pad_none(energy_sum_rel, target=self.pad_max), -1000
                    )
                ),
            ],
            axis=-1,
        )

        # Create the 4-momentum array
        energy_padded = ak.fill_none(
            ak.pad_none(p4_chunk.energy, target=self.max_particles, clip=True), 0
        )
        px_padded = ak.fill_none(
            ak.pad_none(p4_chunk.px, target=self.max_particles, clip=True), 0
        )
        py_padded = ak.fill_none(
            ak.pad_none(p4_chunk.py, target=self.max_particles, clip=True), 0
        )
        pz_padded = ak.fill_none(
            ak.pad_none(p4_chunk.pz, target=self.max_particles, clip=True), 0
        )

        # Pid information
        # Compute charge: Assume charge is known for each PID based on standard physics conventions
        charge = np.where(
            pids == 22,
            0,  # Photons = charge 0
            np.where(
                np.abs(pids) == 11,
                -1,  # Electrons = charge -1
                np.where(
                    np.abs(pids) == 13,
                    -1,  # Muons = charge -1
                    np.where(
                        np.abs(pids) == 211,
                        1,  # Charged pions = charge +/-1
                        np.where(
                            np.abs(pids) == 321,
                            1,  # Charged kaons = charge +/-1
                            np.where(
                                np.abs(pids) == 2212,
                                1,  # Protons = charge +1
                                0,
                            ),
                        ),
                    ),
                ),
            ),
        )  # Everything else neutral

        # Boolean masks
        Electron = np.abs(pids) == 11
        Muon = np.abs(pids) == 13
        Photon = pids == 22
        CH = (np.abs(pids) == 211) | (np.abs(pids) == 321) | (np.abs(pids) == 2212)
        NH = (np.abs(pids) == 130) | (np.abs(pids) == 2112) | (pids == 0)

        charge_padded = ak.fill_none(
            ak.pad_none(charge, target=self.max_particles, clip=True), 0
        )
        Electron_padded = ak.fill_none(
            ak.pad_none(Electron, target=self.max_particles, clip=True), 0
        )
        Muon_padded = ak.fill_none(
            ak.pad_none(Muon, target=self.max_particles, clip=True), 0
        )
        Photon_padded = ak.fill_none(
            ak.pad_none(Photon, target=self.max_particles, clip=True), 0
        )
        CH_padded = ak.fill_none(
            ak.pad_none(CH, target=self.max_particles, clip=True), 0
        )
        NH_padded = ak.fill_none(
            ak.pad_none(NH, target=self.max_particles, clip=True), 0
        )

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
        log_pt_padded = ak.fill_none(
            ak.pad_none(log_pt, target=self.max_particles, clip=True), 0
        )
        log_energy_padded = ak.fill_none(
            ak.pad_none(log_energy, target=self.max_particles, clip=True), 0
        )
        log_pt_rel_padded = ak.fill_none(
            ak.pad_none(log_pt_rel, target=self.max_particles, clip=True), 0
        )
        log_energy_rel_padded = ak.fill_none(
            ak.pad_none(log_energy_rel, target=self.max_particles, clip=True), 0
        )
        delta_eta_padded = ak.fill_none(
            ak.pad_none(delta_eta, target=self.max_particles, clip=True), 0
        )
        delta_phi_padded = ak.fill_none(
            ak.pad_none(delta_phi, target=self.max_particles, clip=True), 0
        )
        delta_r_padded = ak.fill_none(
            ak.pad_none(delta_r, target=self.max_particles, clip=True), 0
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
                ak.to_numpy(charge_padded),
                ak.to_numpy(Electron_padded),
                ak.to_numpy(Muon_padded),
                ak.to_numpy(Photon_padded),
                ak.to_numpy(CH_padded),
                ak.to_numpy(NH_padded),
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

        return (
            feature_matrix.astype(np.float32),
            sparse_adj_matrix.astype(np.float32),
            mask.astype(np.float32),
            labels.astype(np.float32),
        )

    def generate_graphs(self, output_file="output.h5", n_jobs=1):
        """Load data from multiple NumPy arrays and generate node and edge features, storing them in a single HDF5 dataset."""
        logging.info("Processing multiple NumPy files: %s", self.file_paths)

        total_rows = 0
        file_chunks = []  # To store chunk ranges for each file

        for file_path in self.file_paths:
            logging.info("Loading data from NumPy file: %s", file_path)
            array = np.load(file_path)
            arr, lbls = array["X"], array["y"]
            num_rows = arr.shape[0]
            file_chunks.append((total_rows, total_rows + num_rows, arr, lbls))
            total_rows += num_rows

        if self.max_num_chunks != -1:
            max_rows = self.max_num_chunks * self.chunk_size
            total_rows = min(total_rows, max_rows)

        with h5py.File(output_file, "w", libver="latest") as outfile:
            _feature_matrix = outfile.create_dataset(
                "feature_matrix",
                shape=(total_rows, self.max_particles, 17),
                dtype="float32",
                chunks=(self.chunk_size, self.max_particles, 17),
            )
            _adjacancy_matrix = outfile.create_dataset(
                "adjacancy_matrix",
                shape=(
                    total_rows,
                    self.max_particles * (self.max_particles - 1) // 2,
                    6,
                ),
                dtype="float32",
                chunks=(
                    self.chunk_size,
                    self.max_particles * (self.max_particles - 1) // 2,
                    6,
                ),
            )
            _mask = outfile.create_dataset(
                "mask",
                shape=(total_rows, self.max_particles),
                dtype="float32",
                chunks=(self.chunk_size, self.max_particles),
            )
            _labels = outfile.create_dataset(
                "labels",
                shape=(total_rows,),
                dtype="float32",
                chunks=(self.chunk_size,),
            )

            processed_chunks = 0

            def update_progress_bar(current, total):
                bar_length = 100
                progress = current / total
                block = int(bar_length * progress)
                progress_bar = f"[{'=' * block}{'-' * (bar_length - block)}] {current}/{total} chunks"
                sys.stdout.write(f"\r{progress_bar}")
                sys.stdout.flush()

            for start_idx, end_idx, arr, labels in file_chunks:
                num_rows = min(end_idx - start_idx, total_rows - start_idx)
                chunk_ranges = [
                    (i * self.chunk_size, min((i + 1) * self.chunk_size, num_rows))
                    for i in range((num_rows + self.chunk_size - 1) // self.chunk_size)
                ]

                results = Parallel(n_jobs=n_jobs)(
                    delayed(self.process_chunk)(arr[start:end], labels[start:end])
                    for (start, end) in chunk_ranges
                )

                for j, (start, end) in enumerate(chunk_ranges):
                    actual_start = start_idx + start
                    actual_end = start_idx + end
                    features, adj_matrices, mask, label_values = results[j]
                    _feature_matrix[actual_start:actual_end] = features
                    _adjacancy_matrix[actual_start:actual_end] = adj_matrices
                    _mask[actual_start:actual_end] = mask
                    _labels[actual_start:actual_end] = label_values
                    processed_chunks += 1
                    update_progress_bar(processed_chunks, len(chunk_ranges))

            sys.stdout.write("\n")
            logging.info("Data successfully saved to %s", output_file)
