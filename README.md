# Particle Interaction Builder

This Python module processes 4-momenta of jet constituents from HDF5 file to compute particle interactions as a graph. It generates 4-vectors, adjacency matrices, masks, and labels, and saves these processed features into a new HDF5 file. The module handles large datasets by processing and saving data in chunks.

## Features

- **4-Momentum:** Computes the 4-vector components for jet constituents.
- **Adjacency Matrices:** Calculates pairwise relationships between particles, such as ∆R, kT, z, and invariant mass squared.
- **Mask:** Identifies valid entries for variable-length jet constituents.
- **Labels:** Retains signal labels for machine learning tasks.


---
### 0. Download, Install Mamba, and Install Required Packages 
Download and install Mambaforge by excuting the `download-mamba.sh` bash script  
```bash
chmod +x download-mamba.sh
./installation.sh
```
Follow the instructions. Then, install the required packages  
```bash
chmod +x toptagging-setup.sh
./toptagging-setup.sh
```

### 1. Initialize the Calculator
Open Python interpreter  
```bash
ipython
```

Inside Python interpreter create an instance of the `ParticleGraphBuilder` class by specifying the input HDF5 file and the dataset key.

```python
builder = ParticleGraphBuilder("val.h5", key="table")
```

### 2. Load the Data
Call the `load_data()` method to load the particle data from the HDF5 file. This method initializes the 4-momentum components for each particle.  

```python
builder.load_data()
```

### 3. Save Processed Data
Call the `save_to_hdf5()` method to process the data in chunks and save the output to a new HDF5 file.  

```python
builder.save_to_hdf5("graph_val.h5", chunk_size=1000)
```

### 4. Example Code to Read the Output
Once the processed data has been saved, you can read and inspect it using the following example:

```python
import h5py

# Open the processed HDF5 file
with h5py.File("graph_val.h5", "r") as f:
    p4 = f["feature_matrices"][:]
    adj_matrices = f["adj_matrices"][:]
    mask = f["mask"][:]
    labels = f["labels"][:]

    # Example: Access the first event
    first_event_p4 = p4[0]
    first_event_adj = adj_matrices[0]
    first_event_mask = mask[0]
    first_event_label = labels[0]

    print("4-vectors for the first event:", first_event_p4)
    print("Adjacency Matrices for the first event:", first_event_adj)
    print("Masks for the first event:", first_event_mask)
    print("Label for the first event:", first_event_label)
```

### File Structure of the Output

The output HDF5 file contains the following datasets:

1. `feature_matrices`: A 3D array of shape `(N_events, max_particles, 4)` where each entry represents the energy and momentum `(E, Px, Py, Pz)` for particles.  
2. `adj_matrices`: A 4D array of shape `(N_events, max_particles, max_particles, 4)` containing pairwise interaction features `(ΔR, kT, z, m²)`.  
3. `mask`: A 2D boolean array of shape `(N_events, max_particles)` indicating valid particle entries.  
4. `labels`: A 1D array of shape `(N_events,)` containing labels for the events.
