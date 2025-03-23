# IAFormer: Interaction-Aware Sparse Attention Transformer for Collider Data Analysis

## Required Packages
To use this package, you need to install the following dependencies:

1. `numpy`
2. `pandas`
3. `pytables`
4. `scikit-learn`
5. `matplotlib`
6. `awkward`
7. `vector`
8. `uproot`
9. `h5py`
10. `pytorch`
11. `lightning`
12. `torchmetrics`
13. `particle`

You can install them using Conda and Pip (You can use `toptagging-setup.sh` as explained below):

```sh
micromamba install numpy pandas pytables scikit-learn matplotlib seaborn jupyter tqdm awkward vector uproot h5py -c conda-forge -y  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # for gpu and cuda  
pip install lightning  
pip install "lightning[pytorch-extra]"  
python -m pip install particle  
```


## Particle Graph Builder

This Python module processes 4-momenta of jet constituents from HDF5 file to compute particle interactions as a graph. It generates 4-vectors, adjacency matrices, masks, and labels, and saves these processed features into a new HDF5 file. The module handles large datasets by processing and saving data in chunks.

## Features

- **4-Momentum:** Computes the 4-vector components for jet constituents.
- **Adjacency Matrices:** Calculates pairwise relationships between particles, such as ∆R, kT, z, and invariant mass squared.
- **Mask:** Identifies valid entries for variable-length jet constituents.
- **Labels:** Retains signal labels for machine learning tasks.


---
### 0. Download, Install Micromamba, and Install Required Packages 
Download and install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Follow the instructions. Then, install the required packages  
```bash
chmod +x toptagging-setup.sh
./toptagging-setup.sh
mamba activate toptagging
```

### 1. Initialize the Builder
Open Python interpreter  
```bash
ipython
```

Inside Python interpreter create an instance of the `TopParticleGraphBuilder` class for top dataset or `QGParticleGraphBuilder` for quark-gluon dataset.

```python
from graph_builder import TopParticleGraphBuilder
builder = TopParticleGraphBuilder(file_path="train.h5", key="table", max_particles=100, chunk_size=1000, max_num_chunks=-1)
```

or

```python
from graph_builder import QGParticleGraphBuilder
builder = QGParticleGraphBuilder(file_paths=[f"QG_jets_{i:02d}.npz" for i in range(16)], key="table", max_particles=100, chunk_size=1000, max_num_chunks=-1)
```

### 2. Generate and save graphs
Call the `generate_graphs()` method to process the data in chunks and save the output to a new HDF5 file.  

```python
builder.generate_graphs("train_graph.h5", n_jobs=16)
```

### 3. Example Code to Read the Output
Once the processed data has been saved, you can read and inspect it using the following example:

```python
import h5py

# Open the processed HDF5 file
with h5py.File("train_graph.h5", "r") as f:
    p4 = f["feature_matrix"][:]
    adj_matrices = f["adjacancy_matrix"][:]
    mask = f["mask"][:]
    labels = f["labels"][:]

    # Example: Access the first event
    first_event_p4 = p4[0][:4]
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

1. `feature_matrix`: A 3D array of shape `(N_events, max_particles, features)` where each entry represents the energy and momentum `(E, Px, Py, Pz)` for particles and other features.  
2. `adjacancy_matrix`: A 4D array of shape `(N_events, max_particles, max_particles, 4)` containing pairwise interaction features `(ΔR, kT, z, m²)`.  
3. `mask`: A 2D boolean array of shape `(N_events, max_particles)` indicating valid particle entries.  
4. `labels`: A 1D array of shape `(N_events,)` containing labels for the events.

# Training
After creating the graph files in a form of `h5` files for training, validation and testing, you are ready to train the model.  
The first step is to modify the configuration file `configs/config.yaml` according to your needs, e.g., number of epochs, model size, ...  
Then simply run
```python
ipython main.py -- fit --config=configs/ia_former.yaml
```
