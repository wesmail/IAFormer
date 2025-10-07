# Training IAFormer on the JetClass Dataset:  


## 1. JetClass Dataset Downloader `get_jetclass.sh`:

This Bash script automates downloading all JetClass root files from **Zenodo**.  
It uses `aria2c` for fast parallel downloads, which is especially efficient on HPC clusters.
If you don't have `aria2c`, please install it first:
#### **On Ubuntu / Debian**
```bash
sudo apt update
sudo apt install -y aria2
```

### **on MacOS**
```bash
brew install aria2
```

## Usage

```bash
chmod +x get_jetclass.sh
./get_jetclass.sh
```

Then uncompress the downloaded files via:
```bash
tar -xvf train_part_x.tar
```


## 2. JetClass File Regrouper `regroup_jetclass_by_index.sh`:

This utility script reorganizes JetClass ROOT files into **per-index folders** for easier processing.  
Each output folder (e.g., `train_00`, `train_01`, …) contains **one file per physics class**, such as `HToBB_00.root`, `ZToQQ_00.root`, etc.

The aim of this is to create a consistent directory layout before launching graph-building jobs.  
This ensures that each SLURM job processes exactly one folder → one HDF5 output.

## Usage

```bash
chmod +x regroup_jetclass_by_index.sh
./regroup_jetclass_by_index.sh
```

## 3. JetClass Graph Builder `graph_builder.py`:

This Python module converts **JetClass ROOT files** into a single **HDF5 file** optimized for machine learning training within the **IAFormer** pipeline.  
It computes both **node-level** (particle) and **edge-level** (pairwise interaction) features, then stores them efficiently with compression and metadata. 
The script transforms the raw particle-level data into structured graph representations suitable for Transformers.  

Before use, please make sure that all required packages are already installed. This includes `h5py`, `numpy`, `uproot`, `awkward`, `vector` and `tqdm`
## Usage

```bash
ipython graph_builder.py -- \
  --roots "JetClass_Zenodo/train_00/*.root" \
  --out "outputs/train_00.h5" \
  --chunk_size 16384 \
  --compression lzf \
  --max_particles 100
```

This will create a single **HDF5 file** containing **1,000,000 events** in total, spanning **10 physics classes**, where each class includes **100,000 events**.  
If you have access to a SLURM cluster, you can **automate the conversion of the entire JetClass dataset** (≈100 million events) using the accompanying automation scripts.

To do this, simply use the two helper scripts:
- `run_graph_builder.sh` — executes the graph builder for a specific dataset split.
- `submit_array_jobs.slurm` — submits all jobs as a SLURM array for parallel processing.

> ⚠️ **Important:**  
> Before running, make sure to modify both `run_graph_builder.sh` and `submit_array_jobs.slurm` according to your **directory structure** and **cluster specifications**.

Then launch the array job:
```bash
sbatch --array=0-99 submit_array_jobs.slurm
```