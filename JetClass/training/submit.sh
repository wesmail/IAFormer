#!/bin/bash
#SBATCH --job-name=jettag               # Job name
#SBATCH --output=%x_%A_%a.out           # Standard output log for each array job
#SBATCH --error=%x_%A_%a.err            # Error log for each array job
#SBATCH --ntasks=1                      # Number of tasks (usually 1 for single-core jobs)
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --mem=32G                       # Memory allocation
#SBATCH --time=24:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:2			# Number of GPUs
#SBATCH --partition=gpu4090             # Partition name

# Pass the config path (file or folder) to script.sh
CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: sbatch submit.sh <config_file_or_folder>"
  exit 1
fi

source script.sh "$CONFIG_PATH"

