#!/bin/bash

CONFIG_PATH=$1

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: script.sh <config_file_or_folder>"
  exit 1
fi

# Load Conda and activate the environment
# Load the micromamba shell hook for bash
eval "$(micromamba shell hook --shell bash)"
# Activate the environment
micromamba activate pytorch

# Check if it's a file
if [ -f "$CONFIG_PATH" ]; then
  echo "Running training with config: $CONFIG_PATH"
  srun python main.py fit --config "$CONFIG_PATH"

# Or a directory
elif [ -d "$CONFIG_PATH" ]; then
  echo "Running training for all YAML configs in: $CONFIG_PATH"
  for config in "$CONFIG_PATH"/*.yaml; do
    echo "Running training with config: $config"
    srun python main.py fit --config "$config"
  done
else
  echo "Provided path is neither a file nor a directory: $CONFIG_PATH"
  exit 1
fi

