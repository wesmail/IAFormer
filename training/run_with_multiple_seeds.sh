#!/bin/bash

# Default configuration file
default_config_file="configs/ia_former.yaml"

# Use the provided argument as config_file or fall back to default
config_file="${1:-$default_config_file}"

# Check if the configuration file exists
if [ ! -f "$config_file" ]; then
    echo "Configuration file '$config_file' not found!"
    exit 1
fi

python_script="main.py"

# Define the list of seeds you want to use
seeds=(42 0 123 7 12345)

# Loop over each seed
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    
    # Run your Python script with the given seed
    python "$python_script" fit --config "$config_file" --seed_everything "$seed"
done