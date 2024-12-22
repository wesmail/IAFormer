#!/bin/bash

# Check if running on Mac or Linux
OS=$(uname)
if [ "$OS" == "Darwin" ]; then
    echo "Mac detected: Installing PyTorch with CPU-only support."
    mamba create -n toptagging numpy pandas scikit-learn matplotlib seaborn jupyter tqdm awkward vector uproot h5py pytorch torchvision torchaudio cpuonly lightning -c conda-forge -c pytorch -y && pip install "lightning[pytorch-extra]"
else
    echo "Linux detected: Checking for CUDA version..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
        echo "CUDA Version detected: $CUDA_VERSION"
        echo "Installing PyTorch with CUDA support."
        mamba create -n toptagging numpy pandas scikit-learn matplotlib seaborn jupyter tqdm awkward vector uproot h5py pytorch torchvision torchaudio pytorch-cuda=12.4 lightning -c conda-forge -c pytorch -c nvidia -y && pip install "lightning[pytorch-extra]"
    else
        echo "CUDA not detected: Installing PyTorch with CPU-only support."
        mamba create -n toptagging numpy pandas scikit-learn matplotlib seaborn jupyter tqdm awkward vector uproot h5py pytorch torchvision torchaudio cpuonly lightning -c conda-forge -c pytorch -y && pip install "lightning[pytorch-extra]"
    fi
fi

echo "Setup completed successfully!"
