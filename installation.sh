#!/bin/bash

# Prompt user to input their operating system
read -p "Enter your operating system (Linux or MacOSX): " SYS

# Validate input
if [[ "$SYS" != "Linux" && "$SYS" != "MacOSX" ]]; then
    echo "Invalid input. Please enter 'Linux' or 'MacOSX'."
    exit 1
fi

# Get the CPU architecture
ARCH=$(uname -m)

# Map architecture for compatibility
if [[ "$ARCH" == "x86_64" ]]; then
    ARCH="x86_64"
elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    ARCH="arm64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Define the URL for the Mambaforge installer
URL="https://github.com/conda-forge/miniforge/releases/download/24.11.0-0/Mambaforge-24.11.0-0-${SYS}-${ARCH}.sh"

# Download the installer
echo "Downloading Mambaforge for ${SYS}-${ARCH}..."
wget -O Mambaforge-Installer.sh "$URL"

# Confirm download
if [[ $? -eq 0 ]]; then
    echo "Mambaforge installer downloaded successfully as Mambaforge-Installer.sh."
else
    echo "Failed to download Mambaforge installer. Check your internet connection and try again."
    exit 1
fi

# Make the installer executable
chmod +x Mambaforge-Installer.sh

# Run the installer
echo "Running the Mambaforge installer..."
./Mambaforge-Installer.sh

# Confirm installation
if [[ $? -eq 0 ]]; then
    echo "Mambaforge installed successfully."
else
    echo "Mambaforge installation failed."
    exit 1
fi
