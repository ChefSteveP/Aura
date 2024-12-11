#!/bin/bash

# Update and upgrade the system
echo -e "\n\n===== Updating and upgrading the system =====\n\n"
sudo apt update && sudo apt upgrade -y

# Validate that gcc is installed
echo -e "\n\n===== Installing dependencies =====\n\n"
sudo apt install -y gcc tmux git nvidia-open make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev cloud-guest-utils

echo -e "\n\n===== Checking gcc install version =====\n\n"
gcc --version
echo -e "\n"
read -p "Is gcc installed and a version displayed (y/n)? " gcc_installed
if [[ $gcc_installed != "y" ]]; then
    echo -e "\n\n===== gcc is not installed or not detected. Exiting script =====\n\n"
    exit 1
fi

# Check for a CUDA-capable GPU
echo -e "\n\n===== Checking for a CUDA-capable GPU =====\n\n"
lspci | grep -i nvidia
echo -e "\n"
read -p "Is a CUDA-capable NVIDIA GPU detected (y/n)? " gpu_detected
if [[ $gpu_detected != "y" ]]; then
    echo -e "\n\n===== No CUDA-capable GPU detected. Exiting script =====\n\n"
    exit 1
fi

# Inform the user about CUDA installation and provide download links
echo -e "\n\n===== Installing the CUDA toolkit =====\n\n"
echo "Please refer to the official CUDA installation guide here: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
echo "You can download the CUDA toolkit from this link: https://developer.nvidia.com/cuda-downloads"

# Installing CUDA Toolkit
echo -e "\n\n===== Installing CUDA Toolkit (assumes x86_64 Ubuntu 20.04, navigate to CUDA toolkit link above for specific Linux version) =====\n\n"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-12-6

# Set path for CUDA
echo -e "\n\n===== Add CUDA to PATH in .bashrc =====\n\n"
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc


echo -e "\n\n===== NVIDIA driver installed. Rebooting system. Then run 2_setup_github.sh =====\n\n"
sudo reboot