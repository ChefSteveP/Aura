#!/bin/bash

# Validate nvcc
echo -e "\n\n===== Check nvcc version =====\n\n"
nvcc --version
echo -e "\n"
read -p "Is nvcc installed and a version displayed (y/n)? " nvcc_installed
if [[ $nvcc_installed != "y" ]]; then
    echo -e "\n\n===== nvcc is not installed or not detected. Exiting script =====\n\n"
    exit 1
fi

# Install Git
echo -e "\n\n===== Installing Git =====\n\n"
sudo apt install git -y

# SSH Key Generation
echo -e "\n"
read -p "Enter your GitHub email: " email
read -p "Enter your GitHub username: " username
echo -e "\n\n===== Generating SSH key for GitHub =====\n\n"
ssh-keygen -t ed25519 -C "$email"

# Start the SSH agent
echo -e "\n\n===== Starting SSH agent =====\n\n"
eval "$(ssh-agent -s)"

# Add SSH private key to the agent
echo -e "\n\n===== Adding SSH key to SSH agent =====\n\n"
ssh-add ~/.ssh/id_ed25519

# Display the SSH public key
echo -e "\n\n===== Please add the SSH public key to your GitHub account before proceeding =====\n\n"
cat ~/.ssh/id_ed25519.pub
echo -e "\n"
read -p "Press Enter once you have added the key to GitHub..."

# Prompt for the GitHub repo
echo -e "\n\n===== Enter the GitHub email and username for identification =====\n\n"
git config --global user.email "$email"
git config --global user.name "$username"

# Validate CUDA is working via cuda-samples
echo -e "\n\n===== Validate that CUDA is working =====\n\n"
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/6_Performance/LargeKernelParameter
make
./LargeKernelParameter 
cd ~

echo -e "\n"
read -p "Is there message that states 'Test passed!' (y/n)? " cuda_installed
if [[ $cuda_installed != "y" ]]; then
    echo -e "\n\n===== nvcc is not installed or not detected. Exiting script =====\n\n"
    exit 1
fi

# Clone HPML GitHub repo
echo -e "\n"
read -p "Enter the GitHub repository URL to clone (e.g., git@github.com:user/repo.git): " repo
echo -e "\n\n===== Cloning the repository for HPML =====\n\n"
git clone "$repo"

# Install pyenv dependencies
echo -e "\n\n===== Installing pyenv dependencies =====\n\n"
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
echo -e "\n\n===== Installing pyenv =====\n\n"
curl https://pyenv.run | bash

# Add pyenv configuration to .bashrc
echo -e "\n\n===== Configuring pyenv in .bashrc =====\n\n"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo -e "\n\n===== Restart shell and run vm_setup_3.sh =====\n\n"
exec "$SHELL"