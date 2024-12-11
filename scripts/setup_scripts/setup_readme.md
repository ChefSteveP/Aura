
# Instructions for GCP VM Setup for ML Projects

This guide explains how to set up a GCP VM for HPML projects using the provided bash scripts.
The process is divided into three stages, with each script automating a portion of the setup.

## Prerequisites

1. You must have access to a GCP Compute Engine VM.
2. These scripts assume ubuntu-2004-focal-v20240830 image on x86/64 architecture, but can be edited for any image
3. The VM should have internet access to download packages.

## How to Use the Setup Scripts

### 0. Initialize all bash scripts

```bash
chmod +x setup_scripts/1_install_cuda.sh && chmod +x setup_scripts/2_setup_github.sh && chmod +x setup_scripts/3_validate_cuda.sh && chmod +x setup_scripts/4_install_pyenv.sh && chmod +x setup_scripts/5_set_pyenv.sh
```

### 1. **Script: 1_install_cuda.sh**

This script performs the following tasks:

- Updates and upgrades the system.
- Installs `gcc` and checks its version.
- Installs `git`
- Checks for a CUDA-capable NVIDIA GPU on the VM.
- Provides a link to the official CUDA toolkit for installation and automatically installs CUDA Toolkit 12.6 and NVIDIA drivers.
- Adds the necessary CUDA paths to `.bashrc`.

**Steps to run:**
`setup_scripts/1_install_cuda.sh`
The script will reboot the system after completion. Once the system reboots, proceed to the next step by running the second script.

---

### 2. **Script: 2_setup_github.sh**

This script handles the following:

- Prompts you for your GitHub email and username to configure Git.
- Generates an SSH key for GitHub with each individual user, starts the SSH agent, and prompts you to add the key to your individual GitHub account.
- Clones a specified GitHub repository.

**Run the following steps:**
`setup_scripts/2_setup_github.sh`
Make sure to add the generated SSH key to your GitHub account when prompted.

### 3. **Script: 3_validate_cuda.sh**

Steps:

- Validates `nvcc` to ensure proper installation. This is a step that takes place after installing CUDA in `1_install_cuda.sh`.
- Clones the CUDA samples repository and runs a test to verify CUDA installation.

**Run the following steps:**
`setup_scripts/3_validate_cuda.sh`

### 4. **Script: 4_install_pyenv.sh**

- Installs `pyenv` and its dependencies. Restarts the shell to ensure proper installation

### 5. **Script: 5_set_pyenv.sh**

This script completes the final steps of the setup:

- Installs a specified version of Python using `pyenv` (defaults to Python 3.11.2 if none specified).
- Creates a virtual environment using the specified Python version.
- Cleans up unnecessary files and repositories (such as the CUDA samples and downloaded installation files).

Once completed, the GCP VM will be fully set up for ML development with Python and CUDA installed.

### Final Notes

- Refer to the official [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for troubleshooting.
- Refer to the official [pyenv documentation](https://github.com/pyenv/pyenv).
