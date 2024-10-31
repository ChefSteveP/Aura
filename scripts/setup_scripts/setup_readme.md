
Instructions for GCP VM Setup for ML Projects
==============================================

This guide explains how to set up a GCP VM for HPML projects using the provided bash scripts.
The process is divided into three stages, with each script automating a portion of the setup.

Prerequisites:
--------------
1. You must have access to a GCP Compute Engine VM. 
2. These scripts assume ubuntu-2004-focal-v20240830 image on x86/64 architecture, but can be edited for any image
3. The VM should have internet access to download packages.

How to Use the Setup Scripts:
------------------------------

### 0. Initialize all bash scripts
```
chmod +x setup_scripts/vm_setup_1.sh && chmod +x setup_scripts/vm_setup_2.sh && chmod +x setup_scripts/vm_setup_3.sh
```

### 1. **Script: vm_setup_1.sh**

This script performs the following tasks:
- Updates and upgrades the system.
- Installs `gcc` and checks its version.
- Checks for a CUDA-capable NVIDIA GPU on the VM.
- Provides a link to the official CUDA toolkit for installation and automatically installs CUDA Toolkit 12.6 and NVIDIA drivers.
- Adds the necessary CUDA paths to `.bashrc`.

#### Steps to run:
`setup_scripts/vm_setup_1.sh`
The script will reboot the system after completion. Once the system reboots, proceed to the next step by running the second script.

---

### 2. **Script: vm_setup_2.sh**

This script handles the following:
- Verifies the `nvcc` installation (CUDA compiler).
- Installs Git.
- Prompts you for your GitHub email and username to configure Git.
- Generates an SSH key for GitHub, starts the SSH agent, and prompts you to add the key to your GitHub account.
- Clones the CUDA samples repository and runs a test to verify CUDA installation.
- Clones a specified GitHub repository (typically your HPML project).
- Installs `pyenv` and its dependencies.

#### Steps to run:
`setup_scripts/vm_setup_2.sh`
Make sure to add the generated SSH key to your GitHub account when prompted.

---

### 3. **Script: vm_setup_3.sh**

This script completes the final steps of the setup:
- Installs a specified version of Python using `pyenv` (defaults to Python 3.11.2 if none specified).
- Creates a virtual environment using the specified Python version.
- Cleans up unnecessary files and repositories (such as the CUDA samples and downloaded installation files).

#### Steps to run:
`setup_scripts/vm_setup_2.sh`

Once completed, your GCP VM will be fully set up for machine learning development with Python and CUDA installed.

---

Final Notes:
------------
- Ensure that you have sufficient permissions to install software and modify system files.
- If you encounter issues with CUDA, refer to the official [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for troubleshooting.
- For managing Python environments and versions, refer to the official [pyenv documentation](https://github.com/pyenv/pyenv).

Happy coding!
