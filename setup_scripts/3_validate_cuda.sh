echo -e "\n\n===== Check nvcc version =====\n\n"
nvcc --version
echo -e "\n"
read -p "Is nvcc installed and a version displayed (y/n)? " nvcc_installed
if [[ $nvcc_installed != "y" ]]; then
    echo -e "\n\n===== nvcc is not installed or not detected. Exiting script =====\n\n"
    exit 1
fi

# Validate CUDA is working via cuda-samples
echo -e "\n\n===== Validate that CUDA is working =====\n\n"
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/6_Performance/LargeKernelParameter
make
./LargeKernelParameter
cd ~

echo -e "\n"
read -p "Is there a message that states 'Test passed!' (y/n)? " cuda_installed
if [[ $cuda_installed != "y" ]]; then
    echo -e "\n\n===== CUDA is not working as expected. Exiting script =====\n\n"
    exit 1
fi