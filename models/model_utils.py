# model_utils.py
import os
import shutil
import psutil
import logging
import torch
from constants import MODELS_DIR, TOKENIZERS_DIR


class ModelUtils:
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def clear_locks_dir(self):
        """Clear .locks directory in a root directory."""
        for dir in [MODELS_DIR, TOKENIZERS_DIR]:
            if os.path.exists(dir + "/.locks"):
                shutil.rmtree(dir + "/.locks")

    def clear_csv_files(self, file_path):
        """Clear .csv files in a specified directory."""
        for file_name in os.listdir(file_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(file_path, file_name))

    def log_gpu_memory(self):
        print_message = ""
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(gpu_id)
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**2)
            reserved_memory = torch.cuda.memory_reserved(gpu_id) / (1024**2)
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**2)
            free_memory = reserved_memory - allocated_memory

            print_message += (
                f"\nGPU {gpu_id} -> "
                f"Total: {total_memory:.2f} MB, "
                f"Reserved: {reserved_memory:.2f} MB, "
                f"Allocated: {allocated_memory:.2f} MB, "
                f"Free: {free_memory:.2f} MB\n"
            )
        return print_message

    def log_cpu_memory(self):
        rss = psutil.Process().memory_info().rss / (1024**2)
        return f"\nCPU 0: -> Total: {rss:.2f} MB"

    def log_memory(self, message=None):
        print_message = ""
        if message is not None:
            print_message += f"[{message}]"

        print_message += self.log_gpu_memory()
        print_message += self.log_cpu_memory()
        self.log.info(print_message)
