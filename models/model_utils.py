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

    def log_gpu_memory(self, message=None):
        print_message = ""
        if message is not None:
            print_message += f"[{message}]"

        cuda_memory = torch.cuda.memory_reserved() / (1024**2)
        self.log.info(f"{print_message} GPU Memory Usage (CUDA): {cuda_memory:.2f} MB")

    def log_cpu_memory(self, message=None):
        print_message = ""
        if message is not None:
            print_message += f"[{message}]"

        rss = psutil.Process().memory_info().rss / (1024**2)
        self.log.info(f"{print_message} CPU Memory Usage (RSS): {rss:.2f} MB")

    def log_memory(self, message=None):
        self.log_cpu_memory(message=None)
        self.log_gpu_memory(message=None)
