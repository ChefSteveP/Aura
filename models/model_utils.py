# model_utils.py
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from constants import MODELS_DIR
import os
import shutil
import json


class ModelUtils:
    def __init__(self):
        logger = logging.getLogger(__name__)

    def download_hugginface_model(self, model):
        """Downloads a full model from Hugging Face and returns it."""
        return AutoModel.from_pretrained(model)

    def delete_huggingface_cache(self, model_name):
        """Deletes the huggingface model cache."""
        model_name = "models--" + model_name.replace("/", "--")
        model_path = os.path.join(MODELS_DIR, model_name)
        locks_path = os.path.join(MODELS_DIR, ".locks")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            shutil.rmtree(locks_path)
            logging.info(f"Hugging Face cache cleared from '{model_path}'.")
        else:
            logging.info(f"No Hugging Face cache found at '{model_path}'.")

    def save_huggingface_model(self, model_name, file_path):
        if not os.path.exists(file_path):
            model = self.download_huggingface_model(model_name)
            self.save_model(model, file_path)
        logging.info(f"Model already exists at {file_path}. Loading cache")

        self.delete_huggingface_cache(model_name)

    def save_model(self, model, file_path):
        """Saves the full PyTorch model to the specified directory."""
        os.makedirs(file_path, exist_ok=True)
        torch.save(model, file_path)

    def load_model(self, file_path):
        """Loads the full PyTorch model from the specified directory."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Model file not found at '{file_path}'")
        return torch.load(file_path)

    def load_tokenizer(self, model_name, cache_dir=True):
        """Load a tokenizer from Hugging Face."""
        if cache_dir:
            return AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
        return AutoTokenizer.from_pretrained(model_name)

    def load_tokenizer_and_model(self, model_name, cache_dir):
        """Load both tokenizer and model."""
        model = self.load_model(model_name, cache_dir)
        tokenizer = self.load_tokenizer(model_name, cache_dir)
        return model, tokenizer

    def clear_csv_files(self, file_path):
        """Clear .csv files in a specified directory."""
        for file_name in os.listdir(file_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(file_path, file_name))
