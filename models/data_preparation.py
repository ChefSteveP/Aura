import os
import shutil
import logging
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login


class DataPreparation:
    def __init__(
        self,
        model_name,
        huggingface_token=None,
        download_path=None,
        split=None,
        tokenized_field_name=None,
        cache_dir="/home/shared_storage/models/data",
    ):
        self.huggingface_token = huggingface_token
        self.download_path = download_path
        self.split = split
        self.tokenized_field_name = tokenized_field_name
        self.cache_dir = os.path.abspath(cache_dir)
        logger = logging.getLogger(__name__)

        # Set Hugging Face cache directory
        os.environ["HF_HOME"] = os.path.abspath("/home/shared_storage")

        # Login to Hugging Face
        if self.huggingface_token:
            login(self.huggingface_token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def clear_non_english_arrow_files(self):
        """Delete .arrow files that are non-english"""
        base_dir = os.path.abspath("/home/shared_storage/huggingface/datasets")
        if not os.path.exists(base_dir):
            logging.info(f"Directory {base_dir} does not exist.")
            return

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                # Keep project_gutenberg-en files but remove other project_gutenberg language files (fr, de, etc.)
                if (
                    file.endswith(".arrow")
                    and file.startswith("project_gutenberg")
                    and not file.startswith("project_gutenberg-en")
                ):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        # logging.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logging.info(f"Failed to delete {file_path}: {e}")

    def clear_arrow_files(self):
        """Remove .arrow files to save space on disk"""
        base_dir = os.path.expanduser("home/shared_storage/huggingface/datasets")
        if not os.path.exists(base_dir):
            logging.info(f"Directory {base_dir} does not exist.")
            return

        # Remove project_gutenberg .arrow files to save space
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".arrow") and file.startswith("project_gutenberg"):
                    # Check if the filename is not followed by "-en"
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        # logging.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logging.info(f"Failed to delete {file_path}: {e}")

    def clear_raw_files(self):
        """Clear all raw files from /home/shared_storage/huggingface/hub/datasets--<download-path>"""

        # Identify raw, encoded files to delete
        hub_dir = os.path.abspath("/home/shared_storage/huggingface/hub/")
        data_dir = os.path.join(hub_dir, "datasets--" + self.download_path.replace("/", "--"))
        if os.path.exists(data_dir):
            # logging.info(f"Hub dir: {data_dir}")
            shutil.rmtree(data_dir)

    def get_dataset(self):
        """Check for tokenized cache. If missing, load raw dataset, tokenize, and save tokenized dataset."""
        if os.path.exists(self.cache_dir):
            logging.info("Loading tokenized dataset from cache...")
            dataset = load_from_disk(self.cache_dir)
        else:
            logging.info("Downloading and preparing the raw dataset...")

            # Load a very small subset of the dataset
            dataset = load_dataset(path=self.download_path, split=self.split)

            # Remove raw, encoded files
            self.clear_raw_files()
            self.clear_non_english_arrow_files()

            # Tokenize and save the dataset
            dataset = self.tokenize_data(dataset)

        # Print the total number of rows before processing
        logging.info(f"Number of rows in the dataset before processing: {len(dataset)}")

        return dataset

    def tokenize_data(self, dataset):
        """Tokenize the dataset and save it to the cache directory."""
        if "input_ids" in dataset.column_names:
            logging.info("Dataset already tokenized.")
        else:
            logging.info("Tokenizing dataset...")
            dataset = dataset.map(self.tokenizer_function, batched=True, batch_size=100)

            # logging.info("Shuffle data")
            dataset.shuffle()

            # Clear the dataset cache
            self.clear_arrow_files()

            # Save tokenized dataset to the cache directory
            # logging.info("Saving tokenized dataset to cache...")
            dataset.save_to_disk(self.cache_dir)

        return dataset

    def tokenizer_function(self, data):
        """Tokenization logic for the dataset."""
        return self.tokenizer(
            data[self.tokenized_field_name],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
