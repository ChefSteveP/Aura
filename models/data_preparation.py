import os
import shutil
import glob
import logging
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from constants import STORAGE_DIR, DATA_DIR, MODELS_DIR, HUB_DIR, CACHE_DIR


class DataPreparation:
    def __init__(
        self,
        tokenizer,
        download_path=None,
        split=None,
        tokenized_field_name=None,
        tokenize_batch_size=100,
        dataloader_batch_size=8,
        dataset_columns=["input_ids", "attention_mask"],
        test_size=0.2,
    ):
        self.tokenizer = tokenizer
        self.download_path = download_path
        self.split = split
        self.tokenized_field_name = tokenized_field_name
        self.tokenize_batch_size = tokenize_batch_size
        self.dataloader_batch_size = dataloader_batch_size
        self.dataset_columns = dataset_columns
        self.test_size = test_size
        self.storage_dir = STORAGE_DIR
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.cache_dir = CACHE_DIR
        self.hub_dir = HUB_DIR
        self.cached_dirs_to_remove = [self.storage_dir + "/manu___project_gutenberg"]

        # Enable logging
        self.log = logging.getLogger(__name__)

        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def clear_gutenberg_non_english_arrow_files(self):
        """Delete .arrow files that are non-english"""
        if not os.path.exists(self.storage_dir):
            self.log.info(f"Directory {self.storage_dir} does not exist.")
            return

        for root, _, files in os.walk(self.storage_dir):
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
                        # self.log.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        self.log.info(f"Failed to delete {file_path}: {e}")

    def clear_dirs(self, directories):
        """Remove cached dataset directories."""
        for dir in directories:
            shutil.rmtree(dir)

    def clear_lock_files(self):
        # remove .lock files
        files_to_delete = glob.glob(os.path.join(self.hub_dir, "*.lock"))
        for file_path in files_to_delete:
            os.remove(file_path)

    def clear_gutenberg_arrow_files(self):
        """Remove .arrow files to save space on disk"""
        if not os.path.exists(self.data_dir):
            self.log.info(f"Directory {self.data_dir} does not exist.")
            return

        # Remove project_gutenberg .arrow files to save space
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".arrow") and file.startswith("project_gutenberg"):
                    # Check if the filename is not followed by "-en"
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        # self.log.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        self.log.info(f"Failed to delete {file_path}: {e}")

    def clear_raw_dataset_files(self):
        """Clear all raw files from /home/shared_storage/hub/datasets--<download-path>"""

        # path is path to the huggingface dataset download directory
        data_dir = os.path.join(HUB_DIR, "datasets--" + self.download_path.replace("/", "--"))
        if os.path.exists(data_dir):
            # self.log.info(f"Hub dir: {data_dir}")
            shutil.rmtree(data_dir)

    def load_tokenized_dataset(self):
        """Check for tokenized cache. If missing, load raw dataset, tokenize, and save tokenized dataset."""
        if os.path.exists(self.data_dir):
            self.log.info("Loading tokenized dataset from cache...")
            dataset = load_from_disk(self.data_dir)
        else:
            self.log.info("Downloading and preparing the raw dataset...")

            # Load dataset
            dataset = load_dataset(
                path=self.download_path, split=self.split, cache_dir=self.cache_dir
            )
            dataset = dataset.select(range(10))

            # Remove raw, encoded files
            # self.clear_raw_dataset_files()
            # self.clear_gutenberg_non_english_arrow_files()

            # Tokenize and save the dataset
            dataset = self.tokenize_data(dataset, self.tokenize_batch_size)

        # Print the total number of rows before processing
        self.log.info(f"Number of rows in the dataset before processing: {len(dataset)}")
        return dataset

    def create_huggingface_train_test_split(self, dataset):
        """Create train/test split on a huggingface dataset from datasets library."""
        # Create a train/test split
        split_dataset = dataset.train_test_split(test_size=self.test_size, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        return train_dataset, test_dataset

    def convert_train_test_split_to_pytorch(self, dataset):
        train_dataset, test_dataset = self.create_huggingface_train_test_split(dataset)
        train_loader = self.convert_dataset_to_pytorch(train_dataset, shuffle=True)
        test_loader = self.convert_dataset_to_pytorch(test_dataset, shuffle=False)
        return train_loader, test_loader

    def convert_dataset_to_pytorch(self, dataset, shuffle):
        # Convert both splits to PyTorch tensors
        dataset.set_format("torch", columns=self.dataset_columns)
        return DataLoader(dataset, batch_size=self.dataloader_batch_size, shuffle=shuffle)

    def get_calibration_dataset(self, dataset, fraction=0.1):
        """Takes a huggingface `datasets` dataset, samples by N%, and converts to PyTorch tensors."""
        # Randomly sample dataset by fraction (%)
        subset_size = int(len(dataset) * fraction)
        subset_dataset = dataset.shuffle(seed=42).select(range(subset_size))

        # Convert to DataLoader
        calibration_loader = self.convert_dataset_to_pytorch(subset_dataset, shuffle=False)
        return calibration_loader

    def tokenize_data(self, dataset, batch_size):
        """Tokenize the dataset and save it to the cache directory."""
        if "input_ids" in dataset.column_names:
            self.log.info("Dataset already tokenized.")
        else:
            self.log.info("Tokenizing dataset...")
            dataset = dataset.map(self.tokenizer_function, batched=True, batch_size=batch_size)

            # Clear the dataset cache
            # self.clear_gutenberg_arrow_files()
            # self.clear_dirs(self.cached_dirs_to_remove)
            # self.clear_lock_files()

            # Save tokenized dataset to the cache directory
            # self.log.info("Saving tokenized dataset to cache...")
            dataset.save_to_disk(self.data_dir)

        return dataset

    def tokenizer_function(self, data):
        """Tokenization logic for the dataset."""
        return self.tokenizer(
            data[self.tokenized_field_name],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def get_vocab_size(self, dataset):
        """Calculate the vocabulary size based on the maximum token index in the `input_ids` field of the dataset."""
        # Validate `input_ids` exists
        if "input_ids" not in dataset.column_names:
            raise ValueError("The dataset does not contain an `input_ids` column.")

        # Find max token index in `input_ids`
        max_indices = dataset.map(
            lambda x: {"max_index": max(x["input_ids"])},
            batched=False,
            remove_columns=dataset.column_names,
        )

        # Get global maximum from max_index column
        max_token_index = max(max_indices["max_index"])
        return max_token_index + 1
