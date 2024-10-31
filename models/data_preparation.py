from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from huggingface_hub import login
import pickle
import os
from multiprocessing import Pool


class DataPreparation:
    def __init__(
        self,
        model_name,
        huggingface_token=None,
        saved_data_path=None,
        download_path=None,
        split=None,
        tokenized_field_name=None,
    ):
        self.huggingface_token = huggingface_token
        self.saved_data_path = saved_data_path
        self.download_path = download_path
        self.split = split
        self.tokenized_field_name = tokenized_field_name

        # login to huggingface
        if self.huggingface_token:
            login(self.huggingface_token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if pading token exists, else add one
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def get_dataset(self):
        # Check if dataset in cache
        if os.path.exists(self.saved_data_path):
            dataset = load_from_disk(self.saved_data_path)
            print("Dataset found in cache.")
        else:
            # Load from Hugging Face
            dataset = load_dataset(path=self.download_path, split=self.split)
            print("Downloaded dataset.")

        return dataset

    def tokenize_data(self, dataset):
        if "input_ids" in dataset.column_names:
            print("Dataset already tokenized.")
        else:
            print("Dataset is not tokenized. Tokenizing now...")
            dataset = dataset.map(self.tokenizer_function, batched=True, batch_size=100)
            dataset.save_to_disk(self.saved_data_path)
            print("Tokenized dataset saved to disk.")

        return dataset

    def tokenizer_function(self, data):
        return self.tokenizer(
            data[self.tokenized_field_name], truncation=True, padding="max_length", max_length=512
        )

    def prepare_data(self):
        dataset = self.get_dataset()

        # Shuffle data
        # dataset = dataset.shuffle()

        # Select N elements
        dataset = dataset.select(range(205))

        # batching
        dataset = self.tokenize_data(dataset)

        print("data preparation complete")
        return dataset
