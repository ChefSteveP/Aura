from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import login
import os


class DataPreparation:
    def __init__(
        self,
        model_name,
        huggingface_token=None,
        download_path=None,
        split=None,
        tokenized_field_name=None,
    ):
        self.huggingface_token = huggingface_token
        self.download_path = download_path
        self.split = split
        self.tokenized_field_name = tokenized_field_name

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

    def get_dataset(self):
        # Cached in ~/.cache/huggingface
        print("Checking for dataset in Hugging Face cache...")
        dataset = load_dataset(path=self.download_path, split=self.split)
        print("Dataset loaded.")
        return dataset

    def tokenize_data(self, dataset):
        if "input_ids" in dataset.column_names:
            print("Dataset already tokenized.")
        else:
            print("Tokenizing dataset...")
            dataset = dataset.map(self.tokenizer_function, batched=True, batch_size=100)
            print("Dataset tokenized and cached in Hugging Face default directory.")
        return dataset

    def tokenizer_function(self, data):
        return self.tokenizer(
            data[self.tokenized_field_name],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    def prepare_data(self):
        dataset = self.get_dataset()

        # Optionally shuffle or preprocess
        # dataset = dataset.shuffle()

        # Select a subset if needed
        dataset = dataset.select(range(205))

        # Tokenize the dataset
        dataset = self.tokenize_data(dataset)

        return dataset