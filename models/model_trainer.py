import wandb
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_from_disk
import os
import torch


class ModelTrainer:
    def __init__(
        self,
        model_name,
        saved_data_path,
        model_results_path,
        num_labels=2,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        fp16=True,
        wandb_project="aura-ai",
    ):
        self.model_name = model_name
        self.saved_data_path = saved_data_path
        self.model_results_path = model_results_path
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fp16 = fp16
        self.wandb_project = wandb_project

        # init wandb
        wandb.init(
            project=self.wandb_project,
            config={
                "model_name": self.model_name,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
        )

        # check if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        # load tokenizer and model; move to CUDA if applicable
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_labels
        ).to(self.device)

    def load_data_from_path(self):
        # ensure data path exists
        if not os.path.exists(self.saved_data_path):
            raise FileNotFoundError(
                f"The specified data path {self.saved_data_path} does not exist."
            )

        # load data from cache
        dataset = load_from_disk(self.saved_data_path)

        # ensure data is tokenized
        required_columns = ["input_ids", "attention_mask"]
        for column in required_columns:
            if column not in dataset.column_names:
                raise ValueError(
                    f"The dataset is not tokenized properly. Missing column '{column}' in the dataset."
                )

        return dataset

    def train_model(self):
        # load and train/test split data
        dataset = self.load_data_from_path()
        dataset = dataset.train_test_split(test_size=0.2)

        training_args = TrainingArguments(
            output_dir=self.model_results_path,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            report_to="wandb",
            logging_dir=f"{self.model_results_path}/logs",
            save_total_limit=2,
            fp16=self.fp16,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        print("Starting training on device:", self.device)
        trainer.train()
        print("Training complete. Saving model...")
        trainer.save_model(self.model_results_path)
        print(f"Model saved to {self.model_results_path}")

        wandb.finish()
