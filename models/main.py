import logging
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preparation import DataPreparation
from model_trainer import ModelTrainer
from model_pruner import ModelPruner
from model_quantizer import ModelQuantizer
from model_evaluator import ModelEvaluator


def main():
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")

    # General variables
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = "meta-llama/Llama-3.2-1B"
    download_path = "manu/project_gutenberg"
    split = "en"
    tokenized_field_name = "text"

    # Prepare data
    data_preparation = DataPreparation(
        model_name=model_name,
        huggingface_token=huggingface_token,
        download_path=download_path,
        split=split,
        tokenized_field_name=tokenized_field_name,
    )
    dataset = data_preparation.get_dataset()

    # Additional pipeline steps can be added here
    logging.info("Data preparation complete.")

    # Evaluate model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_evaluator = ModelEvaluator(model=model, tokenizer=tokenizer)
    results = model_evaluator.evaluate()


if __name__ == "__main__":
    main()
