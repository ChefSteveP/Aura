# main.py
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preparation import DataPreparation
from model_distiller import ModelDistiller
from model_distiller import RandSnipitDataset
from model_runner import ModelRunner
from constants import (
    LLAMA_1B_MODEL_NAME,
    LLAMA_3B_MODEL_NAME,
    LLAMA_1B_FILE_PATH,
    PTQ_1B_FILE_PATH,
    PTQ_3B_FILE_PATH,
    KD_1B_FILE_PATH,
    KD_1B_FILE_PATH_PT,
    PTQ_KD_1B_FILE_PATH,
    PTQ_KD_3B_FILE_PATH,
    PTQ_1B_CPU_FILE_PATH,
    PTQ_3B_CPU_FILE_PATH,
)


def main():
    # Initialize logging, classes, and input parse args
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")
    runner = ModelRunner()
    args = runner.parse_args()

    # General initializations for models/tokenizers/datasets
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_3B_MODEL_NAME)

    data_preparation = DataPreparation(
        tokenizer,
        download_path="manu/project_gutenberg",
        split="en",
        tokenized_field_name="text",
    )
    dataset = data_preparation.load_tokenized_dataset()
    # dataset = dataset.select(range(10))

    if args.train:
        # TODO: finish training loop
        pass

    if args.ptq:
        # Run on Llama 3B model and save to PTQ_3B_FILE_PATH
        runner.run_ptq(LLAMA_1B_MODEL_NAME, tokenizer, "cuda", PTQ_1B_FILE_PATH)
        runner.run_ptq(LLAMA_3B_MODEL_NAME, tokenizer, "cuda", PTQ_3B_FILE_PATH)

    if args.qat:
        # TODO: finish QAT loop
        pass

    if args.distill:
        ptq_kd_1B_model = runner.run_distill(
            teacher_model_name=LLAMA_1B_MODEL_NAME,
            # student_model_name=PTQ_1B_FILE_PATH,
            student_model_name=LLAMA_1B_MODEL_NAME,
            dataset=dataset,
            device="cpu",
            file_path=PTQ_KD_1B_FILE_PATH,
        )

        ptq_kd_3B_model = runner.run_distill(
            teacher_model_name=LLAMA_3B_MODEL_NAME,
            student_model_name=PTQ_3B_FILE_PATH,
            dataset=dataset,
            device="cuda",
            file_path=PTQ_KD_3B_FILE_PATH,
        )

    if args.evaluate:
        # Define models to evaluate
        models = {
            "Llama-1B": AutoModelForCausalLM.from_pretrained(LLAMA_1B_MODEL_NAME),
            "Llama-3B": AutoModelForCausalLM.from_pretrained(LLAMA_3B_MODEL_NAME),
            "Llama-1B-PTQ": AutoModelForCausalLM.from_pretrained(PTQ_1B_FILE_PATH),
            "Llama-3B-PTQ": AutoModelForCausalLM.from_pretrained(PTQ_3B_FILE_PATH),
            # "Llama-1B-PTQ-KD": AutoModelForCausalLM.from_pretrained(PTQ_KD_1B_FILE_PATH),
            # "Llama-3B-PTQ-KD": AutoModelForCausalLM.from_pretrained(PTQ_KD_3B_FILE_PATH),
        }

        # Run evaluator
        runner.run_evaluate(models, tokenizer, "cuda")


if __name__ == "__main__":
    main()
