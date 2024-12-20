# main.py
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_runner import ModelRunner
from constants import (
    LLAMA_1B_MODEL_NAME,
    LLAMA_3B_MODEL_NAME,
    PTQ_1B_FILE_PATH,
    PTQ_3B_FILE_PATH,
    KD_1B_FILE_PATH,
    KD_PTQ_1B_FILE_PATH,
    LLAMA_3B_TOKENIZER,
)


def main():
    # Initialize logging, classes, and input parse args
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")
    runner = ModelRunner()
    args = runner.parse_args()

    # General initializations for models/tokenizers/datasets
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_3B_TOKENIZER)

    if args.dataprep:
        dataset = runner.run_data_prep(tokenizer)
        logging.info(f"{len(dataset)} rows")

    if args.ptq:
        # Run on Llama 3B model and save to PTQ_3B_FILE_PATH
        runner.run_ptq(LLAMA_1B_MODEL_NAME, tokenizer, PTQ_1B_FILE_PATH)
        runner.run_ptq(LLAMA_3B_MODEL_NAME, tokenizer, PTQ_3B_FILE_PATH)

    if args.distill:
        kd_1B_model = runner.run_distill(
            tokenizer=tokenizer,
            teacher_model_name=LLAMA_3B_MODEL_NAME,
            student_model_name=LLAMA_1B_MODEL_NAME,
            file_path=KD_1B_FILE_PATH,
            T=2.0,
            epochs=3,
            samples=None,
        )

    if args.evaluate:
        # Define models to evaluate
        models = {
            "Llama-1B": AutoModelForCausalLM.from_pretrained(
                LLAMA_1B_MODEL_NAME, device_map="auto"
            ),
            "Llama-3B": AutoModelForCausalLM.from_pretrained(
                LLAMA_3B_MODEL_NAME, device_map="auto"
            ),
            "Llama-1B-PTQ": AutoModelForCausalLM.from_pretrained(
                PTQ_1B_FILE_PATH, device_map="auto"
            ),
            "Llama-3B-PTQ": AutoModelForCausalLM.from_pretrained(
                PTQ_3B_FILE_PATH, device_map="auto"
            ),
            "Llama-1B-KD": AutoModelForCausalLM.from_pretrained(KD_1B_FILE_PATH, device_map="auto"),
            "Llama-1B-KD-PTQ": AutoModelForCausalLM.from_pretrained(
                KD_PTQ_1B_FILE_PATH, device_map="auto"
            ),
        }

        # Run evaluator
        runner.run_evaluate(models, tokenizer)


if __name__ == "__main__":
    main()
