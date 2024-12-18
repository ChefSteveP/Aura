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
)


def main():
    # Initialize logging, classes, and input parse args
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")
    runner = ModelRunner()
    args = runner.parse_args()

    # # General initializations for models/tokenizers/datasets
    # tokenizer = AutoTokenizer.from_pretrained(LLAMA_3B_MODEL_NAME)

    # data_preparation = DataPreparation(
    #     tokenizer,
    #     download_path="manu/project_gutenberg",
    #     split="en",
    #     tokenized_field_name="text",
    # )

    # dataset = data_preparation.load_tokenized_dataset()
    # dataset = dataset.select(range(10))

    if args.train:
        # TODO: finish training loop
        pass

    if args.ptq:
        # Run on Llama 3B model and save to PTQ_3B_FILE_PATH
        runner.run_ptq(LLAMA_1B_MODEL_NAME, tokenizer, PTQ_1B_FILE_PATH)
        runner.run_ptq(LLAMA_3B_MODEL_NAME, tokenizer, PTQ_3B_FILE_PATH)

    if args.qat:
        # TODO: finish QAT loop
        pass

    if args.distill:
        # Step 3.2 Knowledge Distialation (Recovery mthd. 2)

        # hyper parameters
        epochs = 1
        batch_size = 1
        learning_rate = 2e-5
        T = 1.0  # placeholder
        soft_target_loss_weight = 0.5  # placeholder
        ce_loss_weight = 0.5  # placeholder

        # grab random 2000 tokens from each book.
        segment_dataset = RandSnipitDataset(dataset, segment_length=200)  # max length 512?
        train_loader = DataLoader(segment_dataset, batch_size=batch_size, shuffle=True)
        logging.info("Initialized data loader")

        # Inspect one batch
        # batch = next(iter(train_loader))
        # for key, value in batch.items():
        #     print(f"Title: {key}")
        #     if isinstance(value, torch.Tensor):
        #         print("Shape:", value.shape)
        #     else:
        #         print("Value type:", type(value))
        #     print("-" * 40)

        # Load Teacher Model (Llama 3B)
        # llama_1b_name = "meta-llama/Llama-3.2-1B" # For direct huggingface
        try:
            # teacher_model = AutoModelForCausalLM.from_pretrained(llama_1b_name, cache_dir=MODELS_DIR) # For direct huggingface
            teacher_model = torch.load(LLAMA_1B_FILE_PATH, weights_only=False)
            logging.info(f"Loaded teacher model: {LLAMA_1B_FILE_PATH}")
        except Exception as e:
            logging.error(f"Error loading teacher model: {e}")
            exit(1)

        # Load Student Model (LLaMA 1B)
        # This will be swapped out with the ptq quantized model later
        # llama_1b_name = "meta-llama/Llama-3.2-1B" # For direct huggingface
        try:
            # student_model = AutoModelForCausalLM.from_pretrained(llama_1b_name, cache_dir=MODELS_DIR) # For direct huggingface
            student_model = torch.load(LLAMA_1B_FILE_PATH, weights_only=False)
            logging.info(f"Loaded student model: {LLAMA_1B_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to load student model: {LLAMA_1B_FILE_PATH}. Error: {e}")
            exit(1)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"

        logging.info("Initialized model distiller")
        model_distiller = ModelDistiller(teacher=teacher_model, student=student_model)

        logging.info("Starting knowledge distillation training...")
        try:
            model_distiller.train_knowledge_distillation(
                train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight
            )
            logging.info("Knowledge distillation training completed.")
        except Exception as e:
            logging.error(f"Failed to train knowledge distillation: {e}")
            exit(1)

        distilled_model_path = "/home/shared_storage/models/llama_1B_dist_llama_1B.pt"
        try:
            model_distiller.save_model(distilled_model_path)  # optionally provide save_path
        except Exception as e:
            logging.error(f"Failed to save distilled model: {e}")
            exit(1)

    if args.evaluate:
        # Define models to evaluate
        kd_model = torch.load(LLAMA_1B_FILE_PATH)
        print(type(kd_model))
        models = {
            # "Llama-1B": AutoModelForCausalLM.from_pretrained(LLAMA_1B_MODEL_NAME),
            # "Llama-3B": AutoModelForCausalLM.from_pretrained(LLAMA_3B_MODEL_NAME),
            # "Llama-1B-PTQ": AutoModelForCausalLM.from_pretrained(PTQ_1B_FILE_PATH),
            # "Llama-3B-PTQ": AutoModelForCausalLM.from_pretrained(PTQ_3B_FILE_PATH),
            "Llama-1B-KD": torch.load(LLAMA_1B_FILE_PATH),
        }

        # Run evaluator
        # runner.run_evaluate(models, tokenizer)


if __name__ == "__main__":
    main()
