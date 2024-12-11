# main.py
import argparse
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_preparation import DataPreparation
from model_trainer import ModelTrainer
from model_quantizer import ModelQuantizer
from model_distiller import ModelDistiller
from model_evaluator import ModelEvaluator
from evaluate_metrics import EvaluateMetrics


def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run Specific Steps of the Model Training Pipeline"
    )
    parser.add_argument("--train", action="store_true", help="Run model training step.")
    parser.add_argument("--ptq", action="store_true", help="Run Post-Training Quantization (PTQ).")
    parser.add_argument("--qat", action="store_true", help="Run Quantization Aware Training (QAT).")
    parser.add_argument("--distill", action="store_true", help="Run Knowledge Distillation.")
    parser.add_argument("--evaluate", action="store_true", help="Run model evaluation step.")

    args = parser.parse_args()

    # general
    model_name = "meta-llama/Llama-3.2-1B"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    saved_data_path = "data/project_gutenberg"
    model_results_path = "results/llama_3-2_1B"

    if args.train:
        num_labels = 2
        epochs = 3
        batch_size = 8
        learning_rate = 2e-5
        wandb_project = "aura-ai"

        # if data not cached: download, tokenize, and save data locally
        # else, load from cache
        data_preparation = DataPreparation(
            model_name=model_name,
            huggingface_token=huggingface_token,
            download_path="manu/project_gutenberg",
            split="en",
            tokenized_field_name="text",
        )
        dataset = data_preparation.prepare_data()

        model_trainer = ModelTrainer(
            model_name=model_name,
            saved_data_path=saved_data_path,
            model_results_path=model_results_path,
            num_labels=num_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            wandb_project=wandb_project,
        )
        # Check if model is already trained
        if os.path.exists(model_results_path):
            ##### NEED DONE model_trainer.load_model() Idk how wandb works
            pass
        else:
            model_trainer.train_model()

    if args.ptq or args.distill:
        # Step 2: Model Quantization (Destruction) #PTQ
        model_quantizer = ModelQuantizer()
        if os.path.exists("./results/llama_literature_quantized"):
            model_quantizer.model.load_state_dict(
                torch.load("./results/llama_literature_quantized")
            )
        else:
            model_quantizer.ptq()
            model_quantizer.save_model()

    if args.qat:
        # Step 3.1 QAT (Recovery mthd. 1)
        model_trainer = ModelTrainer(
            model_name=model_name,
            saved_data_path=saved_data_path,
            model_results_path=model_results_path,
            num_labels=num_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            wandb_project=wandb_project,
        )
        if os.path.exists(model_results_path):
            ##### NEED DONE model_trainer.load_model() Idk how wandb works
            pass
        else:
            model_trainer.qat()

    if args.distill:

        if os.path.exists("./results/llama_literature_quantized"):
            model_quantizer.model.load_state_dict(
                torch.load("./results/llama_literature_quantized")
            )
        else:
            model_quantizer.ptq()
            model_quantizer.save_model()

        # Step 3.2 Knowledge Distialation (Recovery mthd. 2)
        teacher_model = None  # placeholder
        model_distiller = ModelDistiller(teacher_model, model_quantizer)
        train_loader = None  # placeholder
        T = 1.0  # placeholder
        soft_target_loss_weight = 0.5  # placeholder
        ce_loss_weight = 0.5  # placeholder
        model_distiller.train_knowledge_distillation(
            train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight
        )
        model_distiller.save_model()  # optionally provide save_path

    if args.evaluate:
        # Llama 1B
        model_name = "meta-llama/Llama-3.2-1B"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_evaluator = ModelEvaluator(
            model_name="Llama-3_2-1B", model=model, tokenizer=tokenizer
        )
        results = model_evaluator.evaluate()

        # Clear CUDA memory after 1B
        del model_evaluator
        del model
        del tokenizer
        torch.cuda.empty_cache()

        # Llama 3B
        model_name = "meta-llama/Llama-3.2-3B"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_evaluator = ModelEvaluator(
            model_name="Llama-3_2-3B", model=model, tokenizer=tokenizer
        )
        results = model_evaluator.evaluate()

        # Clear CUDA memory after 3B
        del model_evaluator
        del model
        del tokenizer
        torch.cuda.empty_cache()

        metrics = EvaluateMetrics()
        metrics.plot_all_metrics()


if __name__ == "__main__":
    main()
