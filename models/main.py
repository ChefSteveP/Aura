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
from plot_metrics import PlotMetrics
from constants import STORAGE_DIR, MODELS_DIR, RESULTS_DATA_DIR, RESULTS_PLOTS_DIR
from model_utils import ModelUtils

model_utils = ModelUtils()


def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format="\n%(levelname)s - %(message)s")

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Run Specific Steps of the Model Training Pipeline"
    )
    parser.add_argument("--train", action="store_true", help="Run model training step.")
    parser.add_argument("--dq", action="store_true", help="Run Dynamice Quantization.")
    parser.add_argument("--ptq", action="store_true", help="Run Post-Training Quantization (PTQ).")
    parser.add_argument("--qat", action="store_true", help="Run Quantization Aware Training (QAT).")
    parser.add_argument("--distill", action="store_true", help="Run Knowledge Distillation.")
    parser.add_argument("--evaluate", action="store_true", help="Run model evaluation step.")

    args = parser.parse_args()

    # general
    llama_1b_model_name = "meta-llama/Llama-3.2-1B"
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

    saved_data_path = "data/project_gutenberg"
    model_results_path = "results/llama_3-2_1B"

    data_preparation = DataPreparation(
        model_name=llama_1b_model_name,
        huggingface_token=huggingface_token,
        download_path="manu/project_gutenberg",
        split="en",
        tokenized_field_name="text",
        cache_dir=STORAGE_DIR,
    )
    dataset = data_preparation.get_dataset()
    dataset = dataset.select(range(10))

    if args.train:
        num_labels = 2
        epochs = 3
        batch_size = 8
        learning_rate = 2e-5
        wandb_project = "aura-ai"

        model_trainer = ModelTrainer(
            model_name=llama_1b_model_name,
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

    if args.dq:
        quantizer = ModelQuantizer()
        dq_model = quantizer.dynamic_quantize_model(llama_1b_model_name)
        file_path = MODELS_DIR + "/Llama_1B_dynamic_quantized"
        model_utils.save_model(dq_model, file_path)

    if args.ptq:
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
            model_name=llama_1b_model_name,
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
        model_utils.clear_csv_files(RESULTS_DATA_DIR)

        # Llama 1B
        llama_1b_model, llama_1b_tokenizer = model_utils.load_tokenizer_and_model(
            "meta-llama/Llama-3.2-1B", True
        )
        llama_1b_evaluator = ModelEvaluator(
            model_name="Llama-3_2-1B",
            model=llama_1b_model,
            tokenizer=llama_1b_tokenizer,
            results_dir=RESULTS_DATA_DIR,
        )
        llama_1b_evaluator.evaluate()

        # Clear CUDA memory after 1B
        del llama_1b_evaluator
        del llama_1b_model
        del llama_1b_tokenizer
        torch.cuda.empty_cache()

        # Llama 3B
        llama_3b_model, llama_3b_tokenizer = model_utils.load_tokenizer_and_model(
            "meta-llama/Llama-3.2-3B", True
        )
        llama_3b_evaluator = ModelEvaluator(
            model_name="Llama-3_2-3B",
            model=llama_3b_model,
            tokenizer=llama_3b_tokenizer,
            results_dir=RESULTS_DATA_DIR,
        )
        llama_3b_evaluator.evaluate()

        # Clear CUDA memory after 3B
        del llama_3b_evaluator
        del llama_3b_model
        del llama_3b_tokenizer
        torch.cuda.empty_cache()

        plot_metrics = PlotMetrics(RESULTS_DATA_DIR, RESULTS_PLOTS_DIR)
        plot_metrics.plot_all_metrics()


if __name__ == "__main__":
    main()
