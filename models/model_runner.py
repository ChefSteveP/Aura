import os
import time
import logging
import argparse
import torch
from huggingface_hub import login
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM
from model_utils import ModelUtils
from model_evaluator import ModelEvaluator
from plot_metrics import PlotMetrics
from prompt_generator import PromptGenerator
from model_quantizer import ModelQuantizer
from model_distiller import ModelDistiller
from data_preparation import DataPreparation
from constants import (
    RESULTS_DATA_DIR,
    RESULTS_PLOTS_DIR,
    PROMPTS,
    START_PROMPT,
    DATA_DIR,
)


class ModelRunner:
    def __init__(self):
        self.log = logging.getLogger(__name__)
        login(os.getenv("HUGGINGFACE_TOKEN"))  # login to huggingface
        self.model_utils = ModelUtils()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run Specific Steps of the Model Training Pipeline"
        )
        parser.add_argument("--dataprep", action="store_true", help="Run data processing step.")
        parser.add_argument("--train", action="store_true", help="Run model training step.")
        parser.add_argument("--dq", action="store_true", help="Run Dynamice Quantization.")
        parser.add_argument(
            "--ptq", action="store_true", help="Run Post-Training Quantization (PTQ)."
        )
        parser.add_argument(
            "--qat", action="store_true", help="Run Quantization Aware Training (QAT)."
        )
        parser.add_argument("--distill", action="store_true", help="Run Knowledge Distillation.")
        parser.add_argument("--evaluate", action="store_true", help="Run model evaluation step.")

        return parser.parse_args()

    def run_data_prep(self, tokenizer, samples=None):
        data_preparation = DataPreparation(
            tokenizer,
            download_path="manu/project_gutenberg",
            split="en",
            tokenized_field_name="text",
            tokenize_num_proc=12,
            dataloader_batch_size=1,
            data_dir=DATA_DIR,
        )
        data_loader = data_preparation.create_data_loader(block_size=256)
        # data_preparation.inspect_samples(data_loader, tokenizer, 10)
        return data_loader

    def run_ptq(self, model_name, tokenizer, file_path):
        q = ModelQuantizer()
        model = AutoAWQForCausalLM.from_pretrained(model_name, device_map="auto")
        ptq_model = q.ptq(model, tokenizer, file_path)
        return ptq_model

    # Distill functions
    def run_distill(
        self, tokenizer, teacher_model_name, student_model_name, file_path, samples, T, epochs
    ):
        train_loader = DataPreparation(
            tokenizer,
            download_path="manu/project_gutenberg",
            split="en",
            tokenized_field_name="text",
            tokenize_num_proc=12,
            dataloader_batch_size=8,
            data_dir=DATA_DIR,
        ).create_data_loader(samples=samples)

        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, device_map="auto")
        student_model = AutoModelForCausalLM.from_pretrained(student_model_name, device_map="auto")
        # self.model_utils.log_memory(message=f"After loading KD models into mem")
        # print(torch.cuda.memory_summary())

        model_distiller = ModelDistiller()
        student_model, teacher_model = model_distiller.train_knowledge_distillation(
            train_loader=train_loader,
            epochs=epochs,
            learning_rate=2e-5,
            T=T,
            alpha=0.7,
            teacher=teacher_model,
            student=student_model,
        )
        # self.model_utils.log_memory(message=f"After KD")
        student_model.save_pretrained(file_path)
        return student_model

    # Evaluate functions
    def run_evaluate(self, models, tokenizer):
        self.model_utils.clear_csv_files(RESULTS_DATA_DIR)
        eval_dataset = self.get_eval_dataset()

        total_time = 0
        for model_name, model in models.items():
            # self.log.info(f"Run evaluator for {model_name}")
            model_time = self.evaluate_model(model_name, model, tokenizer, eval_dataset)
            total_time += model_time

        self.log.info(f"{total_time:.2f} sec to complete evaluation loop.")
        self.plot_evaluation_metrics()

    def plot_evaluation_metrics(self):
        plot_metrics = PlotMetrics(RESULTS_DATA_DIR, RESULTS_PLOTS_DIR)
        plot_metrics.plot_all_metrics()

    def evaluate_model(self, model_name, model, tokenizer, eval_dataset):
        """
        Steps include:
        - Print CUDA memory before running evaluator
        - Run evaluator on one model
        - Print CUDA memory after memory cleared via clear_cuda_memory()
        """
        start_time = time.time()
        torch.cuda.empty_cache()
        self.model_utils.log_memory(message=f"{model_name} before")
        eval = ModelEvaluator(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            results_dir=RESULTS_DATA_DIR,
        )
        eval.evaluate()
        torch.cuda.empty_cache()
        self.model_utils.log_memory(message=f"{model_name} after")
        total_time = time.time() - start_time
        self.log.info(f"{total_time:.2f} sec to complete {model_name}.")
        return total_time

    def get_eval_dataset(self):
        pg = PromptGenerator()
        return pg.generate_eval_dataset(START_PROMPT, PROMPTS)
