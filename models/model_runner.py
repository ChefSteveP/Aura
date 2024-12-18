import os
import time
import logging
import argparse
from torch.utils.data import DataLoader
from huggingface_hub import login
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM
from model_utils import ModelUtils
from model_evaluator import ModelEvaluator
from plot_metrics import PlotMetrics
from prompt_generator import PromptGenerator
from model_quantizer import ModelQuantizer
from model_distiller import ModelDistiller
from model_distiller import RandSnipitDataset
from constants import RESULTS_DATA_DIR, RESULTS_PLOTS_DIR, PROMPTS, START_PROMPT


class ModelRunner:
    def __init__(self):
        self.log = logging.getLogger(__name__)
        login(os.getenv("HUGGINGFACE_TOKEN"))  # login to huggingface
        self.model_utils = ModelUtils()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run Specific Steps of the Model Training Pipeline"
        )
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

    def run_ptq(self, model_name, tokenizer, device, file_path):
        q = ModelQuantizer()
        model = AutoAWQForCausalLM.from_pretrained(model_name, device_map=device)
        ptq_model = q.ptq(model, tokenizer, file_path)
        return ptq_model

    # Distill functions
    def run_distill(self, teacher_model_name, student_model_name, dataset, device, file_path):
        self.model_utils.log_memory(message=f"Before loading KD models into mem")
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
        self.model_utils.log_memory(message=f"After loading KD models into mem")

        model_distiller = ModelDistiller()
        self.model_utils.log_memory(message=f"After KD initialization")

        train_loader = self.load_distill_dataset(
            dataset,
            batch_size=1,
            segment_length=200,
        )
        self.model_utils.log_memory(message=f"After Distill Dataset")

        student_model = model_distiller.train_knowledge_distillation(
            train_loader,
            epochs=1,
            learning_rate=2e-5,
            T=1.0,
            soft_target_loss_weight=0.5,
            ce_loss_weight=0.5,
            device=device,
            student=student_model,
            teacher=teacher_model,
        )
        self.model_utils.log_memory(message=f"After KD")

        self.log.info("Save model")
        student_model.save_pretrained(file_path)
        del teacher_model
        return student_model

    def load_distill_dataset(self, dataset, batch_size, segment_length=200):
        segment_dataset = RandSnipitDataset(dataset, segment_length)
        # data_preparation.inspect_one_batch(train_loader)
        return DataLoader(segment_dataset, batch_size=batch_size, shuffle=True)

    # Evaluate functions
    def run_evaluate(self, models, tokenizer, device):
        self.model_utils.clear_csv_files(RESULTS_DATA_DIR)
        eval_dataset = self.get_eval_dataset()

        total_time = 0
        for model_name, model in models.items():
            # self.log.info(f"Run evaluator for {model_name}")
            model_time = self.evaluate_model(model_name, model, tokenizer, device, eval_dataset)
            total_time += model_time
            model.to("cpu")

        self.log.info(f"{total_time:.2f} sec to complete evaluation loop.")
        self.plot_evaluation_metrics()

    def plot_evaluation_metrics(self):
        plot_metrics = PlotMetrics(RESULTS_DATA_DIR, RESULTS_PLOTS_DIR)
        plot_metrics.plot_all_metrics()

    def evaluate_model(self, model_name, model, tokenizer, device, eval_dataset):
        """
        Steps include:
        - Print CUDA memory before running evaluator
        - Run evaluator on one model
        - Print CUDA memory after memory cleared via clear_cuda_memory()
        """
        start_time = time.time()
        self.model_utils.log_memory(message=f"{model_name} before")
        eval = ModelEvaluator(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=eval_dataset,
            results_dir=RESULTS_DATA_DIR,
        )
        eval.evaluate()
        eval.clear_cuda_memory()
        self.model_utils.log_memory(message=f"{model_name} after")
        total_time = time.time() - start_time
        self.log.info(f"{total_time:.2f} sec to complete {model_name}.")
        return total_time

    def get_eval_dataset(self):
        pg = PromptGenerator()
        return pg.generate_eval_dataset(START_PROMPT, PROMPTS)