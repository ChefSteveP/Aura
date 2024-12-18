import os
import logging
import argparse
from huggingface_hub import login
from awq import AutoAWQForCausalLM
from model_utils import ModelUtils
from model_evaluator import ModelEvaluator
from plot_metrics import PlotMetrics
from prompt_generator import PromptGenerator
from model_quantizer import ModelQuantizer
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

    def run_ptq(self, model_name, tokenizer, file_path):
        q = ModelQuantizer()
        model = AutoAWQForCausalLM.from_pretrained(model_name)
        ptq_model = q.ptq(model, tokenizer, file_path)
        return ptq_model

    # Evaluate functions
    def run_evaluate(self, models, tokenizer):
        self.model_utils.clear_csv_files(RESULTS_DATA_DIR)
        eval_dataset = self.get_eval_dataset()

        for model_name, model in models.items():
            self.log.info(f"Run evaluator for {model_name}")
            self.evaluate_model(model_name, model, tokenizer, eval_dataset)
            model.to("cpu")

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
        self.model_utils.print_cuda_memory(message=f"{model_name} before")
        eval = ModelEvaluator(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            dataset=eval_dataset,
            results_dir=RESULTS_DATA_DIR,
        )
        eval.evaluate()
        eval.clear_cuda_memory()
        self.model_utils.print_cuda_memory(message=f"{model_name} after")

    def get_eval_dataset(self):
        pg = PromptGenerator()
        return pg.generate_eval_dataset(START_PROMPT, PROMPTS)
