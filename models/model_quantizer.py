# model_quantizer.py
import torch
from transformers import AutoModelForCausalLM


class ModelQuantizer:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("./llama_literature_pruned")

    def quantize_model(self):
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def save_model(self):
        self.model.save_pretrained("./llama_literature_quantized")
        print(
            "Model quantization complete. Quantized model saved to './llama_literature_quantized'."
        )
