# model_pruner.py
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM


class ModelPruner:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def prune_model(self, amount=0.3):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")

    def save_model(self):
        self.model.save_pretrained("./llama_literature_pruned")
        print("Model pruning complete. Pruned model saved to './llama_literature_pruned'.")
