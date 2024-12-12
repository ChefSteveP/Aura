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
    
    def ptq(self):
        self.model.eval()
        self.model.qconfig = torch.quantization.default_qconfig
        #self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        torch.quantization.convert(self.model, inplace=True)
    
    def prepare_qat(self):
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
        # train and convert after training
