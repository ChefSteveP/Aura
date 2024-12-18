import logging
import torch
from transformers import AwqConfig


class ModelQuantizer:
    def __init__(self):
        self.log = logging.getLogger(__name__)

    def dynamic_quantize_model(self, model):
        """Apply dynamic quantization to the model."""
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model

    def ptq(self, model, tokenizer, file_path):
        """Quantize a model with Post-Training Quantization via AWQ. Then save to <file_path>"""

        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
        model.quantize(tokenizer, quant_config=quant_config)

        quantization_config = AwqConfig(
            bits=quant_config["w_bit"],
            group_size=quant_config["q_group_size"],
            zero_point=quant_config["zero_point"],
            version=quant_config["version"].lower(),
        ).to_dict()

        model.model.config.quantization_config = quantization_config
        model.save_quantized(file_path)
        return model
