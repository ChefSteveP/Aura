import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from model_utils import ModelUtils


class QuantizedModelWrapper(nn.Module, PyTorchModelHubMixin):
    """Wrapper to integrate a quantized PyTorch model with the Hugging Face Hub."""

    def __init__(self, quantized_model, config):
        super().__init__()
        self.quantized_model = quantized_model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.quantized_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Filter out non-tensor entries from the state_dict."""
        raw_state_dict = self.quantized_model.state_dict(*args, **kwargs)
        filtered_state_dict = {
            k: v for k, v in raw_state_dict.items() if isinstance(v, torch.Tensor)
        }
        return filtered_state_dict

    def save_pretrained(self, save_directory, **kwargs):
        """Save the model and its configuration."""
        # Save the configuration as JSON
        config_path = f"{save_directory}/config.json"
        with open(config_path, "w") as f:
            import json

            json.dump(self.config, f)

        # Save the model
        super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load model and its configuration."""
        config_path = f"{pretrained_model_name_or_path}/config.json"
        try:
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise ValueError("Configuration file not found. Unable to load model.")

        # Load the model weights
        state_dict = torch.load(
            f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location="cpu"
        )

        # Create a new quantized model and load the state_dict
        model = nn.Module()  # Replace with your actual model architecture
        model.load_state_dict(state_dict)
        return cls(model, config)


class ModelQuantizer:
    def __init__(self):
        self.model_utils = ModelUtils()

    def dynamic_quantize_model(self, model_name):
        """Apply dynamic quantization to the model."""
        model = self.model_utils.load_model(model_name)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        quantized_model_wrapper = QuantizedModelWrapper(quantized_model, model.config)
        return quantized_model_wrapper

    def ptq(self, model_name, calibration_data):
        """Apply post-training quantization (PTQ) to the model."""
        model = self.model_utils.load_model(model_name)
        model.eval()  # Evaluation mode
        model.qconfig = torch.quantization.default_qconfig  # Default quantization config

        # Run calibration data through the model
        torch.quantization.prepare(model, inplace=True)
        with torch.no_grad():
            for batch in calibration_data:
                model(**batch)

        # Convert the model to its quantized version
        torch.quantization.convert(model, inplace=True)
        quantized_model_wrapper = QuantizedModelWrapper(model, model.config)
        return quantized_model_wrapper
