# model_evaluator.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class ModelEvaluator:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("./llama_literature_quantized")
        self.tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-small")

    def evaluate(self):
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        output = generator(
            "Once upon a midnight dreary, while I pondered, weak and weary,", max_length=50
        )
        print("Generated Text:", output[0]["generated_text"])
