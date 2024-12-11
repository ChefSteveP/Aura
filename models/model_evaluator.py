import time
import logging
from collections import Counter
import torch
import textstat
from transformers import pipeline
import pandas as pd
from datetime import datetime


class ModelEvaluator:
    def __init__(self, model_name, model, tokenizer, results_dir="~/.cache/models/results/data"):
        self.model = model.to("cuda")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.formatted_datetime = self.get_formatted_datetime()
        self.results_dir = results_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generator_params = {
            "text_inputs": None,
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        self.generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=0, truncation=True
        )

        logger = logging.getLogger(__name__)

        # Storytelling prompts for evaluation
        self.prompts = [
            "Imagine you're a wizard who has been banished from your kingdom for practicing forbidden magic. How would you write a story about your journey to reclaim your honor and the challenges you face?",
            "You're the captain of a spaceship exploring a distant galaxy. One day, you encounter an abandoned alien station emitting a strange signal. How would you write a story about what happens next?",
            "A detective is called to investigate a murder at a remote mansion during a thunderstorm. How would you write a story from the detective's perspective as they unravel the mystery?",
            "You're a merchant traveling along the Silk Road in the 14th century. How would you write a story about an unusual event or encounter during your journey?",
            "The world has been devastated by a zombie plague leaving only a handful of survivors. How would you write a story about how you navigate this new world?",
        ]

    def evaluate(self):
        # Determine model size
        num_params, dtype, total_size_bytes, total_size_gb = self.compute_model_size()

        results = []
        for i, prompt in enumerate(self.prompts, start=1):
            # Compute the total time to generate the prompt then return total_time and output
            self.generator_params.update({"text_inputs": prompt})
            total_time, output = self.compute_total_time_and_generate_output()

            # Calculate number of tokens generated
            generated_text = output[0]["generated_text"]
            generated_tokens = len(self.tokenizer.tokenize(generated_text))

            # Calculate metrics
            ttft = self.compute_ttft(prompt)  # time to first token
            avg_time_per_token = self.compute_avg_time_per_token(total_time, generated_tokens)
            perplexity = self.compute_perplexity(generated_text)
            length = self.compute_length(generated_text)
            repetition_rate = self.compute_repetition_rate(generated_text)
            distinct_2 = self.distinct_n(generated_text, n=2)
            readability = self.compute_readability_score(generated_text)

            results.append(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "model": self.model_name,
                    "perplexity": perplexity,
                    "length": length,
                    "repetition_rate": repetition_rate,
                    "distinct_2": distinct_2,
                    "readability": readability,
                    "time_to_first_token": ttft,
                    "avg_time_per_token": avg_time_per_token,
                    "tokens_generated": generated_tokens,
                    "num_params": num_params,
                    "dtype": dtype,
                    "total_size_bytes": total_size_bytes,
                    "total_size_gb": total_size_gb,
                }
            )

            # logging.info(f"---------- PROMPT {i} ----------")
            # logging.info(f"\nEvaluating prompt: {prompt}\n")
            # logging.info(f"\nGenerated Text: {generated_text}\n")
            # logging.info(f"Metrics:")
            # logging.info(f"Time to First Token (TTFT): {ttft:.4f} seconds")
            # logging.info(f"Average Time per Token: {avg_time_per_token:.4f} seconds")
            # logging.info(f"Tokens Generated: {generated_tokens}")
            # logging.info(f"Perplexity: {perplexity}")
            # logging.info(f"Length: {length} words")
            # logging.info(f"Repetition Rate: {repetition_rate}")
            # logging.info(f"Distinct-2: {distinct_2}")
            # logging.info(f"Readability (Flesch Reading Ease): {readability}")
            # logging.info(f"Number of model parameters: {num_params}")
            # logging.info(f"Model Parameter Data Type: {dtype}")
            # logging.info(f"Model Size in Bytes: {total_size_bytes}")
            # logging.info(f"Model Size in GB: {total_size_gb}")

        # self.clear_cuda_memory()
        return self.save_df(results)

    def clear_cuda_memory(self):
        del self.model
        del self.generator
        torch.cuda.empty_cache()

    def get_formatted_datetime(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def save_df(self, results):
        file_path = f"{self.results_dir}/{self.model_name}_{self.formatted_datetime}.csv"
        df = pd.DataFrame(results)
        df.to_csv(file_path, index=False)
        return df

    def compute_model_size(self):
        # Calculate number of parameters
        num_params = sum(p.numel() for p in self.model.parameters())

        # Determine bytes per parameter
        dtypes = {p.dtype for p in self.model.parameters()}  # set to ensure distinct values
        if len(dtypes) > 1:
            raise ValueError("Model contains mixed parameter dtypes.")
        dtype = next(iter(dtypes))
        dtype_to_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int8: 1,
        }
        bytes_per_param = dtype_to_bytes.get(dtype, 4)

        # Compute total size in bytes and convert to GB
        total_size_bytes = num_params * bytes_per_param
        total_size_gb = total_size_bytes / (1024**3)

        return num_params, dtype, total_size_bytes, total_size_gb

    def compute_ttft(self, prompt):
        """Time to first token"""

        # Update params for only 1 new token
        params = self.generator_params
        params.update({"max_new_tokens": 1})

        # Time to first token calc
        start_time = time.time()
        self.generator(**params)
        ttft = time.time() - start_time
        return ttft

    def compute_total_time_and_generate_output(self):
        """Computes the total time taken for the generator to generate output and returns both."""
        start_time = time.time()
        output = self.generator(**self.generator_params)
        total_time = time.time() - start_time
        return total_time, output

    def compute_avg_time_per_token(self, total_time, generated_tokens):
        """Calculate avg time per token."""
        if generated_tokens > 0:
            return total_time / generated_tokens
        return 0

    def compute_perplexity(self, generated_text):
        device = self.model.device
        inputs = self.tokenizer(generated_text, return_tensors="pt", truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            loss = self.model(**inputs, labels=inputs["input_ids"]).loss.item()
        return torch.exp(torch.tensor(loss)).item()

    def compute_length(self, generated_text):
        return len(generated_text.split())

    def compute_repetition_rate(self, generated_text):
        words = generated_text.split()
        word_counts = Counter(words)
        repeated_words = sum(count > 1 for count in word_counts.values())
        return repeated_words / len(words) if len(words) > 0 else 0

    def distinct_n(self, generated_text, n=2):
        words = generated_text.split()
        n_grams = set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        return len(n_grams) / len(words) if len(words) > 0 else 0

    def compute_readability_score(self, generated_text):
        return textstat.flesch_reading_ease(generated_text)
