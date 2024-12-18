import re
import time
import logging
from collections import Counter
import torch
import textstat
from transformers import pipeline
import pandas as pd
from datetime import datetime


class ModelEvaluator:
    def __init__(self, model_name, model, tokenizer, device, dataset, results_dir):
        self.device = device
        self.model = model.to(self.device)
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = dataset
        self.formatted_datetime = self.get_formatted_datetime()
        self.results_dir = results_dir

        # Generator params for each prompt
        self.max_new_tokens = 100
        self.do_sample = True
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        self.eos_token_id = self.tokenizer.eos_token_id

        # Create generator pipeline for text-generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if self.device == "cpu" else 0,  # cuda == 0; cpu == -1
            truncation=True,
        )

        self.log = logging.getLogger(__name__)

    def _get_generator_params(self, **overrides):
        """Helper to construct generator parameters, allowing for overrides."""
        params = {
            "text_inputs": None,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_id": self.eos_token_id,
        }
        params.update(overrides)  # Apply any overrides
        return params

    def warmup_prompt(self):
        """Warm-up step to load the model and pipeline fully."""
        warmup_prompt = "This is a warm-up prompt to prepare the model."
        params = self._get_generator_params(text_inputs=warmup_prompt, max_new_tokens=1)
        output = self.generator(**params)
        # self.log.info(output[0]["generated_text"])

    def evaluate(self):
        """Evaluate a model on all prompts in self.prompts"""

        # self.log.info("Running warm-up to initialize model.")
        self.warmup_prompt()

        # self.log.info("Determine model size.")
        num_params, dtype, bytes_per_param, total_size_bytes, total_size_gb = (
            self.compute_model_size()
        )

        results = []
        for i, row in enumerate(self.dataset, start=1):
            self.log.info(f"Prompt {i}")

            params = self._get_generator_params(text_inputs=row["prompt"])
            total_time_ms, output = self.compute_total_time_and_generate_output(params)

            generated_text = output[0]["generated_text"]
            generated_tokens = len(self.tokenizer.tokenize(generated_text))
            response = self.extract_response(row["query"], generated_text)

            # Calculate metrics
            ttft = self.compute_ttft(row["prompt"])
            avg_time_per_token = self.compute_avg_time_per_token(total_time_ms, generated_tokens)
            perplexity = self.compute_perplexity(generated_text)
            length = self.compute_length(generated_text)
            repetition_rate = self.compute_repetition_rate(generated_text)
            distinct_2 = self.distinct_n(generated_text, n=2)
            readability = self.compute_readability_score(generated_text)

            results.append(
                {
                    "prompt": row["query"],
                    "generated_text": response,
                    "full_generated_text": generated_text,
                    "model_name": self.model_name,
                    "perplexity": perplexity,
                    "response_length": length,
                    "repetition_rate": repetition_rate,
                    "distinct_2": distinct_2,
                    "readability": readability,
                    "time_to_first_token": ttft,
                    "avg_time_per_token": avg_time_per_token,
                    "tokens_generated_per_response": generated_tokens,
                    "num_model_params": num_params,
                    "dtype": dtype,
                    "bytes_per_param": bytes_per_param,
                    "total_model_size_bytes": total_size_bytes,
                    "total_model_size_gb": total_size_gb,
                }
            )
            torch.cuda.empty_cache()
        return self.save_df(results)

    def clear_cuda_memory(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def get_formatted_datetime(self):
        """Utilized for functions such as save_df()."""
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def save_df(self, results):
        """Save a model's results to csv and return a df."""
        file_path = f"{self.results_dir}/{self.model_name}_{self.formatted_datetime}.csv"
        df = pd.DataFrame(results)
        df.to_csv(file_path, index=False)
        return df

    def compute_model_size(self):
        """
        Calculate model size attributes include:
            - num_params
            - dtype
            - bytes_per_param
            - total_size_bytes
            - total_size_gb
        """
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

        return num_params, dtype, bytes_per_param, total_size_bytes, total_size_gb

    def compute_ttft(self, prompt):
        """Time to first token"""

        # Update params for only 1 new token
        params = self._get_generator_params(text_inputs=prompt, max_new_tokens=1)

        # Time to first token calc
        start_time = time.time()
        self.generator(**params)
        ttft = (time.time() - start_time) * 1000
        return ttft

    def compute_total_time_and_generate_output(self, params):
        """Computes the total time taken for the generator to generate output and returns both."""
        start_time = time.time()
        output = self.generator(**params)
        total_time_ms = (time.time() - start_time) * 1000
        return total_time_ms, output

    def compute_avg_time_per_token(self, total_time_ms, generated_tokens):
        """Calculate avg time per token."""
        if generated_tokens > 0:
            return total_time_ms / generated_tokens
        return 0

    def compute_perplexity(self, generated_text):
        inputs = self.tokenizer(generated_text, return_tensors="pt", truncation=True)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
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

    def extract_response(self, prompt, response):
        """Use regex to extract the agent's response from a full prompt-generated text that includes system calls."""
        starting_prompt = f"###\nUser: {re.escape(prompt)}\nAssistant: Here's the story:"
        pattern = rf"{starting_prompt}(.*)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
