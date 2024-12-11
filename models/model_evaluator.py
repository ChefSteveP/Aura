import time
from collections import Counter
import torch
import textstat
from transformers import pipeline


class ModelEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model.to("cuda")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Storytelling prompts for evaluation
        self.prompts = [
            "Imagine you're a wizard who has been banished from your kingdom for practicing forbidden magic. How would you write a story about your journey to reclaim your honor and the challenges you face?",
            "You're the captain of a spaceship exploring a distant galaxy. One day, you encounter an abandoned alien station emitting a strange signal. How would you write a story about what happens next?",
            "A detective is called to investigate a murder at a remote mansion during a thunderstorm. How would you write a story from the detective's perspective as they unravel the mystery?",
            "You're a merchant traveling along the Silk Road in the 14th century. How would you write a story about an unusual event or encounter during your journey?",
            "The world has been devastated by a zombie plague leaving only a handful of survivors. How would you write a story about how you navigate this new world?",
        ]

    def evaluate(self):
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            truncation=True,
        )

        results = []
        for i, prompt in enumerate(self.prompts, start=1):
            print(f"---------- PROMPT {i} ----------")
            print(f"\nEvaluating prompt: {prompt}\n")

            # Time to first token (TTFT)
            ttft_start = time.time()
            generator(prompt, max_new_tokens=1)
            ttft = time.time() - ttft_start

            # Generate output
            token_start_time = time.time()
            output = generator(
                prompt,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            total_time = time.time() - token_start_time

            # Calculate number of tokens generated
            generated_text = output[0]["generated_text"]
            generated_tokens = len(self.tokenizer.tokenize(generated_text))
            
            # Calculate avg time per token
            if generated_tokens > 0:
                avg_time_per_token = total_time / generated_tokens
            else:
                avg_time_per_token = 0

            print(f"\nGenerated Text: {generated_text}\n")

            # Calculate other metrics
            perplexity = self.compute_perplexity(generated_text)
            length = self.compute_length(generated_text)
            repetition_rate = self.compute_repetition_rate(generated_text)
            distinct_2 = self.distinct_n(generated_text, n=2)
            readability = self.compute_readability_score(generated_text)

            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "perplexity": perplexity,
                "length": length,
                "repetition_rate": repetition_rate,
                "distinct_2": distinct_2,
                "readability": readability,
                "time_to_first_token": ttft,
                "avg_time_per_token": avg_time_per_token,
                "tokens_generated": generated_tokens,
            })

            print(f"Metrics:")
            print(f"Time to First Token (TTFT): {ttft:.4f} seconds")
            print(f"Average Time per Token: {avg_time_per_token:.4f} seconds")
            print(f"Tokens Generated: {generated_tokens}")
            print(f"Perplexity: {perplexity}")
            print(f"Length: {length} words")
            print(f"Repetition Rate: {repetition_rate}")
            print(f"Distinct-2: {distinct_2}")
            print(f"Readability (Flesch Reading Ease): {readability}\n")

        return results


    def compute_ttft(self, generator, prompt):
        """Time to first token"""
        start_time = time.time()
        generator(prompt, max_new_tokens=1)
        ttft = time.time() - start_time
        return ttft
    
    def compute_avg_time_per_token(self, generator, prompt, max_new_tokens=50):
        start_time = time.time()
        generator(prompt, max_new_tokens=max_new_tokens)
        total_time = time.time() - start_time
        avg_time_per_token = total_time / max_new_tokens
        return avg_time_per_token

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
        n_grams = set(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        return len(n_grams) / len(words) if len(words) > 0 else 0

    def compute_readability_score(self, generated_text):
        return textstat.flesch_reading_ease(generated_text)
