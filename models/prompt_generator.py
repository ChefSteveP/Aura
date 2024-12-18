from datasets import Dataset


class PromptGenerator:
    def __init__(self):
        pass

    def generate_eval_prompt(self, start_prompt, prompt):
        return f"""{start_prompt}
User: {prompt}
Assistant: Here's the story:
"""

    def generate_prompts_list(self, start_prompt, prompts):
        """Generate list of prompts based on provided prompts and generate_eval_prompt()"""
        return [self.generate_eval_prompt(start_prompt, prompt) for prompt in prompts]

    def generate_prompt_ids(self, prompts):
        return list(range(1, len(prompts) + 1))

    def generate_data_dict(self, story_prompt, prompts):
        return {
            "id": self.generate_prompt_ids(prompts),
            "prompt": self.generate_prompts_list(story_prompt, prompts),
            "query": prompts,
        }

    def generate_eval_dataset(self, start_prompt, prompts):
        return Dataset.from_dict(self.generate_data_dict(start_prompt, prompts))
