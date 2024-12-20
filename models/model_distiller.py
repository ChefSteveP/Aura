# model_distiller.py
import random
from tqdm import tqdm
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset


# Reference: https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
class RandSnipitDataset(Dataset):
    def __init__(self, data, segment_length=512):
        """
        Args:
            data (list of dict): Each element is expected to have keys:
                "id" (str or int): Unique identifier for the entry
                "text" (str): The original text
                "input_ids" (list[int]): Tokenized input IDs
                "attention_mask" (list[int]): Attention mask corresponding to input_ids
        """
        self.data = data
        self.segment_length = segment_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]

        # If the text is shorter than the desired segment length, take it all.
        if len(input_ids) <= self.segment_length:
            start_idx = 0
            end_idx = len(input_ids) - 1
        else:
            start_idx = random.randint(0, len(input_ids) - self.segment_length - 1)
            end_idx = start_idx + self.segment_length

        # Slice out the random segment setting aside the last token for the prediction label
        input_ids_segment = input_ids[start_idx : end_idx - 1]
        attention_mask_segment = attention_mask[start_idx : end_idx - 1]
        label = input_ids[end_idx]

        # Convert to torch tensors
        input_ids_tensor = torch.tensor(input_ids_segment, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask_segment, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "id": example["id"],
            "text": example["text"],  # full reference text
            "input_ids": input_ids_tensor,  # segment token
            "attention_mask": attention_mask_tensor,  # segment mask
            "labels": label_tensor,  # Next token in sequence
        }


class ModelDistiller:
    def __init__(self):
        pass

    def train_knowledge_distillation(
        self, train_loader, epochs, learning_rate, T, alpha, teacher, student
    ):
        """
        Trains the student model using knowledge distillation from the teacher model.

        Args:
            train_loader (DataLoader): Training data loader.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            T (float): Temperature for soft target scaling.
            soft_target_loss_weight (float): Weight for the soft target loss.
            ce_loss_weight (float): Weight for the cross-entropy loss.
            teacher (nn.Module, optional): Teacher model (defaults to self.teacher_model).
            student (nn.Module, optional): Student model (defaults to self.student_model).
        """
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        optimizer = optim.AdamW(student.parameters(), lr=learning_rate)

        teacher.eval()  # Teacher set to evaluation mode
        student.train()  # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

            for batch in progress_bar:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Forward pass with the teacher model
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits

                # Forward pass with the student model
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

                batch_size, seq_length, vocab_size = student_logits.size()
                student_logits_flat = student_logits.view(-1, vocab_size)
                teacher_logits_flat = teacher_logits.view(-1, vocab_size)
                labels_flat = labels.view(-1)

                # Compute soft targets with temperature scaling
                teacher_probs = nn.functional.softmax(teacher_logits_flat / T, dim=-1)
                student_log_probs = nn.functional.log_softmax(student_logits_flat / T, dim=-1)

                # Compute KL Divergence loss
                kd_loss = kl_loss_fn(student_log_probs, teacher_probs) * (T**2)

                # Compute Cross-Entropy loss with hard labels
                ce_loss = ce_loss_fn(student_logits_flat, labels_flat)

                # Combine both losses and weight with alpha
                loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        return student, teacher


############Instruction for use################################################
# In your main script, create an instance of ModelDistiller
# model_distiller = ModelDistiller(teacher_model, student_model)

# Train the distilled model
# model_distiller.train_knowledge_distillation(train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight)

# Save the distilled model
# model_distiller.save_model(save_path)
