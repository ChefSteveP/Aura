# model_distiller.py
import random
from tqdm import tqdm
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import pipeline
from transformers import AutoModelForCausalLM
from model_utils import ModelUtils

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
            end_idx = len(input_ids)
        else:
            start_idx = random.randint(0, len(input_ids) - self.segment_length)
            end_idx = start_idx + self.segment_length
        
        # Slice out the random segment setting aside the last token for the prediction label
        input_ids_segment = input_ids[start_idx:end_idx-1]
        attention_mask_segment = attention_mask[start_idx:end_idx-1]
        label = input_ids[end_idx]
        
        # Convert to torch tensors
        input_ids_tensor = torch.tensor(input_ids_segment, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask_segment, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "id": example["id"],
            "text": example["text"],                 # full reference text
            "input_ids": input_ids_tensor,           # segment token
            "attention_mask": attention_mask_tensor, # segment mask
            "labels": label_tensor                   # Next token in sequence
        }
        
class ModelDistiller:
    
    def __init__(self, teacher, student, device="cpu"):
        """
        Initialize the ModelDistiller with teacher and student models.

        Args:
            teacher (nn.Module): Pretrained teacher model.
            student (nn.Module): Student model to be trained.
        """
        if not teacher or not student:
            raise ValueError("Both teacher and student models must be provided.")
        
        self.device = device 
        self.teacher_model = teacher.to(self.device)
        self.student_model = student.to(self.device)
        
    def train_knowledge_distillation(self, train_loader, epochs, 
                                     learning_rate, T, soft_target_loss_weight, 
                                     ce_loss_weight, teacher=None, student=None):
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
    
        teacher = teacher or self.teacher_model
        student = student or self.student_model
        
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        teacher.eval()  # Teacher set to evaluation mode
        student.train() # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            
            for batch in progress_bar:
                
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)           # size: (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(self.device) # size: (batch_size, seq_len)
                labels = batch["labels"].to(self.device)                 # size: (batch_size,) single token

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits #extract logits

                # Forward pass with the student model
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits  #extract logits

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = (soft_targets * (soft_targets.log() - soft_prob)).sum(dim=-1).mean() * (T**2)

                # Calculate the last token label loss
                last_token_logits = student_logits[:, -1, :]  # shape: (batch_size, vocab_size) 
                label_loss = ce_loss(last_token_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    def save_model(self, save_path="/home/shared_storage/models/llama_1B_distilled.pt"):
        """
        Saves the distilled student model to a directory specified by `save_path`

        Args:
            save_path (str): The directory where the model will be saved. Defaults to "./results/distilled_model"
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.student_model, save_path)


############Instruction for use################################################
# In your main script, create an instance of ModelDistiller
# model_distiller = ModelDistiller(teacher_model, student_model)

# Train the distilled model
# model_distiller.train_knowledge_distillation(train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight)

# Save the distilled model
# model_distiller.save_model(save_path)
