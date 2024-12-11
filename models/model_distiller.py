# model_distiller.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM

# Reference: https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html

class ModelDistiller:
    
    def __init__(self, teacher, student):
        """
        Initialize the ModelDistiller with teacher and student models.

        Args:
            teacher (nn.Module): Pretrained teacher model.
            student (nn.Module): Student model to be trained.
            device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        """
        if not teacher or not student:
            raise ValueError("Both teacher and student models must be provided.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                # Forward pass with the student model
                student_logits = student(inputs)

                #Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = (soft_targets * (soft_targets.log() - soft_prob)).sum(dim=-1).mean() * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    def save_model(self, save_path="./results/distilled_model"):
        """
        Saves the distilled student model to a directory specified by `save_path`

        Args:
            save_path (str): The directory where the model will be saved. Defaults to "./results/distilled_model"
        """
        if hasattr(self.student_model, "save_pretrained"):
            self.student_model.save_pretrained(save_path)
            print(f"Distilled model (Hugging Face) saved to '{save_path}'.")
        else:
            torch.save(self.student_model.state_dict(), f"{save_path}.pth")
            print(f"Distilled model (PyTorch) saved to '{save_path}.pth'.")


############Instruction for use################################################
# In your main script, create an instance of ModelDistiller
# model_distiller = ModelDistiller(teacher_model, student_model)

# Train the distilled model
# model_distiller.train_knowledge_distillation(train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight)

# Save the distilled model
# model_distiller.save_model(save_path)
