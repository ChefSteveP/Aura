# model_distiller.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import AutoModelForCausalLM

# Reference: https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html

class ModelDistiller:
    
    def __init__(self, model_name, device="cpu"):
        """
        Initialize the ModelDistiller with a pretrained model and other state variables.
        """
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
        """
        Train a student model using knowledge distillation from a teacher model.

        Args:
            teacher (nn.Module): The teacher model to distill from.
            student (nn.Module): The student model to train.
            train_loader (DataLoader): The training data loader.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate for the optimizer.
            T (float): The temperature for the knowledge distillation.
            soft_target_loss_weight (float): The weight for the soft targets loss.
            ce_loss_weight (float): The weight for the cross-entropy loss.
            device (torch.device): The device to run the training on.

        Returns:
            None
        """
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        teacher.eval()  # Teacher set to evaluation mode
        student.train() # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

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
                soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    def save_model(self):
        self.model.save_pretrained("./llama_literature_distilled")
        print("Knowledge distillation complete. Distilled model saved to './llama_literature_pruned'.")


############Instruction for use################################################