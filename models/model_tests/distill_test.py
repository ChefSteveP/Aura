import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM
from model_distiller import ModelDistiller

# Step 1: Define synthetic data
def create_dummy_dataset(num_samples=100, input_dim=10, num_classes=2):
    """
    Creates a synthetic dataset for testing.
    """
    # Generate random inputs and labels
    inputs = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return dataset

# Step 2: Define simple models
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, num_classes=2):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Step 3: Test ModelDistiller
def test_model_distiller():
    # Create a synthetic dataset
    input_dim = 10
    num_classes = 2
    dataset = create_dummy_dataset(num_samples=100, input_dim=input_dim, num_classes=num_classes)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Define teacher and student models
    teacher_model = SimpleModel(input_dim=input_dim, num_classes=num_classes)
    student_model = SimpleModel(input_dim=input_dim, num_classes=num_classes)

    # Move teacher to evaluation mode and pre-train it (dummy initialization here)
    teacher_model.eval()

    # Initialize ModelDistiller
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    distiller = ModelDistiller(teacher=teacher_model, student=student_model, device=device)

    # Train using knowledge distillation
    distiller.train_knowledge_distillation(
        train_loader=train_loader,
        epochs=2,
        learning_rate=0.01,
        T=2.0,
        soft_target_loss_weight=0.7,
        ce_loss_weight=0.3
    )

    # Save the student model
    distiller.save_model("./distilled_student_model")

    print("Testing complete. Distilled student model saved.")

# Run the test
if __name__ == "__main__":
    test_model_distiller()
