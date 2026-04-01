import os
import torch
from .dataset import get_dataloaders
from .model import get_model
from .config import MODEL_DIR

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, classes = get_dataloaders()

    model = get_model().to(device)
    model_path = os.path.join(MODEL_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("Classes:", classes)