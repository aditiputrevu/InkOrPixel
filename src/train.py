import os
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import get_dataloaders
from .model import get_model
from .config import NUM_EPOCHS, LEARNING_RATE, MODEL_DIR, DEVICE, WEIGHT_DECAY, PATIENCE


def train_model():
    device = DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = get_dataloaders()
    print("Classes:", classes)

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print("Saved best model.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")