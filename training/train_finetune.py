from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from dataloader import get_dataloaders


MODEL_DIR = Path("models")
BASE_MODEL_PATH = MODEL_DIR / "best_mobilenetv2.pth"
FINETUNED_MODEL_PATH = MODEL_DIR / "best_mobilenetv2_finetuned.pth"

NUM_EPOCHS = 8
LEARNING_RATE = 0.0001


def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model


def compute_class_weights(train_dataset, num_classes, device):
    labels = [label for _, label in train_dataset.samples]
    class_counts = Counter(labels)

    total_samples = sum(class_counts.values())

    weights = []
    for class_idx in range(num_classes):
        weight = total_samples / (num_classes * class_counts[class_idx])
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _, classes = get_dataloaders()
    num_classes = len(classes)

    checkpoint = torch.load(BASE_MODEL_PATH, map_location=device)

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze all feature layers first
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last few feature blocks for fine-tuning
    for layer in model.features[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)

    class_weights = compute_class_weights(
        train_loader.dataset,
        num_classes,
        device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="Fine-tuning"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(val_acc)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "val_acc": best_val_acc
                },
                FINETUNED_MODEL_PATH
            )

            print("Best fine-tuned model saved!")

    print(f"\nFine-tuning completed. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train_finetune()