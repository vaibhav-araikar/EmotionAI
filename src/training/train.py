
"""Train CNN or ResNet on a folder-based facial emotion dataset."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import CHECKPOINT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from src.models.cnn import EmotionCNN
from src.models.resnet import EmotionResNet


DATA_DIR = Path("dataset/raw")
IMAGE_SIZE = 48
NUM_WORKERS = 0  # Reliable on Windows; increase later if needed.


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(training=False):
    operations = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return transforms.Compose(operations)


def create_datasets():
    # Use identical alphabetical class mappings for every split.
    train_base = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=get_transforms(training=True),
    )

    val_base = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=get_transforms(training=False),
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=get_transforms(training=False),
    )

    if train_base.classes != test_dataset.classes:
        raise ValueError(
            f"Train/test classes differ: {train_base.classes} vs "
            f"{test_dataset.classes}"
        )

    labels = [label for _, label in train_base.samples]
    indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    return train_dataset, val_dataset, test_dataset, train_base.classes


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += batch_size

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
    )


def get_macro_f1(model, loader, device):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            predictions.extend(logits.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())

    return f1_score(actual, predictions, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, test_ds, class_names = create_datasets()

    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Test images: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = EmotionCNN() if args.model == "cnn" else EmotionResNet()
    model = model.to(device)

    # Compensate partially for class imbalance without extreme weights.
    train_targets = np.array([
        train_ds.dataset.targets[i] for i in train_ds.indices
    ])
    counts = np.bincount(train_targets, minlength=len(class_names))
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pth"

    best_f1 = -1.0
    best_state = None
    patience = 6
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        val_f1 = get_macro_f1(model, val_loader, device)
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model": best_state,
                    "model_name": args.model,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "class_names": class_names,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print("Early stopping.")
            break

    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Best model saved at: {checkpoint_path}")


if __name__ == "__main__":
    main()
