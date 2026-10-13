
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import CHECKPOINT_DIR, RESULTS_DIR, BATCH_SIZE
from src.models.cnn import EmotionCNN
from src.models.resnet import EmotionResNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet"],
        default="cnn",
        help="Model checkpoint to evaluate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint is missing class_names. Retrain the model.")

    model_name = checkpoint.get("model_name", args.model)

    if model_name == "cnn":
        model = EmotionCNN()
    elif model_name == "resnet":
        model = EmotionResNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_dataset = datasets.ImageFolder(
        Path("dataset/raw/test"),
        transform=test_transform,
    )

    if test_dataset.classes != class_names:
        raise ValueError(
            f"Class mismatch: checkpoint={class_names}, "
            f"dataset={test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    predictions, actual = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            predictions.extend(logits.argmax(1).cpu().tolist())
            actual.extend(labels.tolist())

    accuracy = np.mean(np.array(predictions) == np.array(actual))

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Test accuracy: {accuracy:.4%}")

    report = classification_report(
        actual,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / f"{model_name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix = confusion_matrix(
        actual,
        predictions,
        labels=list(range(len(class_names))),
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"{model_name.upper()} - Test Confusion Matrix")
    fig.tight_layout()

    matrix_path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nReport saved: {report_path}")
    print(f"Confusion matrix saved: {matrix_path}")


if __name__ == "__main__":
    main()
