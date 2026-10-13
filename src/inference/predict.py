"""Predict facial-expression probabilities for one image file."""
import argparse
import torch
from PIL import Image
from src.data.dataset import EMOTION_LABELS
from src.data.transforms import build_transforms
from src.models.cnn import EmotionCNN
from src.models.resnet import EmotionResNet
from config import CHECKPOINT_DIR

def load_model(model_name, device):
    path = CHECKPOINT_DIR / f"{model_name}_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {path}. Train a model first.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = EmotionCNN() if model_name == "cnn" else EmotionResNet()
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    image = Image.open(args.image).convert("L").resize((48, 48))
    x = build_transforms(False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    for idx in probs.argsort(descending=True).tolist():
        print(f"{EMOTION_LABELS[idx]:>8}: {probs[idx].item()*100:5.2f}%")

if __name__ == "__main__":
    main()
