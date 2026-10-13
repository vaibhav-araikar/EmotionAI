"""FER-2013 CSV dataset loader."""
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LABEL_TO_ID = {name.lower(): i for i, name in enumerate(EMOTION_LABELS)}

class FER2013Dataset(Dataset):
    """Loads the common FER-2013 CSV format (emotion, pixels, Usage)."""
    def __init__(self, csv_path, split="Training", transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"FER-2013 CSV not found: {self.csv_path}. "
                "Download fer2013.csv from Kaggle and put it in dataset/raw/."
            )
        self.df = pd.read_csv(self.csv_path)
        required = {"emotion", "pixels", "Usage"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Expected columns {sorted(required)}; found {list(self.df.columns)}")
        self.df = self.df[self.df["Usage"].str.lower() == split.lower()].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split {split!r}. Check the CSV Usage column.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        pixels = np.fromstring(str(row["pixels"]), sep=" ", dtype=np.uint8)
        if pixels.size != 48 * 48:
            raise ValueError(f"Row {index} has {pixels.size} pixels; expected 2304.")
        image = Image.fromarray(pixels.reshape(48, 48), mode="L")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).unsqueeze(0) / 255.0
        label = int(row["emotion"])
        return image, label
