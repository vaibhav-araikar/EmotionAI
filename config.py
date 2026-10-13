"""Central configuration for the Facial Emotion Recognition project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_CSV = ROOT_DIR / "dataset" / "raw" / "fer2013.csv"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

IMAGE_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # Reliable default on Windows; increase after confirming stability.
SEED = 42
