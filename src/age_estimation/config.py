"""Paths and hyperparameters in one place."""

from dataclasses import dataclass
from pathlib import Path

# Repo root = two levels above this file (src/age_estimation/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_URL = "https://archive.org/download/UTKFace/UTKFace.tar.gz"
DEFAULT_WEIGHTS = MODELS_DIR / "age_cnn.pt"

IMAGE_SIZE = 200  # images are resized to IMAGE_SIZE x IMAGE_SIZE


@dataclass
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 64
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 4  # processes that load images in parallel; 0 = load in the main process
