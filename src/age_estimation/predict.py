"""Predict the age for a single image: `uv run age-predict path/to/face.jpg`."""

import argparse
from pathlib import Path

import torch

from age_estimation.config import DEFAULT_WEIGHTS
from age_estimation.data import load_image
from age_estimation.model import AgeCNN, get_device


def load_trained_model(weights: Path = DEFAULT_WEIGHTS, device=None) -> AgeCNN:
    if not Path(weights).exists():
        raise FileNotFoundError(f"{weights} not found. Run `uv run age-train` first.")
    device = device or get_device()
    model = AgeCNN()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device)


@torch.no_grad()
def predict_age(model, image_path) -> float:
    """Works best with a square portrait with the face in the centre."""
    model.eval()
    device = next(model.parameters()).device
    image = load_image(image_path).unsqueeze(0).to(device)  # add the batch dimension
    return model(image).item()


def main():
    parser = argparse.ArgumentParser(description="Predict the age of the person in an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    age = predict_age(load_trained_model(args.weights), args.image)
    print(f"Predicted age: {age:.1f}")


if __name__ == "__main__":
    main()
