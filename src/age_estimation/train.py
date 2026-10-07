"""Training loop, evaluation and the `age-train` command."""

import argparse
import json

import torch
from torch import nn

from age_estimation.config import DEFAULT_WEIGHTS, REPORTS_DIR, TrainConfig
from age_estimation.data import build_loader, load_file_names, split_data
from age_estimation.model import AgeCNN, get_device


@torch.no_grad()
def predict_all(model, loader, device):
    """Return (predictions, true ages) for a whole DataLoader as 1-D tensors."""
    model.eval()  # switches dropout off
    preds, targets = [], []
    for images, ages in loader:
        preds.append(model(images.to(device)).cpu().flatten())
        targets.append(ages.flatten())
    return torch.cat(preds), torch.cat(targets)


def evaluate(model, loader, device) -> float:
    """Mean absolute error: how many years the prediction is off on average."""
    preds, targets = predict_all(model, loader, device)
    return (preds - targets).abs().mean().item()


def train(model, train_loader, val_loader, config: TrainConfig, device) -> dict:
    """Train the model and return the loss per epoch."""
    loss_fn = nn.L1Loss()  # MAE
    # alpha=0.9 matches the Keras RMSprop default used in the original notebook
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.9)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, config.epochs + 1):
        model.train()  # switches dropout on
        total, n = 0.0, 0
        for images, ages in train_loader:  # one full pass over the data = one epoch
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), ages)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
            n += len(images)
        train_loss = total / n
        val_loss = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:>2}/{config.epochs}  train MAE: {train_loss:.2f}  val MAE: {val_loss:.2f}")
    return history


def main():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the age regression CNN on UTKFace.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    args = parser.parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    torch.manual_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    files, labels = load_file_names()
    splits = split_data(files, labels, config.val_fraction, config.test_fraction, config.seed)
    print({name: len(pairs) for name, pairs in splits.items()})

    train_loader = build_loader(splits["train"], config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = build_loader(splits["val"], config.batch_size, num_workers=config.num_workers)
    test_loader = build_loader(splits["test"], config.batch_size, num_workers=config.num_workers)

    model = AgeCNN().to(device)
    history = train(model, train_loader, val_loader, config, device)
    history["test_loss"] = evaluate(model, test_loader, device)
    print(f"Test MAE: {history['test_loss']:.2f} years")

    DEFAULT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_WEIGHTS)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Weights saved to {DEFAULT_WEIGHTS}")


if __name__ == "__main__":
    main()
