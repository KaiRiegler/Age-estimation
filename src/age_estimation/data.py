"""Download UTKFace, read the age labels from the file names, build DataLoaders."""

import random
import tarfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from age_estimation.config import DATA_DIR, DATASET_URL, IMAGE_SIZE

# PIL image -> float tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE) with values in [0, 1]
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def download_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract UTKFace into data_dir (skipped if already there)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir = data_dir / "UTKFace"
    if image_dir.is_dir():
        print("Data set already extracted.")
        return image_dir

    archive = data_dir / "UTKFace.tar.gz"
    if not archive.exists():
        print(f"Downloading {DATASET_URL} (about 107 MB) ...")
        urllib.request.urlretrieve(DATASET_URL, archive)

    print("Extracting data set ...")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    return image_dir


def load_file_names(image_dir: Path = DATA_DIR / "UTKFace") -> tuple[list[str], list[int]]:
    """Return image paths and ages. UTKFace names look like `<age>_<gender>_<race>_<date>.jpg`."""
    files, labels = [], []
    for path in sorted(Path(image_dir).glob("*.jpg")):
        age = path.name.split("_")[0]
        if age.isdigit():
            files.append(str(path))
            labels.append(int(age))
    if not files:
        raise FileNotFoundError(
            f"No images found in {image_dir}. Run `uv run age-download` first."
        )
    return files, labels


def split_data(files, labels, val_fraction=0.1, test_fraction=0.1, seed=42):
    """Shuffle once with a fixed seed, then split into train / validation / test."""
    pairs = list(zip(files, labels))
    random.Random(seed).shuffle(pairs)
    n_val = int(len(pairs) * val_fraction)
    n_test = int(len(pairs) * test_fraction)
    n_train = len(pairs) - n_val - n_test
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }


def load_image(path) -> torch.Tensor:
    """Read an image file and convert it to the model's input format."""
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


class FaceDataset(Dataset):
    """Yields (image, age) with image of shape (3, H, W) and age of shape (1,)."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, age = self.pairs[index]
        return load_image(path), torch.tensor([age], dtype=torch.float32)


def build_loader(pairs, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """Build a batched DataLoader from (filename, age) pairs."""
    return DataLoader(
        FaceDataset(pairs),
        batch_size=batch_size,
        shuffle=shuffle,  # reshuffled every epoch
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    image_dir = download_dataset()
    files, _ = load_file_names(image_dir)
    print(f"{len(files)} images in {image_dir}")


if __name__ == "__main__":
    main()
