"""CNN architecture."""

import torch
from torch import nn


class AgeCNN(nn.Module):
    """Two strided conv layers followed by three dense layers; outputs one number (the age).

    Input:  (batch, 3, 200, 200)
    conv0:  (batch, 8, 49, 49)
    conv1:  (batch, 16, 12, 12)  -> flattened to 16 * 12 * 12 = 2304
    Output: (batch, 1)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=4),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(16 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))


def get_device() -> torch.device:
    """GPU if PyTorch can see one, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
