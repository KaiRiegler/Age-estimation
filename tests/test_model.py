import torch

from age_estimation.config import IMAGE_SIZE
from age_estimation.model import AgeCNN


def test_model_outputs_one_value_per_image():
    model = AgeCNN()
    out = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, 1)
