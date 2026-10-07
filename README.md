# Age Estimation with a CNN

A small convolutional neural network that estimates a person's age from a face image.
Trained on the [UTKFace](https://susanqq.github.io/UTKFace/) data set (about 23,700 face
images, ages 1 to 116) with PyTorch and a hand-written training loop.

## Project structure

```
age-estimation-cnn/
├── src/age_estimation/
│   ├── config.py      paths and hyperparameters
│   ├── data.py        download, labels, Dataset and DataLoader
│   ├── model.py       CNN architecture
│   ├── train.py       training loop and evaluation
│   └── predict.py     prediction for a single image
├── notebooks/
│   └── age_regression_cnn.ipynb   walkthrough with plots
├── tests/
├── pyproject.toml     dependencies
└── data/, models/, reports/       created locally, not in Git
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```
git clone <repo-url>
cd age-estimation-cnn
uv sync
```

`uv sync` installs PyTorch with CUDA support on Windows and Linux (configured in
`pyproject.toml`). Check that the GPU is found:

```
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Without an NVIDIA GPU everything still runs, just on the CPU.

## Usage

```
uv run age-download                  # download and extract UTKFace into data/ (107 MB)
uv run age-train --epochs 20         # train, saves weights to models/
uv run age-train --num-workers 0     # if parallel data loading causes trouble
uv run age-predict path/to/face.jpg  # predict the age for one image
uv run jupyter lab                   # open the notebook
uv run pytest                        # run the tests
```

## Model

| Layer | Output |
|---|---|
| Conv2d, 8 filters, 5x5, stride 4, ReLU | 8 x 49 x 49 |
| Conv2d, 16 filters, 5x5, stride 4, ReLU | 16 x 12 x 12 |
| Flatten + Dropout 0.2 | 2304 |
| Linear 128, ReLU | 128 |
| Linear 64, ReLU | 64 |
| Linear 1 | 1 (age in years) |

Input: 200 x 200 RGB image scaled to [0, 1]. Loss: mean absolute error (MAE),
optimizer: RMSprop. Data split: 80 % training, 10 % validation, 10 % test.

## Results

_To be filled in after training: test MAE and the loss curve from the notebook._

## Data

The data set is not part of this repository. UTKFace is available for non-commercial
research purposes only.
