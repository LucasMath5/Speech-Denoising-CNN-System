# Speech Denoising Training Pipeline

This repository contains the preprocessing and training pipeline for a speech denoising system based on a compact convolutional U-Net. The model receives noisy speech spectrograms and learns to estimate a spectral mask that suppresses noise while preserving speech components.

The first version of the repository focuses on dataset preparation, PyTorch data loading, model definition, and training. Audio files, processed features, checkpoints, reports, and interface files are not tracked in Git.

## Structure

```text
.
├── data/
│   ├── raw/          # extracted WAV files, not tracked
│   └── processed/    # processed NPZ features and manifest, not tracked
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── checkpoints/      # generated model checkpoints, not tracked
├── requirements.txt
└── README.md
```

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the local CUDA driver according to the official PyTorch installation instructions.

## Dataset Preparation

Place the NOIZEUS zip files in a local folder, then run:

```bash
python src/preprocess.py --zip-root path\to\zipped --project-root .
```

The preprocessing step extracts WAV files, pairs noisy and clean utterances by filename, computes STFT features, estimates the ideal ratio mask, and writes a manifest to:

```text
data/processed/manifest.csv
```

Default STFT parameters:

```text
n_fft = 512
hop_length = 128
win_length = 512
window = hann
```

## Training

Run the baseline training configuration:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2
```

Run the deeper configuration used in the main experiment:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2 --window-frames 64 --unet-depth 3 --bottleneck-dropout 0.1 --loss-mode mask_mse_mag_l1 --mask-loss-weight 0.5 --magnitude-loss-weight 0.5
```

Checkpoints and training history are written to:

```text
checkpoints/
```

## Notes

The repository intentionally excludes generated data, WAV files, model weights, experiment reports, and graphical interface code. These artifacts can be added in later releases or stored outside Git when needed.
