# Speech Denoising Training Pipeline

This repository contains the preprocessing, training, evaluation, and reconstruction pipeline for a speech denoising system based on a compact convolutional U-Net. The model receives noisy speech spectrograms and learns to estimate a spectral mask that suppresses noise while preserving speech components.

Audio files, processed features, checkpoints, reports, and interface files are not tracked in Git.

## Structure

```text
.
|-- data/
|   |-- raw/          # extracted WAV files, not tracked
|   `-- processed/    # processed NPZ features and manifest, not tracked
|-- src/
|   |-- preprocess.py
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- evaluate.py
|   |-- reconstruct.py
|   `-- gui.py
|-- checkpoints/      # generated model checkpoints, not tracked
|-- outputs/          # generated metrics and audio, not tracked
|-- requirements.txt
`-- README.md
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

On Windows, the setup can also be run with:

```bat
scripts\setup_windows.bat
```

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

## Tests

Run the lightweight test suite:

```bash
python -m unittest discover -s tests
```

On Windows, the same check can be run with:

```bat
scripts\run_tests.bat
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

## Evaluation

Evaluate a trained checkpoint on the validation split:

```bash
python src/evaluate.py --checkpoint-path checkpoints/best_model.pt --output-csv outputs/evaluation.csv
```

The evaluation script reports mask MSE, input SNR, output SNR, SNR improvement, SDR, STOI, and PESQ when the optional metric packages are installed and compatible with the audio sample rate.

## Audio Reconstruction

Reconstruct denoised WAV files from a trained checkpoint:

```bash
python src/reconstruct.py --checkpoint-path checkpoints/best_model.pt --output-dir outputs/reconstructed
```

To save the noisy reference files next to the reconstructed outputs, add:

```bash
--save-noisy-reference
```

## Experiment Notes

A compact summary of the main training runs is available in:

```text
docs/experiments.md
```

## Graphical Interface

The PyQt5 interface can load a noisy WAV file, apply a trained checkpoint, play the noisy and denoised signals, display waveforms and spectrograms, and compare the result against an optional clean reference.

```bash
python src/gui.py
```

More details are available in:

```text
docs/gui.md
```

For presentation or second-computer setup notes, see:

```text
docs/demo.md
```

## Notes

The repository intentionally excludes generated data, WAV files, model weights, and experiment reports. These artifacts can be added in later releases or stored outside Git when needed.
