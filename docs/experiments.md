# Experiment Summary

This document summarizes the main training runs for the speech denoising U-Net. The results were obtained on the same processed NOIZEUS split, using an utterance-level validation split with `validation_fraction=0.15`.

Generated checkpoints, CSV files, audio files, and processed features are intentionally not tracked in this repository.

## Common Setup

All runs used the same preprocessing pipeline:

```text
n_fft = 512
hop_length = 128
win_length = 512
window = hann
input = log1p(|STFT_noisy|)
target = ideal ratio mask
```

The default target mask is computed as:

```text
IRM = |STFT_clean| / (|STFT_noisy| + epsilon)
```

and clipped to `[0, 1]`, except in the run that explicitly tests a larger mask range.

The baseline training setup used Adam with learning rate `1e-3`, `ReduceLROnPlateau`, early stopping with patience `10`, batch size `16`, and automatic mixed precision when CUDA was available.

## Results

| Experiment | Configuration | Epochs | Best epoch | Time (min) | Mask MSE | Output SNR/SDR (dB) | SNR gain (dB) | STOI | PESQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp00_baseline` | 32 frames, base16, mask MSE | 65 | 55 | 14.49 | 0.0699 | 12.4386 | +5.6430 | 0.8703 | 2.3163 |
| `exp01_context64_combined` | 64 frames, base16, mask MSE + magnitude L1 | 35 | 25 | 10.76 | 0.0716 | 12.3924 | +5.5968 | 0.8667 | N/A |
| `exp02_context64_wide24_combined` | 64 frames, base24, mask MSE + magnitude L1 | 39 | 29 | 13.85 | 0.0705 | 12.5568 | +5.7612 | 0.8745 | 2.2805 |
| `exp03_context64_mask15` | 64 frames, base16, mask clipped to 1.5 | 42 | 32 | 11.00 | 0.0788 | 12.0575 | +5.2618 | 0.8686 | N/A |
| `exp04_deep3_context64_combined` | 64 frames, depth3, base16, dropout 0.1, mask MSE + magnitude L1 | 32 | 22 | 9.46 | 0.0693 | 12.7825 | +5.9869 | 0.8784 | 2.3039 |

The average input SNR for the evaluated split was `6.7956 dB` in all runs.

## Main Findings

The baseline was already strong for a compact U-Net, reaching `12.4386 dB` output SNR/SDR and `0.8703` STOI.

Increasing the context window to 64 frames and adding a magnitude reconstruction term did not improve the model by itself. The `exp01_context64_combined` run obtained slightly lower output SNR/SDR and STOI than the baseline.

Increasing the model capacity improved the combined-loss setup. The wider `base_channels=24` model improved output SNR/SDR and STOI, but had lower PESQ than the baseline.

Allowing the mask to amplify components up to `1.5` did not help in this setup. The `exp03_context64_mask15` run had the lowest output SNR/SDR and the highest mask MSE among the listed runs.

The best model by output SNR/SDR and STOI was `exp04_deep3_context64_combined`. It used a deeper U-Net with three encoder-decoder levels, 64-frame windows, bottleneck dropout `0.1`, and the combined mask/magnitude loss.

## Best Model by Metric

| Criterion | Best run | Value |
|---|---|---:|
| Output SNR/SDR | `exp04_deep3_context64_combined` | 12.7825 dB |
| SNR gain | `exp04_deep3_context64_combined` | +5.9869 dB |
| STOI | `exp04_deep3_context64_combined` | 0.8784 |
| PESQ | `exp00_baseline` | 2.3163 |

The deeper model improved SNR/SDR and STOI, while the baseline remained slightly better in PESQ. This indicates that objective distortion reduction and perceptual quality do not always rank models in the same order.

## Reproduction Commands

Baseline:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2
python src/evaluate.py --checkpoint-path checkpoints/best_model.pt --output-csv outputs/evaluation.csv
```

Combined loss with 64-frame context:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2 --window-frames 64 --loss-mode mask_mse_mag_l1 --mask-loss-weight 0.5 --magnitude-loss-weight 0.5 --checkpoint-dir checkpoints/exp01_context64_combined
python src/evaluate.py --checkpoint-path checkpoints/exp01_context64_combined/best_model.pt --output-csv outputs/exp01_evaluation.csv
```

Wider U-Net:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2 --window-frames 64 --loss-mode mask_mse_mag_l1 --mask-loss-weight 0.5 --magnitude-loss-weight 0.5 --base-channels 24 --checkpoint-dir checkpoints/exp02_context64_wide24_combined
python src/evaluate.py --checkpoint-path checkpoints/exp02_context64_wide24_combined/best_model.pt --output-csv outputs/exp02_evaluation.csv
```

Mask range up to 1.5:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2 --window-frames 64 --target-mode irm --target-clip-max 1.5 --output-scale 1.5 --checkpoint-dir checkpoints/exp03_context64_mask15
python src/evaluate.py --checkpoint-path checkpoints/exp03_context64_mask15/best_model.pt --output-csv outputs/exp03_evaluation.csv
```

Deeper U-Net:

```bash
python src/train.py --epochs 100 --batch-size 16 --num-workers 2 --window-frames 64 --loss-mode mask_mse_mag_l1 --mask-loss-weight 0.5 --magnitude-loss-weight 0.5 --unet-depth 3 --bottleneck-dropout 0.1 --checkpoint-dir checkpoints/exp04_deep3_context64_combined
python src/evaluate.py --checkpoint-path checkpoints/exp04_deep3_context64_combined/best_model.pt --output-csv outputs/exp04_evaluation.csv
```

## Notes on Metrics

`Mask MSE` measures how close the predicted mask is to the target mask. Lower values are better, but this metric does not always match perceptual quality or reconstructed waveform quality.

`Output SNR/SDR` compares the clean reference against the reconstruction error in the time domain. Higher values indicate lower distortion after reconstruction.

`STOI` estimates speech intelligibility and ranges from 0 to 1. Higher values indicate better intelligibility.

`PESQ` estimates perceptual speech quality against a clean reference. Higher values indicate better perceived quality, but availability depends on the optional `pesq` package and sample-rate compatibility.
