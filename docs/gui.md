# Graphical Interface

The project includes a PyQt5 interface for interactive speech denoising with a trained U-Net checkpoint.

## Main Features

The interface supports:

- loading a noisy WAV file;
- selecting a trained PyTorch checkpoint;
- processing the noisy signal with the denoising model;
- saving the reconstructed denoised WAV file;
- playing the noisy, denoised, and optional clean reference signals;
- visualizing separate waveform plots;
- visualizing noisy, denoised, and clean spectrograms when a reference is available;
- displaying the predicted spectral mask;
- comparing noisy and denoised signals against a clean reference;
- exporting comparison metrics to CSV.

## Running

Install the project dependencies and start the interface:

```bash
python src/gui.py
```

By default, the interface looks for a checkpoint at:

```text
checkpoints/best_model.pt
```

The path can be changed directly in the checkpoint field or by using the checkpoint browser button.

## Inference Mode

Use this mode when only a noisy signal is available.

1. Select `Load noisy wav`.
2. Confirm the checkpoint path.
3. Select `Process`.
4. Inspect the `Waveforms`, `Spectrograms`, and `Mask` tabs.
5. Use `Save denoised wav` to export the reconstructed audio.

## Comparison Mode

Use this mode when a clean reference signal is available.

1. Select `Load noisy wav`.
2. Select `Load clean reference`.
3. Select `Compare with clean`.
4. Inspect the comparison metrics and plots.
5. Use `Export comparison CSV` to save the current metrics.

The comparison mode reports SNR/SDR, STOI, PESQ when available, and residual RMS before and after denoising.

## Technical Notes

The interface resamples input audio to `8000 Hz`, matching the NOIZEUS setup used in the training runs.

The STFT configuration is:

```text
n_fft = 512
hop_length = 128
win_length = 512
window = hann
```

The model input is `log1p(abs(STFT_noisy))`. The predicted mask is multiplied by the noisy magnitude spectrogram, and the waveform is reconstructed with the original noisy phase.

Long spectrograms are processed in overlapping time chunks to reduce GPU memory usage. Long plots are downsampled only for visualization; the denoising itself is still performed on the full signal.

PESQ is skipped automatically for audio longer than 30 seconds to avoid slow or unstable native metric calls.

Generated GUI files are written under:

```text
outputs/gui/
```

This directory is ignored by Git.
