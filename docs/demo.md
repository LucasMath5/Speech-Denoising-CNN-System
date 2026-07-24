# Demo Guide

This guide describes how to run the project during a presentation or on a second Windows computer.

The repository does not include model checkpoints, processed features, or WAV files. Copy those files separately when needed.

## Recommended Files for a Live Demo

For a GUI-only presentation, prepare:

```text
checkpoints/best_model.pt
demo_audio/noisy.wav
demo_audio/clean.wav
```

The clean file is optional. It is only required for comparison metrics.

## Setup on Windows

From the repository root, run:

```bat
scripts\setup_windows.bat
```

Then check the environment:

```bat
scripts\check_environment.bat
```

If CUDA is available, the check script prints the detected GPU. The project can also run on CPU, although inference and training will be slower.

## Running the Interface

Start the GUI:

```bat
scripts\run_gui.bat
```

In the interface:

1. Confirm the checkpoint path, usually `checkpoints/best_model.pt`.
2. Load a noisy WAV file.
3. Select `Process`.
4. Use the waveform, spectrogram, and mask tabs to inspect the result.
5. Optionally load a clean reference and select `Compare with clean`.

## Before a Presentation

Run this short checklist before presenting:

```text
[ ] virtual environment created
[ ] dependencies installed
[ ] checkpoint copied to checkpoints/best_model.pt
[ ] noisy demo audio available
[ ] optional clean reference available
[ ] GUI opens
[ ] denoising finishes on the selected audio
[ ] audio playback works
```

## Notes

Use short WAV files for a smoother live demonstration. The GUI processes long files in chunks, but short examples make plotting, PESQ, and playback more responsive.

Generated audio, temporary playback files, logs, CSV files, and checkpoints remain outside Git through `.gitignore`.

