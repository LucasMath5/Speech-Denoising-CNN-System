"""Reconstruct denoised wav files from a trained U-Net checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.amp import autocast
from tqdm import tqdm

try:
    from src.dataset import read_manifest, split_items
    from src.model import UNetConfig, build_model
except ModuleNotFoundError:
    from dataset import read_manifest, split_items
    from model import UNetConfig, build_model


@dataclass(frozen=True)
class ReconstructConfig:
    """Configuration for denoised audio reconstruction."""

    manifest_path: Path = Path("data/processed/manifest.csv")
    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    output_dir: Path = Path("outputs/reconstructed")
    validation_fraction: float = 0.15
    seed: int = 42
    split: str = "validation"
    max_items: int | None = None
    use_amp: bool = True
    save_noisy_reference: bool = False
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    window: str = "hann"


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a denoising U-Net from a training checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model_config = UNetConfig(**checkpoint.get("model_config", {}))
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_processed_paths(config: ReconstructConfig) -> list[Path]:
    """Select processed files from the requested split."""

    items = read_manifest(config.manifest_path)
    if config.split != "all":
        train_items, validation_items = split_items(
            items,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
            strategy="utterance",
        )
        items = validation_items if config.split == "validation" else train_items
    if config.max_items is not None:
        items = items[: config.max_items]
    return [item.processed_path for item in items]


def predict_mask(
    model: torch.nn.Module,
    noisy_log_magnitude: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Predict a full-spectrogram denoising mask."""

    tensor = torch.from_numpy(noisy_log_magnitude[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        prediction = model(tensor)
    return prediction.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def inverse_stft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    config: ReconstructConfig,
    length: int | None,
) -> np.ndarray:
    """Convert magnitude and phase back to a waveform."""

    complex_stft = magnitude * np.exp(1j * phase)
    waveform = librosa.istft(
        complex_stft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
        length=length,
    )
    return np.asarray(waveform, dtype=np.float32)


def output_path_for(processed_path: Path, output_dir: Path, noise_type: str, snr: str, utterance_id: str) -> Path:
    """Build the denoised wav output path for one processed item."""

    _ = processed_path
    return output_dir / noise_type / snr / f"{utterance_id}_denoised.wav"


def reconstruct(config: ReconstructConfig) -> list[Path]:
    """Reconstruct denoised wav files and return their paths."""

    device = get_device()
    use_amp = config.use_amp and device.type == "cuda"
    model = load_checkpoint_model(config.checkpoint_path, device)
    processed_paths = select_processed_paths(config)
    written_paths: list[Path] = []

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Items: {len(processed_paths)}")

    for processed_path in tqdm(processed_paths, desc="reconstructing"):
        with np.load(processed_path) as data:
            noisy_log_magnitude = np.asarray(data["noisy_log_magnitude"], dtype=np.float32)
            noisy_magnitude = np.asarray(data["noisy_magnitude"], dtype=np.float32)
            noisy_phase = np.asarray(data["noisy_phase"], dtype=np.float32)
            sample_rate = int(data["sample_rate"])
            noisy_path = Path(str(data["noisy_path"]))
            noise_type = str(data["noise_type"])
            snr = str(data["snr"])
            utterance_id = str(data["utterance_id"])

        noisy_audio, _ = librosa.load(noisy_path, sr=sample_rate, mono=True)
        predicted_mask = predict_mask(model, noisy_log_magnitude, device, use_amp)
        denoised_magnitude = predicted_mask * noisy_magnitude
        denoised_audio = inverse_stft(
            denoised_magnitude,
            noisy_phase,
            config,
            length=len(noisy_audio),
        )

        output_path = output_path_for(processed_path, config.output_dir, noise_type, snr, utterance_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, denoised_audio, sample_rate)
        written_paths.append(output_path)

        if config.save_noisy_reference:
            noisy_output_path = output_path.with_name(f"{utterance_id}_noisy.wav")
            sf.write(noisy_output_path, noisy_audio, sample_rate)

    print(f"Wrote {len(written_paths)} wav files to {config.output_dir}")
    return written_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Reconstruct denoised wavs from a trained checkpoint.")
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reconstructed"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-noisy-reference", action="store_true")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--window", type=str, default="hann")
    return parser.parse_args()


def main() -> None:
    """Run the reconstruction command-line interface."""

    args = parse_args()
    config = ReconstructConfig(
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split=args.split,
        max_items=args.max_items,
        use_amp=not args.no_amp,
        save_noisy_reference=args.save_noisy_reference,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=args.window,
    )
    reconstruct(config)


if __name__ == "__main__":
    main()
