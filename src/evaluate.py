"""Evaluate a trained denoising U-Net on processed NOIZEUS features."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

try:
    from pystoi import stoi as stoi_score
except ModuleNotFoundError:
    stoi_score = None

try:
    from pesq import pesq as pesq_score
except ModuleNotFoundError:
    pesq_score = None

try:
    from src.dataset import read_manifest, split_items
    from src.model import UNetConfig, build_model
except ModuleNotFoundError:
    from dataset import read_manifest, split_items
    from model import UNetConfig, build_model


@dataclass(frozen=True)
class EvaluateConfig:
    """Configuration for checkpoint evaluation."""

    manifest_path: Path = Path("data/processed/manifest.csv")
    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    output_csv: Path = Path("outputs/evaluation.csv")
    validation_fraction: float = 0.15
    seed: int = 42
    split: str = "validation"
    max_items: int | None = None
    use_amp: bool = True
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    window: str = "hann"


def get_device() -> torch.device:
    """Return the best available evaluation device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a U-Net checkpoint for evaluation."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model_config = UNetConfig(**checkpoint.get("model_config", {}))
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruct_waveform(
    magnitude: np.ndarray,
    phase: np.ndarray,
    config: EvaluateConfig,
    length: int | None = None,
) -> np.ndarray:
    """Reconstruct a waveform from magnitude and phase."""

    complex_stft = magnitude * np.exp(1j * phase)
    waveform = librosa.istft(
        complex_stft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
        length=length,
    )
    return np.asarray(waveform, dtype=np.float32)


def compute_sdr(reference: np.ndarray, estimate: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute signal-to-distortion ratio in decibels."""

    length = min(len(reference), len(estimate))
    reference = reference[:length]
    estimate = estimate[:length]
    distortion = reference - estimate
    signal_power = float(np.sum(reference**2))
    distortion_power = float(np.sum(distortion**2))
    return 10.0 * math.log10((signal_power + epsilon) / (distortion_power + epsilon))


def compute_stoi(sample_rate: int, reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Compute STOI when the optional package is available."""

    if stoi_score is None:
        return None
    length = min(len(reference), len(estimate))
    try:
        return float(stoi_score(reference[:length], estimate[:length], sample_rate, extended=False))
    except Exception:
        return None


def compute_pesq(sample_rate: int, reference: np.ndarray, estimate: np.ndarray) -> float | None:
    """Compute PESQ when the optional package is available."""

    if pesq_score is None:
        return None
    if sample_rate not in (8000, 16000):
        return None
    mode = "nb" if sample_rate == 8000 else "wb"
    length = min(len(reference), len(estimate))
    try:
        return float(pesq_score(sample_rate, reference[:length], estimate[:length], mode))
    except Exception:
        return None


def predict_full_mask(
    model: torch.nn.Module,
    noisy_log_magnitude: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Predict a mask for a full spectrogram."""

    tensor = torch.from_numpy(noisy_log_magnitude[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        prediction = model(tensor)
    return prediction.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def select_items(config: EvaluateConfig) -> list[Path]:
    """Select processed files for the requested split."""

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


def evaluate(config: EvaluateConfig) -> dict[str, float | int | None]:
    """Evaluate a checkpoint and write per-file metrics."""

    device = get_device()
    use_amp = config.use_amp and device.type == "cuda"
    model = load_checkpoint_model(config.checkpoint_path, device)
    processed_paths = select_items(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | None]] = []
    mse_values: list[float] = []
    input_snr_values: list[float] = []
    output_snr_values: list[float] = []
    snr_improvement_values: list[float] = []
    sdr_values: list[float] = []
    stoi_values: list[float] = []
    pesq_values: list[float] = []

    for processed_path in tqdm(processed_paths, desc="evaluating"):
        with np.load(processed_path) as data:
            noisy_log_magnitude = np.asarray(data["noisy_log_magnitude"], dtype=np.float32)
            noisy_magnitude = np.asarray(data["noisy_magnitude"], dtype=np.float32)
            clean_magnitude = np.asarray(data["clean_magnitude"], dtype=np.float32)
            noisy_phase = np.asarray(data["noisy_phase"], dtype=np.float32)
            ideal_mask = np.asarray(data["mask"], dtype=np.float32)
            sample_rate = int(data["sample_rate"])
            noisy_path = Path(str(data["noisy_path"]))
            clean_path = Path(str(data["clean_path"]))
            noise_type = str(data["noise_type"])
            snr = str(data["snr"])
            utterance_id = str(data["utterance_id"])

        predicted_mask = predict_full_mask(model, noisy_log_magnitude, device, use_amp)
        predicted_clean_magnitude = predicted_mask * noisy_magnitude
        mse = float(np.mean((predicted_mask - ideal_mask) ** 2))

        clean_audio, _ = librosa.load(clean_path, sr=sample_rate, mono=True)
        noisy_audio, _ = librosa.load(noisy_path, sr=sample_rate, mono=True)
        estimate_audio = reconstruct_waveform(
            predicted_clean_magnitude,
            noisy_phase,
            config,
            length=len(clean_audio),
        )
        input_snr = compute_sdr(clean_audio.astype(np.float32), noisy_audio.astype(np.float32))
        output_snr = compute_sdr(clean_audio.astype(np.float32), estimate_audio)
        snr_improvement = output_snr - input_snr
        sdr = compute_sdr(clean_audio.astype(np.float32), estimate_audio)
        stoi = compute_stoi(sample_rate, clean_audio.astype(np.float32), estimate_audio)
        pesq = compute_pesq(sample_rate, clean_audio.astype(np.float32), estimate_audio)

        mse_values.append(mse)
        input_snr_values.append(input_snr)
        output_snr_values.append(output_snr)
        snr_improvement_values.append(snr_improvement)
        sdr_values.append(sdr)
        if stoi is not None:
            stoi_values.append(stoi)
        if pesq is not None:
            pesq_values.append(pesq)
        rows.append(
            {
                "processed_path": str(processed_path),
                "noise_type": noise_type,
                "snr": snr,
                "utterance_id": utterance_id,
                "mask_mse": mse,
                "input_snr": input_snr,
                "output_snr": output_snr,
                "snr_improvement": snr_improvement,
                "sdr": sdr,
                "stoi": stoi,
                "pesq": pesq,
            }
        )

    with config.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "processed_path",
            "noise_type",
            "snr",
            "utterance_id",
            "mask_mse",
            "input_snr",
            "output_snr",
            "snr_improvement",
            "sdr",
            "stoi",
            "pesq",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, float | int | None] = {
        "items": len(rows),
        "mask_mse": float(np.mean(mse_values)) if mse_values else None,
        "input_snr": float(np.mean(input_snr_values)) if input_snr_values else None,
        "output_snr": float(np.mean(output_snr_values)) if output_snr_values else None,
        "snr_improvement": float(np.mean(snr_improvement_values)) if snr_improvement_values else None,
        "sdr": float(np.mean(sdr_values)) if sdr_values else None,
        "stoi": float(np.mean(stoi_values)) if stoi_values else None,
        "pesq": float(np.mean(pesq_values)) if pesq_values else None,
    }
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Items: {summary['items']}")
    print(f"Mask MSE: {summary['mask_mse']}")
    print(f"Input SNR: {summary['input_snr']}")
    print(f"Output SNR: {summary['output_snr']}")
    print(f"SNR improvement: {summary['snr_improvement']}")
    print(f"SDR: {summary['sdr']}")
    print(f"STOI: {summary['stoi']} {'(optional package unavailable or skipped)' if summary['stoi'] is None else ''}")
    print(f"PESQ: {summary['pesq']} {'(optional package unavailable or skipped)' if summary['pesq'] is None else ''}")
    print(f"Per-file metrics: {config.output_csv}")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate a trained speech denoising checkpoint.")
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/evaluation.csv"))
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--window", type=str, default="hann")
    return parser.parse_args()


def main() -> None:
    """Run the evaluation command-line interface."""

    args = parse_args()
    config = EvaluateConfig(
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        output_csv=args.output_csv,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split=args.split,
        max_items=args.max_items,
        use_amp=not args.no_amp,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=args.window,
    )
    evaluate(config)


if __name__ == "__main__":
    main()
