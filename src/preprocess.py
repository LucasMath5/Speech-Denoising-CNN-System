"""Prepare NOIZEUS audio pairs for speech denoising runs.

This module extracts the NOIZEUS zip files, pairs each noisy utterance with
its clean reference, computes STFT features, and stores one processed ``.npz``
file per pair. The generated files are intentionally simple so the PyTorch
Dataset can load them either lazily or into memory in the next stage.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 128
DEFAULT_WIN_LENGTH = 512
DEFAULT_WINDOW = "hann"
DEFAULT_EPSILON = 1e-8
SPEAKER_ID_PATTERN = re.compile(r"^(sp\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class StftConfig:
    """Configuration used to compute STFT features."""

    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    win_length: int = DEFAULT_WIN_LENGTH
    window: str = DEFAULT_WINDOW
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True)
class AudioPair:
    """Path and metadata for a noisy-clean utterance pair."""

    pair_id: str
    noisy_path: Path
    clean_path: Path
    noise_type: str
    snr: str
    utterance_id: str


def extract_zips(zip_root: Path, raw_root: Path, overwrite: bool = False) -> None:
    """Extract every zip file from ``zip_root`` into an organized raw tree.

    Clean files are placed under ``raw/clean``. Noisy files are placed under
    ``raw/noisy/{noise_type}``, preserving the SNR folder stored inside each
    zip file.
    """

    zip_paths = sorted(zip_root.rglob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No zip files found under {zip_root}")

    raw_root.mkdir(parents=True, exist_ok=True)
    for zip_path in tqdm(zip_paths, desc="Extracting zips"):
        relative_parent = zip_path.parent.relative_to(zip_root)
        top_level = relative_parent.parts[0] if relative_parent.parts else zip_path.stem
        destination = raw_root / "clean" if top_level == "clean" else raw_root / "noisy" / top_level
        destination.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".wav"):
                    continue
                target_path = destination / member.filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if target_path.exists() and not overwrite:
                    continue
                with archive.open(member) as source, target_path.open("wb") as target:
                    target.write(source.read())


def utterance_id_from_path(path: Path) -> str:
    """Return the canonical utterance id, such as ``sp01``, from a wav path."""

    match = SPEAKER_ID_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Could not infer utterance id from {path.name}")
    return match.group(1).lower()


def build_pairs(raw_root: Path) -> list[AudioPair]:
    """Build noisy-clean pairs using the NOIZEUS ``spXX`` utterance ids."""

    clean_paths = sorted((raw_root / "clean").rglob("*.wav"))
    clean_by_id = {utterance_id_from_path(path): path for path in clean_paths}
    if not clean_by_id:
        raise FileNotFoundError(f"No clean wavs found under {raw_root / 'clean'}")

    pairs: list[AudioPair] = []
    noisy_root = raw_root / "noisy"
    for noisy_path in sorted(noisy_root.rglob("*.wav")):
        utterance_id = utterance_id_from_path(noisy_path)
        clean_path = clean_by_id.get(utterance_id)
        if clean_path is None:
            continue

        relative = noisy_path.relative_to(noisy_root)
        noise_type = relative.parts[0]
        snr = relative.parts[1] if len(relative.parts) > 2 else "unknown"
        pair_id = f"{noise_type}_{snr}_{utterance_id}"
        pairs.append(
            AudioPair(
                pair_id=pair_id,
                noisy_path=noisy_path,
                clean_path=clean_path,
                noise_type=noise_type,
                snr=snr,
                utterance_id=utterance_id,
            )
        )

    if not pairs:
        raise FileNotFoundError(f"No noisy-clean pairs found under {raw_root}")
    return pairs


def load_audio_pair(pair: AudioPair) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a noisy-clean pair as mono float32 arrays with matching lengths."""

    noisy, noisy_sr = sf.read(pair.noisy_path, dtype="float32", always_2d=False)
    clean, clean_sr = sf.read(pair.clean_path, dtype="float32", always_2d=False)
    if noisy_sr != clean_sr:
        clean = librosa.resample(clean, orig_sr=clean_sr, target_sr=noisy_sr)

    noisy = np.asarray(noisy, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=1)
    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)

    target_length = min(len(noisy), len(clean))
    return noisy[:target_length], clean[:target_length], noisy_sr


def compute_stft(audio: np.ndarray, config: StftConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute STFT magnitude and phase for a mono waveform."""

    stft = librosa.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=config.window,
    )
    magnitude = np.abs(stft).astype(np.float32)
    phase = np.angle(stft).astype(np.float32)
    return magnitude, phase


def process_pair(pair: AudioPair, processed_root: Path, config: StftConfig) -> Path:
    """Process one noisy-clean pair and save its features as an ``.npz`` file."""

    noisy_audio, clean_audio, sample_rate = load_audio_pair(pair)
    noisy_magnitude, noisy_phase = compute_stft(noisy_audio, config)
    clean_magnitude, clean_phase = compute_stft(clean_audio, config)
    mask = np.clip(clean_magnitude / (noisy_magnitude + config.epsilon), 0.0, 1.0).astype(np.float32)
    noisy_log_magnitude = np.log1p(noisy_magnitude).astype(np.float32)
    clean_log_magnitude = np.log1p(clean_magnitude).astype(np.float32)

    output_dir = processed_root / pair.noise_type / pair.snr
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pair.utterance_id}.npz"
    np.savez_compressed(
        output_path,
        noisy_log_magnitude=noisy_log_magnitude,
        clean_log_magnitude=clean_log_magnitude,
        noisy_magnitude=noisy_magnitude.astype(np.float32),
        clean_magnitude=clean_magnitude.astype(np.float32),
        noisy_phase=noisy_phase,
        clean_phase=clean_phase,
        mask=mask,
        sample_rate=np.array(sample_rate, dtype=np.int32),
        noisy_path=np.array(str(pair.noisy_path)),
        clean_path=np.array(str(pair.clean_path)),
        noise_type=np.array(pair.noise_type),
        snr=np.array(pair.snr),
        utterance_id=np.array(pair.utterance_id),
    )
    return output_path


def write_manifest(manifest_path: Path, rows: Iterable[dict[str, str]]) -> None:
    """Write the processed dataset manifest as CSV."""

    fieldnames = [
        "pair_id",
        "utterance_id",
        "noise_type",
        "snr",
        "noisy_path",
        "clean_path",
        "processed_path",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    zip_root: Path,
    project_root: Path,
    config: StftConfig,
    overwrite_raw: bool = False,
    overwrite_processed: bool = False,
) -> None:
    """Run extraction, pairing, STFT feature generation, and manifest writing."""

    raw_root = project_root / "data" / "raw"
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    extract_zips(zip_root=zip_root, raw_root=raw_root, overwrite=overwrite_raw)
    pairs = build_pairs(raw_root)

    rows: list[dict[str, str]] = []
    for pair in tqdm(pairs, desc="Processing STFT pairs"):
        output_path = processed_root / pair.noise_type / pair.snr / f"{pair.utterance_id}.npz"
        if overwrite_processed or not output_path.exists():
            output_path = process_pair(pair, processed_root, config)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "utterance_id": pair.utterance_id,
                "noise_type": pair.noise_type,
                "snr": pair.snr,
                "noisy_path": str(pair.noisy_path),
                "clean_path": str(pair.clean_path),
                "processed_path": str(output_path),
            }
        )

    write_manifest(processed_root / "manifest.csv", rows)
    print(f"Prepared {len(rows)} noisy-clean pairs.")
    print(f"Raw data: {raw_root}")
    print(f"Processed data: {processed_root}")
    print(f"Manifest: {processed_root / 'manifest.csv'}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""

    parser = argparse.ArgumentParser(description="Prepare NOIZEUS STFT denoising features.")
    parser.add_argument(
        "--zip-root",
        type=Path,
        default=Path("database/zipped"),
        help="Directory containing the NOIZEUS zip files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project directory where data/raw and data/processed will be created.",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--win-length", type=int, default=DEFAULT_WIN_LENGTH)
    parser.add_argument("--window", type=str, default=DEFAULT_WINDOW)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--overwrite-raw", action="store_true", help="Re-extract wav files even if they exist.")
    parser.add_argument(
        "--overwrite-processed",
        action="store_true",
        help="Recompute processed .npz files even if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preprocessing command-line entry point."""

    args = parse_args()
    config = StftConfig(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        window=args.window,
        epsilon=args.epsilon,
    )
    prepare_dataset(
        zip_root=args.zip_root,
        project_root=args.project_root,
        config=config,
        overwrite_raw=args.overwrite_raw,
        overwrite_processed=args.overwrite_processed,
    )


if __name__ == "__main__":
    main()
