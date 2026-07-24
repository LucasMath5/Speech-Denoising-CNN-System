"""Tests for processed spectrogram dataset utilities."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset import DatasetConfig, ManifestItem, NoizeusSpectrogramDataset, read_manifest, split_items


def write_npz(path: Path, frames: int) -> None:
    """Write a minimal processed spectrogram file."""

    freq_bins = 257
    noisy_magnitude = np.ones((freq_bins, frames), dtype=np.float32)
    clean_magnitude = np.full((freq_bins, frames), 0.5, dtype=np.float32)
    np.savez_compressed(
        path,
        noisy_log_magnitude=np.log1p(noisy_magnitude).astype(np.float32),
        noisy_magnitude=noisy_magnitude,
        clean_magnitude=clean_magnitude,
        mask=np.clip(clean_magnitude / (noisy_magnitude + 1e-8), 0.0, 1.0).astype(np.float32),
    )


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a manifest compatible with the dataset loader."""

    fieldnames = ["pair_id", "utterance_id", "noise_type", "snr", "noisy_path", "clean_path", "processed_path"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DatasetTests(unittest.TestCase):
    """Validate dataset indexing, loading, and split behavior."""

    def test_dataset_reads_manifest_and_builds_windows(self) -> None:
        """Deterministic windowing should pad short files and cover final frames."""

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            first_npz = tmp_dir / "first.npz"
            second_npz = tmp_dir / "second.npz"
            manifest_path = tmp_dir / "manifest.csv"
            write_npz(first_npz, frames=40)
            write_npz(second_npz, frames=20)
            write_manifest(
                manifest_path,
                [
                    {
                        "pair_id": "airport_0dB_sp01",
                        "utterance_id": "sp01",
                        "noise_type": "airport",
                        "snr": "0dB",
                        "noisy_path": "noisy_sp01.wav",
                        "clean_path": "clean_sp01.wav",
                        "processed_path": str(first_npz),
                    },
                    {
                        "pair_id": "airport_0dB_sp02",
                        "utterance_id": "sp02",
                        "noise_type": "airport",
                        "snr": "0dB",
                        "noisy_path": "noisy_sp02.wav",
                        "clean_path": "clean_sp02.wav",
                        "processed_path": str(second_npz),
                    },
                ],
            )

            items = read_manifest(manifest_path)
            config = DatasetConfig(manifest_path=manifest_path, window_frames=32, random_windows=False)
            dataset = NoizeusSpectrogramDataset(config, items=items)

            self.assertEqual(len(dataset), 3)
            sample = dataset[0]
            self.assertEqual(tuple(sample["input"].shape), (1, 257, 32))
            self.assertEqual(tuple(sample["target"].shape), (1, 257, 32))

    def test_utterance_split_has_no_leakage(self) -> None:
        """All variants of an utterance should stay in the same split."""

        items = [
            ManifestItem(f"pair_{utterance}_{index}", utterance, "noise", "0dB", Path("n.wav"), Path("c.wav"), Path("p.npz"))
            for utterance in ("sp01", "sp02", "sp03", "sp04")
            for index in range(2)
        ]
        train_items, validation_items = split_items(items, validation_fraction=0.5, seed=7, strategy="utterance")
        train_ids = {item.utterance_id for item in train_items}
        validation_ids = {item.utterance_id for item in validation_items}
        self.assertTrue(train_ids)
        self.assertTrue(validation_ids)
        self.assertTrue(train_ids.isdisjoint(validation_ids))


if __name__ == "__main__":
    unittest.main()

