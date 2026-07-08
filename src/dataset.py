"""PyTorch Dataset and DataLoader utilities for processed NOIZEUS features."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


SplitStrategy = Literal["utterance", "pair"]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for spectrogram window loading."""

    manifest_path: Path = Path("data/processed/manifest.csv")
    window_frames: int = 32
    input_key: str = "noisy_log_magnitude"
    target_key: str = "mask"
    target_mode: str = "stored"
    target_clip_max: float = 1.0
    epsilon: float = 1e-8
    include_magnitudes: bool = False
    random_windows: bool = True
    augment: bool = False
    noise_std: float = 0.01
    load_into_memory: bool = False


@dataclass(frozen=True)
class ManifestItem:
    """One processed noisy-clean pair listed in the manifest."""

    pair_id: str
    utterance_id: str
    noise_type: str
    snr: str
    noisy_path: Path
    clean_path: Path
    processed_path: Path


@dataclass(frozen=True)
class WindowIndex:
    """Mapping from a dataset index to a fixed time window."""

    item_index: int
    start_frame: int


def read_manifest(manifest_path: Path) -> list[ManifestItem]:
    """Read processed pair metadata from a CSV manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    items: list[ManifestItem] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            items.append(
                ManifestItem(
                    pair_id=row["pair_id"],
                    utterance_id=row["utterance_id"],
                    noise_type=row["noise_type"],
                    snr=row["snr"],
                    noisy_path=Path(row["noisy_path"]),
                    clean_path=Path(row["clean_path"]),
                    processed_path=Path(row["processed_path"]),
                )
            )

    if not items:
        raise ValueError(f"Manifest has no data rows: {manifest_path}")
    return items


def _load_npz_arrays(
    path: Path,
    input_key: str,
    target_key: str,
    target_mode: str = "stored",
    target_clip_max: float = 1.0,
    epsilon: float = 1e-8,
    include_magnitudes: bool = False,
) -> dict[str, np.ndarray]:
    """Load input and target arrays from one processed ``.npz`` file."""

    with np.load(path) as data:
        inputs = np.asarray(data[input_key], dtype=np.float32)
        if target_mode == "irm":
            noisy_magnitude = np.asarray(data["noisy_magnitude"], dtype=np.float32)
            clean_magnitude = np.asarray(data["clean_magnitude"], dtype=np.float32)
            targets = np.clip(clean_magnitude / (noisy_magnitude + epsilon), 0.0, target_clip_max).astype(np.float32)
        else:
            targets = np.asarray(data[target_key], dtype=np.float32)
        arrays = {"input": inputs, "target": targets}
        if include_magnitudes:
            arrays["noisy_magnitude"] = np.asarray(data["noisy_magnitude"], dtype=np.float32)
            arrays["clean_magnitude"] = np.asarray(data["clean_magnitude"], dtype=np.float32)
    return arrays


def _pad_time_axis(array: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad a spectrogram on the time axis until it has ``target_frames``."""

    if array.shape[1] >= target_frames:
        return array
    pad_width = target_frames - array.shape[1]
    return np.pad(array, ((0, 0), (0, pad_width)), mode="constant")


class NoizeusSpectrogramDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset of noisy spectrogram windows and ideal ratio masks.

    Each item returns tensors shaped ``(1, freq_bins, window_frames)`` so they
    can be fed directly into a convolutional U-Net.
    """

    def __init__(self, config: DatasetConfig, items: list[ManifestItem] | None = None) -> None:
        """Create a dataset from a manifest or a provided item subset."""

        self.config = config
        self.items = items if items is not None else read_manifest(config.manifest_path)
        self._cache: list[dict[str, np.ndarray]] | None = None
        if config.load_into_memory:
            self._cache = [
                _load_npz_arrays(
                    item.processed_path,
                    config.input_key,
                    config.target_key,
                    config.target_mode,
                    config.target_clip_max,
                    config.epsilon,
                    config.include_magnitudes,
                )
                for item in self.items
            ]

        self.windows = self._build_windows() if not config.random_windows else []

    def __len__(self) -> int:
        """Return the number of available samples."""

        if self.config.random_windows:
            return len(self.items)
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Return one spectrogram window and its target mask."""

        if self.config.random_windows:
            item_index = index
            arrays = self._load_item(item_index)
            inputs = arrays["input"]
            start_frame = self._sample_start_frame(inputs.shape[1])
        else:
            window = self.windows[index]
            item_index = window.item_index
            start_frame = window.start_frame
            arrays = self._load_item(item_index)
            inputs = arrays["input"]

        targets = arrays["target"]
        input_window = self._slice_window(inputs, start_frame)
        target_window = self._slice_window(targets, start_frame)
        if self.config.augment:
            input_window = self._augment_input(input_window)

        item = self.items[item_index]
        sample: dict[str, torch.Tensor | str] = {
            "input": torch.from_numpy(input_window[None, :, :].copy()),
            "target": torch.from_numpy(target_window[None, :, :].copy()),
            "pair_id": item.pair_id,
            "utterance_id": item.utterance_id,
            "noise_type": item.noise_type,
            "snr": item.snr,
        }
        if self.config.include_magnitudes:
            sample["noisy_magnitude"] = torch.from_numpy(
                self._slice_window(arrays["noisy_magnitude"], start_frame)[None, :, :].copy()
            )
            sample["clean_magnitude"] = torch.from_numpy(
                self._slice_window(arrays["clean_magnitude"], start_frame)[None, :, :].copy()
            )
        return sample

    def _load_item(self, item_index: int) -> dict[str, np.ndarray]:
        """Load arrays for one manifest item from cache or disk."""

        if self._cache is not None:
            return self._cache[item_index]
        item = self.items[item_index]
        return _load_npz_arrays(
            item.processed_path,
            self.config.input_key,
            self.config.target_key,
            self.config.target_mode,
            self.config.target_clip_max,
            self.config.epsilon,
            self.config.include_magnitudes,
        )

    def _build_windows(self) -> list[WindowIndex]:
        """Build deterministic non-overlapping window indices."""

        windows: list[WindowIndex] = []
        for item_index, item in enumerate(self.items):
            arrays = _load_npz_arrays(
                item.processed_path,
                self.config.input_key,
                self.config.target_key,
                self.config.target_mode,
                self.config.target_clip_max,
                self.config.epsilon,
                False,
            )
            inputs = arrays["input"]
            total_frames = inputs.shape[1]
            if total_frames <= self.config.window_frames:
                windows.append(WindowIndex(item_index=item_index, start_frame=0))
                continue
            last_start = total_frames - self.config.window_frames
            start_frames = list(range(0, last_start + 1, self.config.window_frames))
            if start_frames[-1] != last_start:
                start_frames.append(last_start)
            for start_frame in start_frames:
                windows.append(WindowIndex(item_index=item_index, start_frame=start_frame))
        return windows

    def _sample_start_frame(self, total_frames: int) -> int:
        """Sample a valid start frame for random-window training."""

        max_start = max(0, total_frames - self.config.window_frames)
        return random.randint(0, max_start)

    def _slice_window(self, array: np.ndarray, start_frame: int) -> np.ndarray:
        """Slice or pad a spectrogram to the configured window length."""

        padded = _pad_time_axis(array, start_frame + self.config.window_frames)
        return padded[:, start_frame : start_frame + self.config.window_frames].astype(np.float32)

    def _augment_input(self, input_window: np.ndarray) -> np.ndarray:
        """Apply lightweight additive noise augmentation to log magnitudes."""

        if self.config.noise_std <= 0.0:
            return input_window
        noise = np.random.normal(0.0, self.config.noise_std, size=input_window.shape).astype(np.float32)
        return np.maximum(input_window + noise, 0.0).astype(np.float32)


def split_items(
    items: list[ManifestItem],
    validation_fraction: float = 0.15,
    seed: int = 42,
    strategy: SplitStrategy = "utterance",
) -> tuple[list[ManifestItem], list[ManifestItem]]:
    """Split manifest items into train and validation subsets.

    The default ``utterance`` strategy keeps all noise/SNR variants for a given
    utterance id in the same split, avoiding leakage through repeated content.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = random.Random(seed)
    if strategy == "pair":
        shuffled = list(items)
        rng.shuffle(shuffled)
        validation_size = max(1, int(round(len(shuffled) * validation_fraction)))
        return shuffled[validation_size:], shuffled[:validation_size]

    utterance_ids = sorted({item.utterance_id for item in items})
    rng.shuffle(utterance_ids)
    validation_count = max(1, int(round(len(utterance_ids) * validation_fraction)))
    validation_ids = set(utterance_ids[:validation_count])
    train_items = [item for item in items if item.utterance_id not in validation_ids]
    validation_items = [item for item in items if item.utterance_id in validation_ids]
    return train_items, validation_items


def create_dataloaders(
    config: DatasetConfig,
    batch_size: int = 16,
    num_workers: int = 2,
    validation_fraction: float = 0.15,
    seed: int = 42,
    split_strategy: SplitStrategy = "utterance",
) -> tuple[DataLoader[dict[str, torch.Tensor | str]], DataLoader[dict[str, torch.Tensor | str]]]:
    """Create train and validation DataLoaders from a processed manifest."""

    items = read_manifest(config.manifest_path)
    train_items, validation_items = split_items(
        items,
        validation_fraction=validation_fraction,
        seed=seed,
        strategy=split_strategy,
    )

    train_config = DatasetConfig(
        manifest_path=config.manifest_path,
        window_frames=config.window_frames,
        input_key=config.input_key,
        target_key=config.target_key,
        target_mode=config.target_mode,
        target_clip_max=config.target_clip_max,
        epsilon=config.epsilon,
        include_magnitudes=config.include_magnitudes,
        random_windows=True,
        augment=config.augment,
        noise_std=config.noise_std,
        load_into_memory=config.load_into_memory,
    )
    validation_config = DatasetConfig(
        manifest_path=config.manifest_path,
        window_frames=config.window_frames,
        input_key=config.input_key,
        target_key=config.target_key,
        target_mode=config.target_mode,
        target_clip_max=config.target_clip_max,
        epsilon=config.epsilon,
        include_magnitudes=config.include_magnitudes,
        random_windows=False,
        augment=False,
        noise_std=0.0,
        load_into_memory=config.load_into_memory,
    )
    train_dataset = NoizeusSpectrogramDataset(train_config, items=train_items)
    validation_dataset = NoizeusSpectrogramDataset(validation_config, items=validation_items)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader


def create_random_split_dataloaders(
    config: DatasetConfig,
    batch_size: int = 16,
    num_workers: int = 2,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader[dict[str, torch.Tensor | str]], DataLoader[dict[str, torch.Tensor | str]]]:
    """Create DataLoaders by randomly splitting a single dataset instance."""

    dataset = NoizeusSpectrogramDataset(config)
    validation_size = max(1, int(round(len(dataset) * validation_fraction)))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader
