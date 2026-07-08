"""Training loop for the NOIZEUS speech denoising U-Net."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from src.dataset import DatasetConfig, create_dataloaders
    from src.model import UNetConfig, build_model, count_parameters
except ModuleNotFoundError:
    from dataset import DatasetConfig, create_dataloaders
    from model import UNetConfig, build_model, count_parameters


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for model training."""

    manifest_path: Path = Path("data/processed/manifest.csv")
    checkpoint_dir: Path = Path("checkpoints")
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    validation_fraction: float = 0.15
    window_frames: int = 32
    patience: int = 10
    min_delta: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    seed: int = 42
    augment: bool = False
    load_into_memory: bool = False
    use_amp: bool = True
    use_gradient_checkpointing: bool = False
    base_channels: int = 16
    output_scale: float = 1.0
    unet_depth: int = 2
    bottleneck_dropout: float = 0.0
    loss_mode: str = "mask_mse"
    mask_loss_weight: float = 1.0
    magnitude_loss_weight: float = 1.0
    target_mode: str = "stored"
    target_clip_max: float = 1.0
    show_progress: bool = True
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass
class EpochMetrics:
    """Loss metrics collected for one epoch."""

    train_loss: float
    val_loss: float
    learning_rate: float
    epoch_seconds: float


class EarlyStopping:
    """Track validation loss improvements and decide when to stop."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        """Initialize early stopping state."""

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        """Update state and return ``True`` if training should stop."""

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_seed(seed: int) -> None:
    """Set random seeds for repeatable training runs."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available training device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(
    predicted_mask: torch.Tensor,
    batch: dict[str, torch.Tensor | str],
    config: TrainConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute the configured training objective."""

    target_mask = batch["target"].to(device, non_blocking=True)
    mask_loss = nn.functional.mse_loss(predicted_mask, target_mask)
    if config.loss_mode == "mask_mse":
        return mask_loss
    if config.loss_mode == "mask_mse_mag_l1":
        noisy_magnitude = batch["noisy_magnitude"].to(device, non_blocking=True)
        clean_magnitude = batch["clean_magnitude"].to(device, non_blocking=True)
        estimated_clean_magnitude = predicted_mask * noisy_magnitude
        magnitude_loss = nn.functional.l1_loss(estimated_clean_magnitude, clean_magnitude)
        return config.mask_loss_weight * mask_loss + config.magnitude_loss_weight * magnitude_loss
    raise ValueError(f"Unknown loss_mode: {config.loss_mode}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor | str]],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    train_config: TrainConfig,
    max_batches: int | None = None,
) -> float:
    """Run one training epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, desc="train", leave=False, disable=not train_config.show_progress)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs = batch["input"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            predictions = model(inputs)
            loss = compute_loss(predictions, batch, train_config, device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = inputs.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / max(total_samples, 1))

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor | str]],
    device: torch.device,
    use_amp: bool,
    train_config: TrainConfig,
    max_batches: int | None = None,
) -> float:
    """Run one validation epoch and return the average loss."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(dataloader, desc="valid", leave=False, disable=not train_config.show_progress)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break

        inputs = batch["input"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            predictions = model(inputs)
            loss = compute_loss(predictions, batch, train_config, device)

        batch_size = inputs.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=total_loss / max(total_samples, 1))

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    metrics: EpochMetrics,
    train_config: TrainConfig,
    model_config: UNetConfig,
) -> None:
    """Save a training checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": asdict(metrics),
            "train_config": _json_ready(asdict(train_config)),
            "model_config": asdict(model_config),
        },
        path,
    )


def _json_ready(value: Any) -> Any:
    """Convert dataclass values into JSON-serializable objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def write_history(path: Path, history: list[EpochMetrics]) -> None:
    """Write training history as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [_json_ready(asdict(metrics)) for metrics in history]
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def train(config: TrainConfig) -> list[EpochMetrics]:
    """Train the U-Net and save checkpoints."""

    set_seed(config.seed)
    device = get_device()
    amp_enabled = config.use_amp and device.type == "cuda"
    dataset_config = DatasetConfig(
        manifest_path=config.manifest_path,
        window_frames=config.window_frames,
        augment=config.augment,
        load_into_memory=config.load_into_memory,
        include_magnitudes=config.loss_mode == "mask_mse_mag_l1",
        target_mode=config.target_mode,
        target_clip_max=config.target_clip_max,
    )
    train_loader, val_loader = create_dataloaders(
        dataset_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    model_config = UNetConfig(
        base_channels=config.base_channels,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        output_scale=config.output_scale,
        depth=config.unet_depth,
        bottleneck_dropout=config.bottleneck_dropout,
    )
    model = build_model(model_config).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    scaler = GradScaler(device.type, enabled=amp_enabled)
    stopper = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    history: list[EpochMetrics] = []

    print(f"Device: {device}")
    print(f"AMP enabled: {amp_enabled}")
    print(f"Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    print(f"Trainable parameters: {count_parameters(model)}")

    best_checkpoint = config.checkpoint_dir / "best_model.pt"
    latest_checkpoint = config.checkpoint_dir / "latest_model.pt"
    history_path = config.checkpoint_dir / "history.json"

    for epoch in range(1, config.epochs + 1):
        started_at = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=amp_enabled,
            train_config=config,
            max_batches=config.max_train_batches,
        )
        val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=amp_enabled,
            train_config=config,
            max_batches=config.max_val_batches,
        )
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        metrics = EpochMetrics(
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epoch_seconds=time.perf_counter() - started_at,
        )
        history.append(metrics)

        save_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            config,
            model_config,
        )
        improved = val_loss < stopper.best_loss - stopper.min_delta
        should_stop = stopper.step(val_loss)
        if improved:
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config,
                model_config,
            )
        write_history(history_path, history)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"lr={learning_rate:.2e} | seconds={metrics.epoch_seconds:.1f}"
        )
        if should_stop:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    return history


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train the speech denoising U-Net.")
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-into-memory", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--output-scale", type=float, default=1.0)
    parser.add_argument("--unet-depth", type=int, choices=[2, 3], default=2)
    parser.add_argument("--bottleneck-dropout", type=float, default=0.0)
    parser.add_argument("--loss-mode", choices=["mask_mse", "mask_mse_mag_l1"], default="mask_mse")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=["stored", "irm"], default="stored")
    parser.add_argument("--target-clip-max", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the training command-line interface."""

    args = parse_args()
    config = TrainConfig(
        manifest_path=args.manifest_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        window_frames=args.window_frames,
        patience=args.patience,
        min_delta=args.min_delta,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        seed=args.seed,
        augment=args.augment,
        load_into_memory=args.load_into_memory,
        use_amp=not args.no_amp,
        use_gradient_checkpointing=args.gradient_checkpointing,
        base_channels=args.base_channels,
        output_scale=args.output_scale,
        unet_depth=args.unet_depth,
        bottleneck_dropout=args.bottleneck_dropout,
        loss_mode=args.loss_mode,
        mask_loss_weight=args.mask_loss_weight,
        magnitude_loss_weight=args.magnitude_loss_weight,
        target_mode=args.target_mode,
        target_clip_max=args.target_clip_max,
        show_progress=not args.no_progress,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    train(config)


if __name__ == "__main__":
    main()
