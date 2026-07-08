"""Lightweight convolutional U-Net for speech denoising masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class UNetConfig:
    """Configuration for the compact denoising U-Net."""

    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    use_batch_norm: bool = True
    use_gradient_checkpointing: bool = False
    output_scale: float = 1.0
    depth: int = 2
    bottleneck_dropout: float = 0.0


class ConvBlock(nn.Module):
    """Two convolution layers with normalization and ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True) -> None:
        """Initialize a convolutional block."""

        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batch_norm),
            ]
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block."""

        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate the skip connection, and refine features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_batch_norm: bool = True) -> None:
        """Initialize an upsampling block."""

        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, use_batch_norm=use_batch_norm)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x`` and concatenate it with ``skip`` features."""

        x = self.up(x)
        x = self._match_spatial_shape(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))

    @staticmethod
    def _match_spatial_shape(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Pad or crop ``x`` so its spatial shape matches ``reference``."""

        diff_freq = reference.shape[-2] - x.shape[-2]
        diff_time = reference.shape[-1] - x.shape[-1]
        if diff_freq > 0 or diff_time > 0:
            x = nn.functional.pad(
                x,
                [
                    max(diff_time // 2, 0),
                    max(diff_time - diff_time // 2, 0),
                    max(diff_freq // 2, 0),
                    max(diff_freq - diff_freq // 2, 0),
                ],
            )
        if diff_freq < 0 or diff_time < 0:
            freq_start = max((-diff_freq) // 2, 0)
            time_start = max((-diff_time) // 2, 0)
            x = x[
                :,
                :,
                freq_start : freq_start + reference.shape[-2],
                time_start : time_start + reference.shape[-1],
            ]
        return x


class DenoisingUNet(nn.Module):
    """Compact U-Net that predicts an ideal-ratio-mask-like spectrogram mask."""

    def __init__(self, config: UNetConfig | None = None) -> None:
        """Initialize the U-Net."""

        super().__init__()
        self.config = config or UNetConfig()
        if self.config.depth not in (2, 3):
            raise ValueError("UNetConfig.depth must be 2 or 3")
        channels = self.config.base_channels

        self.encoder1 = ConvBlock(self.config.in_channels, channels, self.config.use_batch_norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(channels, channels * 2, self.config.use_batch_norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.config.depth == 3:
            self.encoder3 = ConvBlock(channels * 2, channels * 4, self.config.use_batch_norm)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.bottleneck = ConvBlock(channels * 4, channels * 8, self.config.use_batch_norm)
            self.decoder3 = UpBlock(channels * 8, channels * 4, channels * 4, self.config.use_batch_norm)
            decoder2_in_channels = channels * 4
        else:
            self.bottleneck = ConvBlock(channels * 2, channels * 4, self.config.use_batch_norm)
            decoder2_in_channels = channels * 4
        self.bottleneck_dropout = (
            nn.Dropout2d(self.config.bottleneck_dropout)
            if self.config.bottleneck_dropout > 0.0
            else nn.Identity()
        )
        self.decoder2 = UpBlock(decoder2_in_channels, channels * 2, channels * 2, self.config.use_batch_norm)
        self.decoder1 = UpBlock(channels * 2, channels, channels, self.config.use_batch_norm)
        self.output_conv = nn.Conv2d(channels, self.config.out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a denoising mask with the same shape as ``x``."""

        enc1 = self._run_block(self.encoder1, x)
        enc2 = self._run_block(self.encoder2, self.pool1(enc1))
        if self.config.depth == 3:
            enc3 = self._run_block(self.encoder3, self.pool2(enc2))
            bottleneck = self._run_block(self.bottleneck, self.pool3(enc3))
            bottleneck = self.bottleneck_dropout(bottleneck)
            dec3 = self.decoder3(bottleneck, enc3)
            dec2 = self.decoder2(dec3, enc2)
        else:
            bottleneck = self._run_block(self.bottleneck, self.pool2(enc2))
            bottleneck = self.bottleneck_dropout(bottleneck)
            dec2 = self.decoder2(bottleneck, enc2)
        dec1 = self.decoder1(dec2, enc1)
        return self.output_activation(self.output_conv(dec1)) * self.config.output_scale

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a block, optionally using gradient checkpointing during training."""

        if self.config.use_gradient_checkpointing and self.training and x.requires_grad:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of parameters in ``model``."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def build_model(config: UNetConfig | None = None) -> DenoisingUNet:
    """Create a denoising U-Net instance."""

    return DenoisingUNet(config=config)
