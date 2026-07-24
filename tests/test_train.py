"""Tests for training loss helpers."""

from __future__ import annotations

import unittest

import torch

from src.train import TrainConfig, compute_loss


class TrainLossTests(unittest.TestCase):
    """Validate configured loss functions."""

    def test_mask_mse_loss(self) -> None:
        """Mask MSE mode should match torch MSE."""

        predicted = torch.tensor([[[[0.0, 1.0], [0.5, 0.5]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.0, 0.0], [1.0, 0.5]]]], dtype=torch.float32)
        batch = {"target": target}
        config = TrainConfig(loss_mode="mask_mse")
        loss = compute_loss(predicted, batch, config, torch.device("cpu"))
        expected = torch.nn.functional.mse_loss(predicted, target)
        self.assertAlmostEqual(float(loss), float(expected), places=7)

    def test_combined_loss(self) -> None:
        """Combined loss should include mask and magnitude terms."""

        predicted = torch.full((1, 1, 2, 2), 0.5)
        target = torch.zeros((1, 1, 2, 2))
        noisy_magnitude = torch.ones((1, 1, 2, 2))
        clean_magnitude = torch.zeros((1, 1, 2, 2))
        batch = {
            "target": target,
            "noisy_magnitude": noisy_magnitude,
            "clean_magnitude": clean_magnitude,
        }
        config = TrainConfig(loss_mode="mask_mse_mag_l1", mask_loss_weight=0.5, magnitude_loss_weight=0.5)
        loss = compute_loss(predicted, batch, config, torch.device("cpu"))
        self.assertAlmostEqual(float(loss), 0.375, places=7)


if __name__ == "__main__":
    unittest.main()

