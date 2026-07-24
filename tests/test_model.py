"""Smoke tests for the denoising U-Net."""

from __future__ import annotations

import unittest

import torch

from src.model import UNetConfig, build_model, count_parameters


class DenoisingUNetTests(unittest.TestCase):
    """Validate model construction and forward shapes."""

    def test_depth2_preserves_input_shape(self) -> None:
        """Depth-2 U-Net should return a mask with the input shape."""

        model = build_model(UNetConfig(base_channels=4, depth=2)).eval()
        inputs = torch.rand(2, 1, 257, 32)
        with torch.no_grad():
            outputs = model(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertGreaterEqual(float(outputs.min()), 0.0)
        self.assertLessEqual(float(outputs.max()), 1.0)

    def test_depth3_preserves_input_shape(self) -> None:
        """Depth-3 U-Net should also handle odd frequency dimensions."""

        model = build_model(UNetConfig(base_channels=4, depth=3, bottleneck_dropout=0.1)).eval()
        inputs = torch.rand(2, 1, 257, 64)
        with torch.no_grad():
            outputs = model(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_depth3_has_more_parameters_than_depth2(self) -> None:
        """The deeper configuration should increase model capacity."""

        depth2 = build_model(UNetConfig(base_channels=4, depth=2))
        depth3 = build_model(UNetConfig(base_channels=4, depth=3))
        self.assertGreater(count_parameters(depth3), count_parameters(depth2))

    def test_invalid_depth_raises_error(self) -> None:
        """Only the supported U-Net depths should be accepted."""

        with self.assertRaises(ValueError):
            build_model(UNetConfig(depth=4))


if __name__ == "__main__":
    unittest.main()

