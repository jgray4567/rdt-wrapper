"""Test ACT halting — sigmoid output, shape, and halting behavior."""

import torch
import pytest
from rdt_wrapper.halting import ACTHalting


class TestACTHalting:
    def test_output_range(self):
        """Halting probabilities are in (0, 1) via sigmoid."""
        halt = ACTHalting(dim=128)
        h = torch.randn(2, 16, 128)
        p = halt(h)
        assert (p > 0).all() and (p < 1).all(), "Probabilities must be in (0, 1)"

    def test_output_shape(self):
        """Output shape is (B, T) — squeezed from (B, T, 1)."""
        halt = ACTHalting(dim=128)
        h = torch.randn(2, 16, 128)
        p = halt(h)
        assert p.shape == (2, 16), f"Expected (2, 16), got {p.shape}"

    def test_param_count(self):
        """ACT halting adds dim + 1 params."""
        dim = 3072
        halt = ACTHalting(dim=dim)
        count = sum(p.numel() for p in halt.parameters())
        assert count == dim + 1, f"Expected {dim+1}, got {count}"

    def test_grad_flows(self):
        """Gradients flow through the halting head."""
        halt = ACTHalting(dim=128)
        h = torch.randn(2, 16, 128, requires_grad=True)
        p = halt(h)
        loss = p.sum()
        loss.backward()
        assert halt.halt.weight.grad is not None, "Gradient must reach halting weights"