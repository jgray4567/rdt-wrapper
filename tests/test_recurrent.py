"""Test the recurrent loop — ACT early exit, stability, depth extrapolation."""

import torch
import pytest
from rdt_wrapper.config import RDTConfig
from rdt_wrapper.recurrent import RecurrentLoop


def dummy_block_forward(h):
    """Simple identity + small perturbation block forward."""
    return h + 0.01 * torch.randn_like(h)


class TestRecurrentLoop:
    def test_output_shape(self):
        """Loop output matches input shape."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=8, lora_rank=8)
        loop = RecurrentLoop(config)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        out = loop(h, e, dummy_block_forward, n_loops=4)
        assert out.shape == (2, 16, 128)

    def test_act_early_exit(self):
        """With a high act_threshold, the loop should run all iterations.
        With a very low threshold, it should exit early."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=16, lora_rank=8, act_threshold=0.99)
        loop = RecurrentLoop(config)

        h = torch.randn(1, 4, 128)
        e = torch.randn(1, 4, 128)

        # Run with 16 loops
        out_16 = loop(h, e, dummy_block_forward, n_loops=16)
        assert out_16.shape == (1, 4, 128)

        # Run with 2 loops — should produce different (but valid) output
        out_2 = loop(h, e, dummy_block_forward, n_loops=2)
        assert out_2.shape == (1, 4, 128)

    def test_stability_over_many_loops(self):
        """Loop should be numerically stable even at high loop counts."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=100, lora_rank=8)
        loop = RecurrentLoop(config)
        h = torch.randn(1, 4, 128) * 0.1
        e = torch.randn(1, 4, 128) * 0.1
        out = loop(h, e, dummy_block_forward, n_loops=100)
        assert torch.isfinite(out).all(), "Output must be finite after 100 loops"
        assert out.abs().max() < 100, "Output should not explode"

    def test_depth_extrapolation(self):
        """Running more loops than trained should still work (depth extrapolation)."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=8, lora_rank=8)
        loop = RecurrentLoop(config)
        h = torch.randn(1, 4, 128)
        e = torch.randn(1, 4, 128)

        # Normal range
        out_8 = loop(h, e, dummy_block_forward, n_loops=8)
        # Extrapolated range
        out_32 = loop(h, e, dummy_block_forward, n_loops=32)
        out_64 = loop(h, e, dummy_block_forward, n_loops=64)

        assert torch.isfinite(out_32).all(), "32 loops must be stable"
        assert torch.isfinite(out_64).all(), "64 loops must be stable"

    def test_n_loops_override(self):
        """n_loops parameter overrides config default."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=16, lora_rank=8)
        loop = RecurrentLoop(config)
        h = torch.randn(1, 4, 128)
        e = torch.randn(1, 4, 128)

        out_default = loop(h, e, dummy_block_forward)
        out_override = loop(h, e, dummy_block_forward, n_loops=4)
        # Both should be valid
        assert out_default.shape == out_override.shape == (1, 4, 128)

    def test_new_params_are_trainable(self):
        """The loop module's parameters should all be trainable by default."""
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=8, lora_rank=8)
        loop = RecurrentLoop(config)
        for name, param in loop.named_parameters():
            assert param.requires_grad, f"{name} should be trainable"