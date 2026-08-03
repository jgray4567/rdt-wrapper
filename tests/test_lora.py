"""Test depth-wise LoRA adapter — shape, per-loop differentiation, param count."""

import torch
import pytest
from rdt_wrapper.lora_adapter import DepthLoRAAdapter


class TestDepthLoRAAdapter:
    def test_output_shape(self):
        """LoRA delta has same shape as input."""
        adapter = DepthLoRAAdapter(dim=256, rank=16, max_loops=8)
        x = torch.randn(2, 16, 256)
        out = adapter(x, loop_t=0)
        assert out.shape == x.shape

    def test_different_loops_different_output(self):
        """Different loop indices produce different deltas."""
        adapter = DepthLoRAAdapter(dim=256, rank=16, max_loops=8)
        x = torch.randn(1, 4, 256)
        out0 = adapter(x, 0)
        out3 = adapter(x, 3)
        out7 = adapter(x, 7)
        assert not torch.allclose(out0, out3), "Loop 0 and 3 should differ"
        assert not torch.allclose(out3, out7), "Loop 3 and 7 should differ"

    def test_extrapolation_beyond_max_loops(self):
        """Loop index beyond max_loops should clamp, not crash."""
        adapter = DepthLoRAAdapter(dim=256, rank=16, max_loops=8)
        x = torch.randn(1, 4, 256)
        out = adapter(x, loop_t=20)  # beyond max_loops=8
        assert out.shape == x.shape, "Should handle loop_t > max_loops via clamping"

    def test_param_count(self):
        """Param count: dim*rank + rank*dim + max_loops*rank."""
        dim, rank, max_loops = 3072, 16, 16
        adapter = DepthLoRAAdapter(dim=dim, rank=rank, max_loops=max_loops)
        count = sum(p.numel() for p in adapter.parameters())
        expected = dim * rank + rank * dim + max_loops * rank
        assert count == expected, f"Expected {expected}, got {count}"

    def test_grad_flows(self):
        """Gradients flow through the adapter."""
        adapter = DepthLoRAAdapter(dim=256, rank=16, max_loops=8)
        x = torch.randn(1, 4, 256, requires_grad=True)
        out = adapter(x, 0)
        loss = out.sum()
        loss.backward()
        assert adapter.down.weight.grad is not None
        assert adapter.B.grad is not None
        assert adapter.scale.weight.grad is not None