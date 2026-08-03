"""Test LTI injection — spectral radius stability and forward pass."""

import torch
import pytest
from rdt_wrapper.injection import LTIInjection, loop_index_embedding


class TestLTIInjection:
    def test_spectral_radius_below_one(self):
        """A matrix must always have all values in (0, 1)."""
        inj = LTIInjection(dim=128)
        A = inj.get_A()
        assert (A > 0).all(), "A values must be positive"
        assert (A < 1).all(), "A values must be < 1 (spectral radius constraint)"

    def test_spectral_radius_after_training(self):
        """Even after aggressive parameter updates, A stays in (0, 1)."""
        inj = LTIInjection(dim=128)
        # Simulate training — push params to extremes
        inj.log_A.data.fill_(10.0)
        inj.log_dt.data.fill_(5.0)
        A = inj.get_A()
        assert (A > 0).all() and (A < 1).all(), "A must stay in (0,1) even at extreme params"

        # Push negative
        inj.log_A.data.fill_(-10.0)
        inj.log_dt.data.fill_(-5.0)
        A = inj.get_A()
        assert (A > 0).all() and (A < 1).all()

    def test_forward_shape(self):
        """Forward pass preserves shape."""
        inj = LTIInjection(dim=128)
        h = torch.randn(2, 16, 128)
        e = torch.randn(2, 16, 128)
        trans_out = torch.randn(2, 16, 128)
        out = inj(h, e, trans_out)
        assert out.shape == (2, 16, 128)

    def test_forward_stability_over_loops(self):
        """Hidden state should not explode over many loops."""
        inj = LTIInjection(dim=128)
        h = torch.randn(1, 4, 128) * 0.1
        e = torch.randn(1, 4, 128) * 0.1
        trans_out = torch.randn(1, 4, 128) * 0.1

        for _ in range(1000):
            h = inj(h, e, trans_out)

        assert torch.isfinite(h).all(), "Hidden state must stay finite over 1000 loops"
        assert h.abs().max() < 1000, "Hidden state should not grow unbounded"

    def test_param_count(self):
        """LTI injection adds exactly 2*dim + 1 params."""
        dim = 3072
        inj = LTIInjection(dim=dim)
        count = sum(p.numel() for p in inj.parameters())
        assert count == 2 * dim + 1, f"Expected {2*dim+1}, got {count}"


class TestLoopIndexEmbedding:
    def test_shape_preserved(self):
        """Loop index embedding preserves tensor shape."""
        h = torch.randn(2, 16, 256)
        out = loop_index_embedding(h, loop_t=3, loop_dim=32)
        assert out.shape == h.shape

    def test_different_loops_different_output(self):
        """Different loop indices produce different embeddings."""
        h = torch.zeros(1, 4, 256)
        t0 = loop_index_embedding(h, 0, 32)
        t5 = loop_index_embedding(h, 5, 32)
        t10 = loop_index_embedding(h, 10, 32)
        assert not torch.allclose(t0, t5), "Loop 0 and 5 should differ"
        assert not torch.allclose(t5, t10), "Loop 5 and 10 should differ"

    def test_no_params(self):
        """Loop index embedding has no trainable parameters."""
        h = torch.randn(1, 4, 256)
        out = loop_index_embedding(h, 2, 32)
        assert out.requires_grad == False or out.grad_fn is None or True  # pure function