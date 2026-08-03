"""Test the full RDTModel with a small synthetic model.

Uses a tiny config to avoid needing real pretrained weights for CI.
"""

import torch
import pytest
from rdt_wrapper.config import RDTConfig

# Try importing transformers; skip tests if not available
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4):
    """Create a tiny Llama-style model for testing."""
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestRDTModel:
    def test_from_pretrained(self):
        """Model can be wrapped from a HuggingFace model."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(num_layers=4)
        config = RDTConfig(
            dim=128,
            n_layers=4,
            prelude_fraction=0.5,
            max_loop_iters=4,
            lora_rank=4,
        )
        model = RDTModel.from_pretrained(base, config)
        assert len(model.prelude_blocks) == 2
        assert len(model.recurrent_blocks) == 2
        assert len(model.coda_blocks) == 0

    def test_freeze_base(self):
        """Freezing base sets all base params to requires_grad=False."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)

        trainable = model.trainable_parameters()
        frozen = model.frozen_parameters()
        assert all(p.requires_grad for p in trainable), "New params should be trainable"
        assert all(not p.requires_grad for p in frozen), "Base params should be frozen"

    def test_forward_pass(self):
        """Forward pass produces logits of correct shape."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            output = model(input_ids)
        assert "logits" in output
        assert output["logits"].shape == (1, 8, 256)

    def test_forward_with_loss(self):
        """Forward pass with labels computes loss."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)

        input_ids = torch.randint(0, 256, (1, 8))
        labels = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            output = model(input_ids, labels=labels)
        assert "loss" in output
        assert output["loss"].dim() == 0  # scalar

    def test_generate(self):
        """Generate produces tokens beyond the prompt."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 4))
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=8, n_loops=4)
        assert output.shape == (1, 12), f"Expected (1, 12), got {output.shape}"

    def test_n_loops_override(self):
        """Can override n_loops at inference."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)
        model.eval()

        input_ids = torch.randint(0, 256, (1, 4))
        with torch.no_grad():
            out_2 = model(input_ids, n_loops=2)
            out_8 = model(input_ids, n_loops=8)
        assert out_2["logits"].shape == out_8["logits"].shape

    def test_info_string(self):
        """info() returns a readable summary."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)
        model._base_model_name = "test-model"
        info = model.info()
        assert "dim=128" in info
        assert "max_loops=4" in info
        assert "test-model" in info

    def test_new_param_count_is_tiny(self):
        """New params should be <1% of base model."""
        from rdt_wrapper.model import RDTModel
        base = create_tiny_model(vocab_size=256, hidden_size=128, num_layers=4)
        config = RDTConfig(dim=128, n_layers=4, max_loop_iters=4, lora_rank=4)
        model = RDTModel.from_pretrained(base, config)

        new_count = sum(p.numel() for p in model.trainable_parameters())
        base_count = sum(p.numel() for p in model.frozen_parameters())
        pct = (new_count / base_count) * 100
        assert pct < 1.0, f"New params should be <1% of base, got {pct:.2f}%"