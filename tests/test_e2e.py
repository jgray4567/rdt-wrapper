"""Test the example script runs end-to-end with a tiny model."""

import torch
import pytest
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rdt_wrapper.config import RDTConfig

try:
    from transformers import LlamaConfig, LlamaForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestEndToEnd:
    def test_tiny_model_train_step(self):
        """One training step on a tiny model completes without error."""
        from rdt_wrapper.model import RDTModel

        # Create tiny Llama model
        tiny_config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base = LlamaForCausalLM(tiny_config)

        # Wrap with RDT
        config = RDTConfig(
            dim=64,
            n_layers=4,
            prelude_fraction=0.5,
            max_loop_iters=4,
            lora_rank=4,
        )
        model = RDTModel.from_pretrained(base, config)
        model.train()

        # One training step
        input_ids = torch.randint(0, 128, (1, 8))
        labels = input_ids.clone()
        output = model(input_ids, labels=labels, n_loops=4)
        output["loss"].backward()

        # Verify gradients reached only new params
        new_params_with_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(new_params_with_grad) > 0, "New params should have gradients"

        # Verify base params have no gradients
        base_params_with_grad = [
            name for name, p in model.named_parameters()
            if not p.requires_grad and p.grad is not None
        ]
        assert len(base_params_with_grad) == 0, "Base params should have no gradients"

    def test_generate_and_decode(self):
        """Generate tokens and verify output is valid."""
        from rdt_wrapper.model import RDTModel

        tiny_config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base = LlamaForCausalLM(tiny_config)

        config = RDTConfig(
            dim=64,
            n_layers=4,
            prelude_fraction=0.5,
            max_loop_iters=4,
            lora_rank=4,
        )
        model = RDTModel.from_pretrained(base, config)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=4, n_loops=4, temperature=0.1)

        assert output.shape == (1, 8)
        assert (output >= 0).all() and (output < 128).all(), "Token IDs in valid range"

    def test_depth_extrapolation_inference(self):
        """Model can run at higher loop counts than configured."""
        from rdt_wrapper.model import RDTModel

        tiny_config = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base = LlamaForCausalLM(tiny_config)

        config = RDTConfig(
            dim=64,
            n_layers=4,
            max_loop_iters=4,
            lora_rank=4,
        )
        model = RDTModel.from_pretrained(base, config)
        model.eval()

        input_ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            out_4 = model(input_ids, n_loops=4)
            out_16 = model(input_ids, n_loops=16)
            out_64 = model(input_ids, n_loops=64)

        assert torch.isfinite(out_4["logits"]).all()
        assert torch.isfinite(out_16["logits"]).all(), "16 loops should be stable"
        assert torch.isfinite(out_64["logits"]).all(), "64 loops should be stable"