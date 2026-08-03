"""Test config validation and param counting."""

import pytest
from rdt_wrapper.config import RDTConfig


class TestRDTConfig:
    def test_defaults(self):
        """Default config creates valid setup."""
        config = RDTConfig()
        assert config.prelude_layers == 24  # 48 * 0.5
        assert config.recurrent_layers == 24
        assert config.loop_embedding_dim == 384  # 3072 // 8

    def test_custom_prelude_fraction(self):
        """Custom prelude_fraction computes correct layer counts."""
        config = RDTConfig(dim=1024, n_layers=12, prelude_fraction=0.33)
        assert config.prelude_layers == 3  # int(12 * 0.33) = 3
        assert config.recurrent_layers == 9

    def test_coda_layers(self):
        """Coda layers reduce recurrent layers."""
        config = RDTConfig(dim=1024, n_layers=12, prelude_fraction=0.5, coda_layers=2)
        assert config.prelude_layers == 6
        assert config.recurrent_layers == 4  # 12 - 6 - 2

    def test_invalid_prelude_too_large(self):
        """Prelude >= n_layers should raise."""
        with pytest.raises(ValueError, match="prelude_layers"):
            RDTConfig(dim=128, n_layers=4, prelude_fraction=1.0)

    def test_invalid_no_recurrent(self):
        """No recurrent layers should raise."""
        with pytest.raises(ValueError, match="No recurrent layers"):
            RDTConfig(dim=128, n_layers=4, prelude_layers=2, coda_layers=2)

    def test_loop_embedding_dim_min(self):
        """Loop embedding dim has a minimum of 2."""
        config = RDTConfig(dim=8, n_layers=4)
        assert config.loop_embedding_dim >= 2

    def test_new_param_count(self):
        """Param count matches expected formula."""
        config = RDTConfig(dim=3072, n_layers=48, max_loop_iters=16, lora_rank=16)
        count = config.new_param_count()
        # LTI: 2*3072 + 1 = 6145
        # ACT: 3072 + 1 = 3073
        # LoRA: 3072*16 + 16*3072 + 16*16 = 49152 + 49152 + 256 = 98560
        # Norm: 3072
        expected = 6145 + 3073 + 98560 + 3072
        assert count == expected, f"Expected {expected}, got {count}"

    def test_new_param_percentage(self):
        """Percentage is computed correctly."""
        config = RDTConfig(dim=3072, n_layers=48, max_loop_iters=16, lora_rank=16)
        # Simulate a 12B param model
        base_count = 12_000_000_000
        pct = config.new_param_percentage(base_count)
        assert pct < 0.01, f"Should be <0.01%, got {pct:.4f}%"