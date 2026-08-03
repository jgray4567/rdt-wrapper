"""Configuration for the RDT wrapper."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RDTConfig:
    """Configuration for wrapping a pretrained transformer with recurrent depth.

    Attributes:
        dim: Hidden dimension of the base model.
        n_layers: Total number of transformer blocks in the base model.
        prelude_fraction: Fraction of layers that run once (prelude).
            The remaining layers form the recurrent block. Default 0.5.
        max_loop_iters: Maximum recurrent loop depth at inference.
        act_threshold: ACT cumulative probability threshold for halting.
            Higher = more loops (deeper compute). Default 0.99.
        lora_rank: Rank of the depth-wise LoRA adapter.
        loop_embedding_dim: Number of channels receiving loop-index embedding.
            Default dim // 8.
        rope_theta: Base frequency for loop-index sinusoidal embedding.
        prelude_layers: Number of prelude layers (computed from prelude_fraction
            if not set explicitly).
        coda_layers: Number of coda layers after the recurrent block.
            Default 0 (the recurrent block feeds directly to the LM head).
    """

    dim: int = 3072
    n_layers: int = 48
    prelude_fraction: float = 0.5
    max_loop_iters: int = 16
    act_threshold: float = 0.99
    lora_rank: int = 16
    loop_embedding_dim: Optional[int] = None
    rope_theta: float = 10000.0
    prelude_layers: Optional[int] = None
    coda_layers: int = 0

    def __post_init__(self):
        if self.prelude_layers is None:
            self.prelude_layers = int(self.n_layers * self.prelude_fraction)
        if self.loop_embedding_dim is None:
            self.loop_embedding_dim = max(self.dim // 8, 2)
        # Validate
        if self.prelude_layers >= self.n_layers:
            raise ValueError(
                f"prelude_layers ({self.prelude_layers}) must be < n_layers ({self.n_layers})"
            )
        recurrent = self.n_layers - self.prelude_layers - self.coda_layers
        if recurrent <= 0:
            raise ValueError(
                f"No recurrent layers: prelude={self.prelude_layers}, coda={self.coda_layers}, "
                f"total={self.n_layers}"
            )

    @property
    def recurrent_layers(self) -> int:
        """Number of layers in the recurrent block."""
        return self.n_layers - self.prelude_layers - self.coda_layers

    def new_param_count(self) -> int:
        """Count the new trainable parameters added by the wrapper."""
        # LTI injection: log_A (dim) + log_dt (1) + B (dim)
        lti = 2 * self.dim + 1
        # ACT halting: linear(dim, 1) + bias
        act = self.dim + 1
        # LoRA adapter: down (dim*rank) + B (rank*dim) + scale (max_loops*rank)
        lora = self.dim * self.lora_rank + self.lora_rank * self.dim + self.max_loop_iters * self.lora_rank
        # RMSNorm (if used for input mixing)
        norm = self.dim
        return lti + act + lora + norm

    def new_param_percentage(self, base_param_count: int) -> float:
        """New params as a percentage of the base model's total params."""
        return (self.new_param_count() / base_param_count) * 100.0