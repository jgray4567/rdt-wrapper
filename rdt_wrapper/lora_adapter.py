"""Depth-wise LoRA adapter for per-loop specialization.

Shared low-rank down/up projections with a per-loop scale vector,
allowing each loop iteration to perform a slightly different transformation
without duplicating the full parameter set.
"""

import torch
import torch.nn as nn


class DepthLoRAAdapter(nn.Module):
    """Depth-wise LoRA adaptation for the recurrent block.

    delta(x, t) = (down(x) * scale[t]) @ B

    The down projection and B matrix are shared across all loops.
    A per-loop scale vector shifts the effective transformation at each depth.

    Args:
        dim: Model hidden dimension.
        rank: Low-rank bottleneck dimension.
        max_loops: Maximum number of loop iterations (determines embedding table size).
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)
        # Initialize with small random values so each loop is slightly different from the start
        nn.init.normal_(self.scale.weight, mean=1.0, std=0.1)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """Compute the LoRA delta for the given loop iteration.

        Args:
            x: Input tensor, shape (B, T, dim).
            loop_t: Current loop index.

        Returns:
            Delta tensor, shape (B, T, dim), to be added to block output.
        """
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)