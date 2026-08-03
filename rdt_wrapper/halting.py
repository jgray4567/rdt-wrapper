"""Adaptive Computation Time (ACT) halting mechanism.

Learns a per-position halting probability at each loop iteration.
Positions where the hidden state has converged stop accumulating updates,
while positions still being refined continue. Easy tokens halt early,
hard tokens get more computation — all within the same batch.
"""

import torch
import torch.nn as nn


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting head.

    Args:
        dim: Hidden state dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Predict per-position halting probability.

        Args:
            h: Hidden state, shape (B, T, dim).

        Returns:
            Halting probability, shape (B, T), values in (0, 1).
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)