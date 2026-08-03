"""RMSNorm — root mean square layer normalization (no bias, no mean subtraction)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm with learned per-channel rescaling.

    Args:
        dim: Feature dimension to normalize over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight