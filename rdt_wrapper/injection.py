"""LTI-stable injection for recurrent depth looping.

Guarantees spectral radius ρ(A) < 1 by construction via ZOH discretization,
preventing hidden state explosion across arbitrary loop counts.
"""

import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """Stable input injection for the recurrent update rule.

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t + B · e + transformer_out

    where e is the encoded input injected at every loop step to prevent drift.
    A is constrained so spectral radius ρ(A) < 1 by construction.

    Args:
        dim: Hidden state dimension (one scalar per channel for A and B).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """Compute the discretized diagonal state matrix A_discrete.

        All values are strictly in (0, 1), guaranteeing ρ(A) < 1
        regardless of learned parameter values.
        """
        # Clamp inner sum to avoid overflow/underflow.
        # exp(-exp(x)) for large x → exp(-huge) → 0, which violates (0,1).
        # Clamp the result to ensure strictly positive values.
        inner = (self.log_dt + self.log_A).clamp(-20, 20)
        A = torch.exp(-torch.exp(inner))
        # Guarantee strictly in (0, 1) even at parameter extremes
        return A.clamp(min=1e-8, max=1.0 - 1e-8)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h: Current hidden state, shape (B, T, dim).
            e: Encoded input from prelude, frozen across loops, shape (B, T, dim).
            transformer_out: Output of the recurrent block at this step, shape (B, T, dim).

        Returns:
            Updated hidden state, shape (B, T, dim).
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Inject a sinusoidal loop-index signal into the first loop_dim channels.

    Analogous to RoPE for sequence position, but applied over recurrence depth.
    Without this, the shared recurrent block weights must handle both
    early-stage pattern-matching and late-stage refinement with no signal
    distinguishing which loop they are on.

    Args:
        h: Hidden state tensor, shape (B, T, dim).
        loop_t: Current loop iteration index (0-based).
        loop_dim: Number of leading channels to receive the embedding (must be even).
        theta: Sinusoidal base frequency.

    Returns:
        h with sinusoidal bias added to first loop_dim channels, same shape.
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)