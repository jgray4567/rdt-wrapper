"""Recurrent loop — the core wrapper that loops frozen transformer blocks.

Runs the recurrent block for up to n_loops iterations with ACT early exit.
At each iteration:
1. Inject loop-index sinusoidal signal
2. Mix hidden state with encoded input
3. Run through frozen transformer blocks
4. Apply depth-wise LoRA delta
5. Stable state update via LTI injection
6. ACT halting — converged positions stop contributing
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable

from rdt_wrapper.injection import LTIInjection, loop_index_embedding
from rdt_wrapper.halting import ACTHalting
from rdt_wrapper.lora_adapter import DepthLoRAAdapter
from rdt_wrapper.norm import RMSNorm


class RecurrentLoop(nn.Module):
    """The recurrent loop that wraps frozen transformer blocks.

    Args:
        config: RDTConfig instance.
        block_forward: Callable that runs one forward pass through the
            frozen recurrent blocks. Set via ``set_block_forward`` after
            model construction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.injection = LTIInjection(config.dim)
        self.act = ACTHalting(config.dim)
        self.lora = DepthLoRAAdapter(config.dim, config.lora_rank, config.max_loop_iters)
        self.norm = RMSNorm(config.dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        block_forward: Callable,
        n_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h: Initial hidden state from prelude, shape (B, T, dim).
            e: Encoded input frozen for injection each step, shape (B, T, dim).
            block_forward: Callable(h_normed) -> transformer_out that runs
                the frozen recurrent blocks. Passed in by RDTModel.
            n_loops: Number of loop iterations. Defaults to config.max_loop_iters.

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim).
        """
        n_loops = n_loops or self.config.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            # 1. Inject loop-index signal
            h_loop = loop_index_embedding(h, t, self.config.loop_embedding_dim)

            # 2. Mix hidden state with encoded input
            combined = self.norm(h_loop + e)

            # 3. Run frozen transformer blocks
            trans_out = block_forward(combined)

            # 4. Apply depth-wise LoRA delta
            trans_out = trans_out + self.lora(trans_out, t)

            # 5. Stable state update
            h = self.injection(h, e, trans_out)

            # 6. ACT halting
            p = self.act(h)
            still_running = ~halted
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.config.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.config.act_threshold)

            if halted.all():
                break

        return h_out