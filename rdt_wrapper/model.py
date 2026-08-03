"""RDTModel — wraps any HuggingFace decoder-only transformer with recurrent depth.

The pretrained model is split into three parts:
  - Prelude: first N layers, run once (frozen)
  - Recurrent: remaining layers, looped T times (frozen weights, new params control looping)
  - Coda: final norm + LM head, run once (frozen)

Only the injection, halting, and LoRA adapter parameters are trainable.

Design goal: model-agnostic. Any HuggingFace AutoModelForCausalLM should work
without architecture-specific code. The block runner detects the correct
forward signature automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import inspect

from rdt_wrapper.config import RDTConfig
from rdt_wrapper.recurrent import RecurrentLoop
from rdt_wrapper.norm import RMSNorm


class RDTModel(nn.Module):
    """Wraps a pretrained transformer with recurrent-depth looping.

    Attributes:
        config: RDTConfig instance.
        embed: Token embedding layer (frozen).
        prelude_blocks: List of frozen transformer blocks run once.
        recurrent_blocks: List of frozen transformer blocks run in a loop.
        coda_blocks: List of frozen transformer blocks run after looping (optional).
        final_norm: Final normalization layer (frozen).
        lm_head: LM head / output projection (frozen).
        recurrent: The RecurrentLoop module with trainable new params.
    """

    def __init__(self, config: RDTConfig):
        super().__init__()
        self.config = config

        # These are set by from_pretrained or _load_base_model
        self.embed: Optional[nn.Module] = None
        self.prelude_blocks: nn.ModuleList = nn.ModuleList()
        self.recurrent_blocks: nn.ModuleList = nn.ModuleList()
        self.coda_blocks: nn.ModuleList = nn.ModuleList()
        self.final_norm: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

        # The trainable wrapper
        self.recurrent = RecurrentLoop(config)

        # Base model metadata
        self._base_model_name: Optional[str] = None
        self._base_param_count: int = 0
        self._rotary_emb: Optional[nn.Module] = None
        self._block_forward_style: str = "unknown"  # detected at load time

    @classmethod
    def from_pretrained(cls, base_model: nn.Module, config: RDTConfig) -> "RDTModel":
        """Wrap a HuggingFace pretrained model with recurrent depth.

        Args:
            base_model: A HuggingFace AutoModelForCausalLM instance.
            config: RDTConfig specifying the wrapper parameters.

        Returns:
            An RDTModel with the base model's weights loaded and frozen.

        Raises:
            ValueError: If the base model architecture is not recognized.
        """
        model = cls(config)
        model._load_base_model(base_model)
        model.freeze_base()
        model._base_param_count = sum(p.numel() for p in base_model.parameters())
        return model

    def _load_base_model(self, base_model: nn.Module):
        """Extract components from a HuggingFace model.

        Uses a series of fallback attribute lookups to handle different
        architectures (Llama, Gemma, Qwen, Mistral, Phi, etc.) without
        hardcoding each one.

        Args:
            base_model: The pretrained model to extract from.
        """
        # --- Embedding ---
        self.embed = self._get_attr(
            base_model,
            ["model.embed_tokens", "embed_tokens", "model.model.embed_tokens",
             "transformer.wte", "model.wte"],
        )

        # --- Transformer blocks ---
        all_blocks = self._get_attr(
            base_model,
            ["model.layers", "layers", "model.model.layers",
             "transformer.h", "h", "model.blocks", "blocks"],
        )
        all_blocks = list(all_blocks)

        # --- Final norm ---
        self.final_norm = self._get_attr(
            base_model,
            ["model.norm", "norm", "model.model.norm",
             "transformer.ln_f", "ln_f", "model.final_layernorm",
             "final_layernorm", "model.final_norm"],
            default=nn.Identity(),
        )

        # --- LM head ---
        # Try explicit lm_head; fall back to tied embeddings
        self.lm_head = self._get_attr(
            base_model,
            ["lm_head", "model.lm_head"],
            default=self.embed,
        )

        # --- Rotary embeddings (transformers v5+) ---
        # Newer transformers versions moved RoPE to the model level and require
        # position_embeddings to be passed explicitly to each block.
        self._rotary_emb = self._get_attr(
            base_model,
            ["model.rotary_emb", "rotary_emb", "model.model.rotary_emb"],
            default=None,
        )

        # --- Detect block forward signature ---
        self._block_forward_style = self._detect_block_forward(all_blocks[0])

        # --- Split blocks into prelude / recurrent / coda ---
        n_prelude = self.config.prelude_layers
        n_coda = self.config.coda_layers
        n_recurrent = len(all_blocks) - n_prelude - n_coda

        if n_recurrent <= 0:
            raise ValueError(
                f"Cannot split {len(all_blocks)} layers into "
                f"prelude={n_prelude}, coda={n_coda}. Need at least 1 recurrent layer."
            )

        self.prelude_blocks = nn.ModuleList(all_blocks[:n_prelude])
        self.recurrent_blocks = nn.ModuleList(all_blocks[n_prelude:n_prelude + n_recurrent])
        self.coda_blocks = nn.ModuleList(all_blocks[n_prelude + n_recurrent:])

    @staticmethod
    def _get_attr(obj, paths, default=None):
        """Try multiple attribute paths and return the first that exists."""
        for path in paths:
            current = obj
            for part in path.split("."):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = None
                    break
            if current is not None:
                return current
        if default is not None:
            return default
        raise AttributeError(f"None of the paths {paths} found on {type(obj).__name__}")

    @staticmethod
    def _detect_block_forward(block: nn.Module) -> str:
        """Detect the block's forward signature style.

        Inspects the forward method signature to determine how to call it.

        Returns one of:
            - 'v5_position_embeddings': block(h, position_embeddings=(cos,sin), ...)
            - 'v4_position_ids': block(h, position_ids=..., ...)
            - 'legacy': block(h, ...) with no position args
        """
        try:
            sig = inspect.signature(block.forward)
            params = sig.parameters
            if "position_embeddings" in params:
                return "v5_position_embeddings"
            elif "position_ids" in params:
                return "v4_position_ids"
            else:
                return "legacy"
        except (ValueError, TypeError):
            return "legacy"

    def freeze_base(self):
        """Freeze all pretrained model weights."""
        for module in [self.embed, self.final_norm, self.lm_head]:
            if module is not None and hasattr(module, 'parameters'):
                for p in module.parameters():
                    p.requires_grad = False

        for blocks in [self.prelude_blocks, self.recurrent_blocks, self.coda_blocks]:
            for block in blocks:
                for p in block.parameters():
                    p.requires_grad = False

        if self._rotary_emb is not None:
            for p in self._rotary_emb.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        """Return only the trainable (new) parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_parameters(self):
        """Return only the frozen (base model) parameters."""
        return [p for p in self.parameters() if not p.requires_grad]

    def _get_position_embeddings(self, h: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Compute position embeddings (cos, sin) if the model needs them.

        Returns:
            (cos, sin) tuple, or None if the model doesn't use explicit position embeddings.
        """
        if self._rotary_emb is None:
            return None
        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        # HuggingFace rotary emb returns (cos, sin) given hidden_states and position_ids
        return self._rotary_emb(h, position_ids=position_ids)

    def _run_blocks(
        self,
        blocks,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run a list of transformer blocks with the correct forward signature.

        Handles three generations of HuggingFace block interfaces:
          - v5: block(h, position_embeddings=(cos,sin), attention_mask=..., ...)
          - v4: block(h, position_ids=..., attention_mask=..., ...)
          - legacy: block(h, attention_mask=..., ...)

        Args:
            blocks: List of transformer block modules.
            h: Hidden states, shape (B, T, dim).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            position_embeddings: Optional (cos, sin) tuple for v5 models.

        Returns:
            Updated hidden states, shape (B, T, dim).
        """
        for block in blocks:
            if self._block_forward_style == "v5_position_embeddings":
                outputs = block(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    past_key_value=None,
                )
            elif self._block_forward_style == "v4_position_ids":
                outputs = block(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            else:
                # legacy or unknown — try with minimal args, then fallback
                try:
                    outputs = block(h, attention_mask=attention_mask, use_cache=False)
                except TypeError:
                    outputs = block(h)

            if isinstance(outputs, tuple):
                h = outputs[0]
            elif hasattr(outputs, 'last_hidden_state'):
                h = outputs.last_hidden_state
            else:
                h = outputs
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: Prelude → Recurrent Loop → Coda → LM Head.

        Args:
            input_ids: Token indices, shape (B, T).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            n_loops: Override max loop iterations. If None, uses config default.
            labels: Optional target token IDs for loss computation.

        Returns:
            Dict with 'logits' and optionally 'loss'.
        """
        # 1. Embed
        h = self.embed(input_ids)  # (B, T, dim)

        # 2. Compute position embeddings if needed (v5 models)
        position_embeddings = self._get_position_embeddings(h, position_ids)

        # 3. Prelude (run once, frozen)
        h = self._run_blocks(
            self.prelude_blocks, h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        # 4. Recurrent loop
        e = h  # encoded input, frozen for injection each loop
        block_forward = lambda combined: self._run_blocks(
            self.recurrent_blocks, combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        h = self.recurrent(h, e, block_forward, n_loops=n_loops)

        # 5. Coda (run once, frozen)
        h = self._run_blocks(
            self.coda_blocks, h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

        # 6. Final norm + LM head
        if self.final_norm is not None:
            h = self.final_norm(h)
        logits = self.lm_head(h)

        output = {"logits": logits}

        # 7. Loss (optional)
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive token generation with recurrent depth.

        Args:
            input_ids: Prompt token indices, shape (B, T).
            max_new_tokens: Number of tokens to generate.
            n_loops: Recurrent loop depth per decode step.
            temperature: Softmax temperature.
            top_k: Top-K sampling filter (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).

        Returns:
            Token indices, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            output = self.forward(input_ids, n_loops=n_loops)
            logits = output["logits"][:, -1, :] / temperature

            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def info(self) -> str:
        """Return a human-readable summary of the model."""
        new_params = self.config.new_param_count()
        pct = self.config.new_param_percentage(self._base_param_count) if self._base_param_count else 0
        return (
            f"RDTModel(base={self._base_model_name or 'unknown'}, "
            f"dim={self.config.dim}, "
            f"prelude={len(self.prelude_blocks)}, "
            f"recurrent={len(self.recurrent_blocks)}, "
            f"coda={len(self.coda_blocks)}, "
            f"max_loops={self.config.max_loop_iters}, "
            f"block_style={self._block_forward_style}, "
            f"new_params={new_params:,} ({pct:.4f}%), "
            f"base_params={self._base_param_count:,})"
        )