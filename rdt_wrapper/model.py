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
        # Cast wrapper params to match base model dtype (e.g. bfloat16)
        base_dtype = next(base_model.parameters()).dtype
        model.recurrent = model.recurrent.to(base_dtype)
        return model

    def _load_base_model(self, base_model: nn.Module):
        """Extract components from a HuggingFace model.

        Uses a series of fallback attribute lookups to handle different
        architectures (Llama, Gemma, Qwen, Mistral, Phi, etc.) without
        hardcoding each one.

        Args:
            base_model: The pretrained model to extract from.
        """
        # --- Find the text model ---
        # Most models: text layers are at model.model.layers
        # Gemma 4 (unified/multimodal): text layers are at model.model.language_model.layers
        # We try both paths to stay model-agnostic.
        text_model = None
        for path in ["model.language_model", "language_model", "model.text_model", "text_model"]:
            try:
                text_model = self._get_attr(base_model, [path])
                break
            except AttributeError:
                continue
        if text_model is None:
            # Standard path — text model IS model.model (or model itself)
            try:
                text_model = self._get_attr(base_model, ["model", "model.model"])
            except AttributeError:
                text_model = base_model  # last resort: model itself

        # --- Embedding ---
        self.embed = self._get_attr(
            text_model,
            ["embed_tokens", "model.embed_tokens"],
        )

        # --- Transformer blocks ---
        # Use try/except instead of default= to avoid eager evaluation
        try:
            all_blocks = self._get_attr(text_model, ["layers", "model.layers", "blocks", "h"])
        except AttributeError:
            all_blocks = self._get_attr(
                base_model,
                ["model.layers", "layers", "transformer.h", "h", "model.blocks", "blocks"],
            )
        all_blocks = list(all_blocks)

        # --- Final norm ---
        try:
            self.final_norm = self._get_attr(text_model, ["norm", "final_layernorm", "final_norm", "ln_f"])
        except AttributeError:
            try:
                self.final_norm = self._get_attr(
                    base_model,
                    ["model.norm", "norm", "model.model.norm",
                     "transformer.ln_f", "ln_f", "model.final_layernorm",
                     "final_layernorm", "model.final_norm"],
                )
            except AttributeError:
                self.final_norm = nn.Identity()

        # --- LM head ---
        try:
            self.lm_head = self._get_attr(base_model, ["lm_head", "model.lm_head"])
        except AttributeError:
            self.lm_head = self.embed

        # --- Rotary embeddings (transformers v5+) ---
        try:
            self._rotary_emb = self._get_attr(text_model, ["rotary_emb"])
        except AttributeError:
            try:
                self._rotary_emb = self._get_attr(
                    base_model, ["model.rotary_emb", "rotary_emb", "model.model.rotary_emb"]
                )
            except AttributeError:
                self._rotary_emb = None

        # --- Detect block forward signature ---
        self._block_forward_style = self._detect_block_forward(all_blocks[0])

        # --- Detect extra block forward args (e.g. shared_kv_states for Gemma 4) ---
        self._block_extra_args = self._detect_block_extra_args(all_blocks[0])

        # --- Detect Gemma 4 hybrid attention (layer_types) ---
        # Gemma 4 uses different rotary embeddings per layer type
        # (sliding_attention vs full_attention). The rotary emb takes a
        # `layer_type` argument and the model computes a dict of position
        # embeddings keyed by layer_type.
        self._has_layer_types = False
        self._layer_types: List[str] = []
        self._unique_layer_types: List[str] = []
        base_config = getattr(base_model, "config", None)
        if base_config is not None:
            # Gemma 4 nests text config under text_config
            text_config = getattr(base_config, "text_config", base_config)
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types is not None:
                self._has_layer_types = True
                self._layer_types = list(layer_types)
                self._unique_layer_types = list(set(layer_types))

        # Also check if rotary emb forward takes layer_type arg
        if self._rotary_emb is not None and not self._has_layer_types:
            try:
                rot_sig = inspect.signature(self._rotary_emb.forward)
                if "layer_type" in rot_sig.parameters:
                    # Rotary emb supports layer_type but config didn't specify
                    # Default to a single layer type
                    self._has_layer_types = True
                    self._layer_types = ["full_attention"] * len(all_blocks)
                    self._unique_layer_types = ["full_attention"]
            except (ValueError, TypeError):
                pass

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

        # Store per-block layer types for position embedding lookup
        if self._has_layer_types:
            self._prelude_layer_types = self._layer_types[:n_prelude]
            self._recurrent_layer_types = self._layer_types[n_prelude:n_prelude + n_recurrent]
            self._coda_layer_types = self._layer_types[n_prelude + n_recurrent:]
        else:
            self._prelude_layer_types = [None] * n_prelude
            self._recurrent_layer_types = [None] * n_recurrent
            self._coda_layer_types = [None] * max(n_coda, 0)

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

    @staticmethod
    def _detect_block_extra_args(block: nn.Module) -> List[str]:
        """Detect extra (non-standard) args in block forward beyond the usual set.

        Standard args: hidden_states, attention_mask, position_ids,
        position_embeddings, past_key_values, past_key_value, use_cache, kwargs.

        Returns a list of extra arg names that need to be passed (as None).
        e.g. ['shared_kv_states'] for Gemma 4.
        """
        standard = {"self", "hidden_states", "attention_mask", "position_ids",
                    "position_embeddings", "past_key_values", "past_key_value",
                    "use_cache", "kwargs"}
        try:
            sig = inspect.signature(block.forward)
            extra = [name for name in sig.parameters if name not in standard]
            return extra
        except (ValueError, TypeError):
            return []

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
        """Compute position embeddings if the model needs them.

        Returns:
            For standard models: (cos, sin) tuple, or None.
            For Gemma 4 (layer_types): dict[str, (cos, sin)] keyed by layer_type.
        """
        if self._rotary_emb is None:
            return None
        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        
        if self._has_layer_types:
            # Gemma 4: compute position embeddings per layer type
            pos_emb_dict = {}
            for layer_type in self._unique_layer_types:
                pos_emb_dict[layer_type] = self._rotary_emb(h, position_ids, layer_type)
            return pos_emb_dict
        else:
            # Standard: single (cos, sin) tuple
            return self._rotary_emb(h, position_ids=position_ids)

    def _run_blocks(
        self,
        blocks,
        h: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings=None,
        block_layer_types: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Run a list of transformer blocks with the correct forward signature.

        Handles three generations of HuggingFace block interfaces:
          - v5: block(h, position_embeddings=(cos,sin), attention_mask=..., ...)
          - v4: block(h, position_ids=..., attention_mask=..., ...)
          - legacy: block(h, attention_mask=..., ...)

        For Gemma 4 hybrid attention, position_embeddings is a dict keyed by
        layer_type, and each block gets its own layer_type's embeddings.

        Args:
            blocks: List of transformer block modules.
            h: Hidden states, shape (B, T, dim).
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.
            position_embeddings: (cos, sin) tuple OR dict[str, (cos, sin)] for Gemma 4.
            block_layer_types: Per-block layer types for Gemma 4 dict lookup.

        Returns:
            Updated hidden states, shape (B, T, dim).
        """
        for i, block in enumerate(blocks):
            # Determine position embeddings for this block
            if position_embeddings is not None and isinstance(position_embeddings, dict):
                # Gemma 4: dict keyed by layer_type
                lt = block_layer_types[i] if block_layer_types else None
                block_pos_emb = position_embeddings.get(lt) if lt else None
            else:
                block_pos_emb = position_embeddings

            # Build kwargs for extra args (e.g. shared_kv_states for Gemma 4)
            extra_kwargs = {arg: None for arg in self._block_extra_args}

            if self._block_forward_style == "v5_position_embeddings":
                outputs = block(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=block_pos_emb,
                    use_cache=False,
                    past_key_value=None,
                    **extra_kwargs,
                )
            elif self._block_forward_style == "v4_position_ids":
                outputs = block(
                    h,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    **extra_kwargs,
                )
            else:
                # legacy or unknown
                try:
                    outputs = block(h, attention_mask=attention_mask, use_cache=False, **extra_kwargs)
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
            block_layer_types=self._prelude_layer_types if self._has_layer_types else None,
        )

        # 4. Recurrent loop
        e = h  # encoded input, frozen for injection each loop
        block_forward = lambda combined: self._run_blocks(
            self.recurrent_blocks, combined,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            block_layer_types=self._recurrent_layer_types if self._has_layer_types else None,
        )
        h = self.recurrent(h, e, block_forward, n_loops=n_loops)

        # 5. Coda (run once, frozen)
        h = self._run_blocks(
            self.coda_blocks, h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            block_layer_types=self._coda_layer_types if self._has_layer_types else None,
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