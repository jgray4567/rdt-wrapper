"""
RDT Wrapper — Recurrent-Depth Adaptation for Pretrained Transformers.

A library that wraps any HuggingFace decoder-only transformer with recurrent-depth
looping and adaptive computation time (ACT) halting. Pretrained weights stay frozen;
only injection, halting, and LoRA adapter parameters are trained.
"""

from rdt_wrapper.config import RDTConfig
from rdt_wrapper.model import RDTModel
from rdt_wrapper.injection import LTIInjection, loop_index_embedding
from rdt_wrapper.halting import ACTHalting
from rdt_wrapper.lora_adapter import DepthLoRAAdapter

__version__ = "0.1.0"
__all__ = [
    "RDTConfig",
    "RDTModel",
    "LTIInjection",
    "ACTHalting",
    "DepthLoRAAdapter",
    "loop_index_embedding",
]