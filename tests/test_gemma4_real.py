#!/usr/bin/env python3
"""Test RDT wrapper against real Gemma 4 12B.

Run manually: python tests/test_gemma4_real.py
Not collected by pytest (guarded by __main__ check).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdt_wrapper.config import RDTConfig
from rdt_wrapper.model import RDTModel


def main():
    print("Loading Gemma 4 12B (bfloat16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", dtype=torch.bfloat16)
    print(f"Loaded. Class: {type(base_model).__name__}", flush=True)

    tc = base_model.config.text_config
    dim = tc.hidden_size
    n_layers = tc.num_hidden_layers
    layer_types = getattr(tc, "layer_types", None)
    print(f"dim={dim}, layers={n_layers}", flush=True)
    if layer_types:
        print(f"layer_types: {len(layer_types)} layers, unique: {set(layer_types)}", flush=True)

    # Wrap with RDT
    config = RDTConfig(
        dim=dim,
        n_layers=n_layers,
        prelude_fraction=0.5,
        max_loop_iters=8,
        lora_rank=16,
        act_threshold=0.99,
    )
    print("Wrapping with RDT...", flush=True)
    model = RDTModel.from_pretrained(base_model, config)
    model._base_model_name = "google/gemma-4-12B-it"
    print(model.info(), flush=True)

    # Verify layer_types detection
    print(f"has_layer_types: {model._has_layer_types}", flush=True)
    if model._has_layer_types:
        print(f"prelude layer types: {model._prelude_layer_types[:5]}...", flush=True)
        print(f"recurrent layer types: {model._recurrent_layer_types[:5]}...", flush=True)

    # Forward pass test
    print("Testing forward pass (4 loops)...", flush=True)
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids, n_loops=4)
    print(f"Forward pass OK. Logits shape: {out['logits'].shape}", flush=True)

    # Generate test
    print("Testing generate (20 tokens, 4 loops)...", flush=True)
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20, n_loops=4, temperature=0.1)
    print("Generate OK.", flush=True)
    print(f"Output: {tokenizer.decode(generated[0], skip_special_tokens=True)}", flush=True)

    # Depth extrapolation test
    print("Testing depth extrapolation (16 loops)...", flush=True)
    with torch.no_grad():
        out16 = model(input_ids, n_loops=16)
    print(f"16 loops OK. Logits shape: {out16['logits'].shape}", flush=True)
    print(f"Finite: {torch.isfinite(out16['logits']).all().item()}", flush=True)

    print("\nAll tests passed! RDT wrapper works with Gemma 4 12B.", flush=True)


if __name__ == "__main__":
    main()