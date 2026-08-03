# RDT Wrapper — Recurrent-Depth Adaptation for Pretrained Transformers

A PyTorch library that wraps any HuggingFace transformer with recurrent-depth looping and adaptive computation time (ACT) halting. The pretrained weights stay frozen — only a tiny set of new parameters (~0.01% of model size) are trained to control looping, state injection, and halting.

## What It Does

- **Adaptive compute:** Easy tokens exit the loop early (2-4 iterations). Hard tokens get more (8-16+). Same model, variable depth.
- **Frozen weights:** Pretrained model weights are never touched. Only injection, halting, and LoRA adapter params are trained.
- **Depth extrapolation:** Train at N loops, run at N+k. Quality improves with more compute at inference.
- **Model-agnostic:** Works with any decoder-only transformer from HuggingFace (Gemma, Qwen, Llama, Mistral, etc.).

## Quick Start

```python
from rdt_wrapper import RDTConfig, RDTModel
from transformers import AutoModelForCausalLM

# Load any pretrained model
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12b-it")

# Wrap it with recurrent depth
config = RDTConfig(
    dim=3072,
    max_loop_iters=16,
    prelude_fraction=0.5,  # first half of layers = prelude, second half = recurrent
    lora_rank=16,
    act_threshold=0.99,
)
model = RDTModel.from_pretrained(base_model, config)

# Train only the new params (~0.01% of model)
model.freeze_base()
# ... train injection + halting + LoRA ...

# Inference with adaptive depth
output = model.generate(input_ids, max_new_tokens=128, n_loops=8)

# Depth extrapolation — run harder than trained
output = model.generate(input_ids, max_new_tokens=128, n_loops=32)
```

## Installation

```bash
pip install -e .
```

## License

MIT

## Citation

If you use this in research, please cite:

```bibtex
@software{rdt_wrapper,
  title={RDT Wrapper: Recurrent-Depth Adaptation for Pretrained Transformers},
  author={Grayson, Jon},
  year={2026},
  url={https://github.com/jgray4567/rdt-wrapper}
}
```