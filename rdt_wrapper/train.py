"""Training script for RDT wrapper — trains only the new params.

Freezes all pretrained weights and trains only:
  - LTI injection (A, B matrices)
  - ACT halting head
  - Depth-wise LoRA adapter

Usage:
    python -m rdt_wrapper.train --model google/gemma-4-12b-it --data data/combined.jsonl
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List
from pathlib import Path

from rdt_wrapper.config import RDTConfig
from rdt_wrapper.model import RDTModel


class JSONLDataset(Dataset):
    """Simple JSONL dataset for text fine-tuning.

    Each line should have 'input' and 'output' fields (or 'text').
    """

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if "input" in sample and "output" in sample:
            # Instruction format: input -> output
            text = f"Input: {sample['input']}\nOutput: {sample['output']}"
        elif "text" in sample:
            text = sample["text"]
        else:
            text = json.dumps(sample)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels = input_ids (causal LM), but mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train(
    model: RDTModel,
    train_dataloader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    warmup_steps: int = 100,
    device: str = "auto",
    log_every: int = 10,
    save_dir: Optional[str] = None,
):
    """Train only the new RDT wrapper parameters.

    Args:
        model: RDTModel with frozen base weights.
        train_dataloader: Training data loader.
        epochs: Number of training epochs.
        lr: Learning rate for new params (higher than typical — these are small).
        warmup_steps: Linear warmup steps.
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps').
        log_every: Log every N steps.
        save_dir: Directory to save checkpoints.
    """
    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    model.train()

    # Only optimize trainable params (base is frozen)
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    # Linear warmup with cosine decay
    total_steps = epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
        else 0.5 * (1.0 + torch.cos(torch.tensor((step - warmup_steps) / max(1, total_steps - warmup_steps) * 3.14159)).item()),
    )

    print(f"Training {sum(p.numel() for p in trainable):,} trainable params "
          f"({len(trainable)} tensors) on {device}")
    print(f"Total steps: {total_steps} ({epochs} epochs × {len(train_dataloader)} batches)")

    step = 0
    for epoch in range(epochs):
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % log_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step}/{total_steps} | "
                      f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint per epoch
        if save_dir:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "step": step,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items() if "recurrent" in k},
                "optimizer_state": optimizer.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train RDT wrapper new params")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL training data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-loops", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Auto-detect dimensions
    if hasattr(base_model, 'config'):
        dim = getattr(base_model.config, 'hidden_size', None) or getattr(base_model.config, 'dim', None)
        n_layers = getattr(base_model.config, 'num_hidden_layers', None) or getattr(base_model.config, 'n_layers', None)
        if dim and n_layers:
            print(f"Detected: dim={dim}, n_layers={n_layers}")
        else:
            raise ValueError(f"Could not detect dim/n_layers from {base_model.config}")
    else:
        raise ValueError("Base model has no config attribute")

    config = RDTConfig(
        dim=dim,
        n_layers=n_layers,
        max_loop_iters=args.max_loops,
        lora_rank=args.lora_rank,
    )

    print(f"Wrapping with RDT: {config}")
    model = RDTModel.from_pretrained(base_model, config)
    model._base_model_name = args.model
    print(model.info())

    dataset = JSONLDataset(args.data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(
        model=model,
        train_dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()