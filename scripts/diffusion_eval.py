"""
Evaluate Diffusion Language Model on validation set and compute metrics.

Usage:
python -m scripts.diffusion_eval --checkpoint checkpoints/diffusion_step_005000.pt
"""

import os
import argparse
import torch
import time

from nanochat.diffusion_roberta import DiffusionRoBERTa, DiffusionRoBERTaConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, print_banner

print_banner()

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate diffusion model")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--eval_tokens", type=int, default=1048576, help="Number of tokens to evaluate on")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
args = parser.parse_args()

# Setup
device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
print0(f"Using device: {device}")

# Load tokenizer
tokenizer = get_tokenizer()
print0(f"Vocab size: {tokenizer.get_vocab_size()}")

# Load checkpoint
print0(f"\nLoading checkpoint from {args.checkpoint}...")
if not os.path.exists(args.checkpoint):
    print0(f"Error: Checkpoint not found at {args.checkpoint}")
    exit(1)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

# Get model config
if "config" in checkpoint:
    model_config = checkpoint["config"]
else:
    print0("Warning: Config not found in checkpoint, using default")
    model_config = DiffusionRoBERTaConfig(
        sequence_len=args.seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=12,
        n_head=12,
        n_embd=768,
        diffusion_steps=1000,
    )

# Initialize model
print0("\nInitializing model...")
model = DiffusionRoBERTa(model_config)
model.to(device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()
num_params = sum(p.numel() for p in model.parameters())
print0(f"Model loaded! Parameters: {num_params:,}")

# Create validation dataloader
print0("\nLoading validation data...")
val_loader = tokenizing_distributed_data_loader(
    args.batch_size,
    args.seq_len,
    split="val",
    device=device
)

# Evaluation
print0("\n" + "="*80)
print0("Evaluating model on validation set")
print0("="*80 + "\n")

num_batches = max(1, args.eval_tokens // (args.batch_size * args.seq_len))
print0(f"Evaluating on {num_batches} batches ({args.eval_tokens:,} tokens)")

total_loss = 0.0
total_batches = 0
start_time = time.time()

with torch.inference_mode():
    for i in range(num_batches):
        x, y = next(val_loader)

        # Compute loss
        loss = model.compute_loss(x)
        total_loss += loss.item()
        total_batches += 1

        if (i + 1) % 10 == 0:
            avg_loss = total_loss / total_batches
            print0(f"Batch {i+1}/{num_batches} | avg loss: {avg_loss:.4f}")

end_time = time.time()
elapsed = end_time - start_time

# Compute final metrics
avg_loss = total_loss / total_batches
tokens_evaluated = num_batches * args.batch_size * args.seq_len
tokens_per_sec = tokens_evaluated / elapsed

# Results
print0("\n" + "="*80)
print0("Evaluation Results")
print0("="*80)
print0(f"Average diffusion loss: {avg_loss:.4f}")
print0(f"Tokens evaluated: {tokens_evaluated:,}")
print0(f"Time elapsed: {elapsed:.2f}s")
print0(f"Throughput: {tokens_per_sec:.0f} tokens/sec")
print0("="*80)

# Generate a few sample texts to demonstrate quality
print0("\n" + "="*80)
print0("Sample Generations")
print0("="*80 + "\n")

for i in range(3):
    print0(f"Sample {i+1}:")
    print0("-" * 60)
    tokens = model.sample(batch_size=1, seq_len=64, temperature=1.0, num_steps=50)
    text = tokenizer.decode(tokens[0].tolist())
    print0(text)
    print0("-" * 60 + "\n")

print0("Evaluation complete!")
