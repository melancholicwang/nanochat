"""
Generate text samples using trained Diffusion Language Model.

Usage:
python -m scripts.diffusion_sample --checkpoint checkpoints/diffusion_step_005000.pt --num_samples 5 --seq_len 128

This demonstrates the iterative denoising process of diffusion models.
"""

import os
import argparse
import torch

from nanochat.diffusion_roberta import DiffusionRoBERTa, DiffusionRoBERTaConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, print_banner

print_banner()

# Parse arguments
parser = argparse.ArgumentParser(description="Generate samples from diffusion model")
parser.add_argument("--checkpoint", type=str, default="checkpoints/diffusion_latest.pt", help="Path to checkpoint")
parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
parser.add_argument("--seq_len", type=int, default=64, help="Sequence length to generate")
parser.add_argument("--num_steps", type=int, default=100, help="Number of denoising steps (default: 100, max: model's diffusion_steps)")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu/mps)")
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
    print0("Please train a model first using: python -m scripts.diffusion_train")
    exit(1)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

# Extract model config from checkpoint
if "config" in checkpoint:
    model_config = checkpoint["config"]
else:
    # Infer config from model state dict (fallback)
    print0("Warning: Config not found in checkpoint, using default config")
    model_config = DiffusionRoBERTaConfig(
        sequence_len=args.seq_len * 2,
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

# Load weights
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print0(f"Model loaded successfully!")
print0(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Generate samples
print0("\n" + "="*80)
print0("Generating samples using diffusion process...")
print0(f"Configuration: seq_len={args.seq_len}, num_steps={args.num_steps}, temperature={args.temperature}")
print0("="*80 + "\n")

with torch.inference_mode():
    for i in range(args.num_samples):
        print0(f"Sample {i+1}/{args.num_samples}:")
        print0("-" * 60)

        # Generate using diffusion
        tokens = model.sample(
            batch_size=1,
            seq_len=args.seq_len,
            temperature=args.temperature,
            num_steps=args.num_steps,
        )

        # Decode tokens
        text = tokenizer.decode(tokens[0].tolist())
        print0(text)
        print0("-" * 60 + "\n")

print0("="*80)
print0("Generation complete!")
print0("="*80)
