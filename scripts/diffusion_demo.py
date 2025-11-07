"""
Quick demo of Diffusion Language Model - shows the complete workflow.

This script demonstrates:
1. Model initialization
2. Forward diffusion (adding noise)
3. Training step
4. Reverse diffusion (sampling/generation)

Usage:
python -m scripts.diffusion_demo
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from nanochat.diffusion_roberta import DiffusionRoBERTa, DiffusionRoBERTaConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import print0, print_banner

print_banner()

print0("="*80)
print0("Diffusion Language Model Demo")
print0("="*80)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print0(f"\nUsing device: {device}")

# Initialize tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Create a small model for demonstration
print0("\n" + "-"*80)
print0("Step 1: Initialize Model")
print0("-"*80)

config = DiffusionRoBERTaConfig(
    sequence_len=128,
    vocab_size=vocab_size,
    n_layer=4,  # Small model for demo
    n_head=4,
    n_embd=256,
    diffusion_steps=100,  # Fewer steps for demo
    noise_schedule="cosine"
)

model = DiffusionRoBERTa(config)
model.to(device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print0(f"Model initialized with {num_params:,} parameters")
print0(f"Architecture: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads")
print0(f"Diffusion steps: {config.diffusion_steps}")

# Demonstrate forward diffusion process
print0("\n" + "-"*80)
print0("Step 2: Forward Diffusion (Adding Noise)")
print0("-"*80)

# Create sample text
sample_text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(sample_text)
print0(f"\nOriginal text: '{sample_text}'")
print0(f"Tokens: {tokens[:20]}... ({len(tokens)} total)")

# Pad to sequence length
seq_len = 32
if len(tokens) < seq_len:
    tokens = tokens + [0] * (seq_len - len(tokens))
else:
    tokens = tokens[:seq_len]

# Convert to tensor
x = torch.tensor([tokens], device=device)  # (1, seq_len)

# Get embeddings
with torch.inference_mode():
    embeddings = model.token_embedding(x)  # (1, seq_len, n_embd)
    print0(f"\nEmbedding shape: {embeddings.shape}")
    print0(f"Embedding norm: {embeddings.norm().item():.4f}")

    # Show noise at different timesteps
    print0("\nAdding noise at different timesteps:")
    timesteps_to_show = [0, 25, 50, 75, 99]
    noise_norms = []

    for t in timesteps_to_show:
        t_tensor = torch.tensor([t], device=device)
        noisy_emb = model.q_sample(embeddings, t_tensor)
        noise_norm = (noisy_emb - embeddings).norm().item()
        noise_norms.append(noise_norm)
        signal_to_noise = embeddings.norm().item() / (noise_norm + 1e-8)
        print0(f"  t={t:3d}: noise_norm={noise_norm:7.4f}, SNR={signal_to_noise:.4f}")

# Training step demonstration
print0("\n" + "-"*80)
print0("Step 3: Training Step (Denoising Objective)")
print0("-"*80)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print0("\nPerforming a few training steps on random data...")
losses = []

for step in range(5):
    # Random batch
    batch = torch.randint(0, vocab_size, (2, seq_len), device=device)

    # Compute loss
    loss = model.compute_loss(batch)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print0(f"  Step {step+1}: loss = {loss.item():.4f}")

print0("\nNote: Loss is high because model is untrained. After proper training,")
print0("loss should decrease to ~0.01-0.05 range.")

# Generation demonstration
print0("\n" + "-"*80)
print0("Step 4: Reverse Diffusion (Generation)")
print0("-"*80)

model.eval()

print0("\nGenerating samples from random noise...")
print0("(Note: Output will be nonsensical since model is untrained)")

with torch.inference_mode():
    # Generate with fewer steps for demo
    num_steps = 20
    sample_len = 32

    print0(f"\nConfiguration: seq_len={sample_len}, num_steps={num_steps}")
    print0("Generating 3 samples:\n")

    for i in range(3):
        tokens = model.sample(
            batch_size=1,
            seq_len=sample_len,
            temperature=1.0,
            num_steps=num_steps
        )

        text = tokenizer.decode(tokens[0].tolist())
        print0(f"Sample {i+1}:")
        print0(f"  {text[:100]}...")  # Show first 100 chars
        print0()

# Visualize noise schedule
print0("-"*80)
print0("Step 5: Visualizing Noise Schedule")
print0("-"*80)

timesteps = np.arange(config.diffusion_steps)
sqrt_alphas = model.sqrt_alphas_cumprod.cpu().numpy()
sqrt_one_minus_alphas = model.sqrt_one_minus_alphas_cumprod.cpu().numpy()

print0(f"\nCosine noise schedule with {config.diffusion_steps} steps:")
print0(f"  t=0   (clean):     alpha_bar = {sqrt_alphas[0]**2:.6f}")
print0(f"  t=50  (mid):       alpha_bar = {sqrt_alphas[50]**2:.6f}")
print0(f"  t=99  (noisy):     alpha_bar = {sqrt_alphas[99]**2:.6f}")

try:
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, sqrt_alphas**2, label='α̅_t (signal weight)', linewidth=2)
    plt.plot(timesteps, sqrt_one_minus_alphas**2, label='1 - α̅_t (noise weight)', linewidth=2)
    plt.xlabel('Timestep t')
    plt.ylabel('Weight')
    plt.title('Cosine Noise Schedule for Diffusion Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('diffusion_noise_schedule.png', dpi=150, bbox_inches='tight')
    print0("\nNoise schedule plot saved to: diffusion_noise_schedule.png")
except Exception as e:
    print0(f"\nCould not save plot: {e}")

# Summary
print0("\n" + "="*80)
print0("Demo Complete!")
print0("="*80)

print0("""
Summary:
--------
✓ Initialized a small diffusion model (4 layers, 256 dim)
✓ Demonstrated forward diffusion (adding noise to clean text)
✓ Showed training step with denoising objective
✓ Generated samples using reverse diffusion (iterative denoising)
✓ Visualized the cosine noise schedule

Next Steps:
-----------
1. Train a full model:
   python -m scripts.diffusion_train --depth=8 --num_iterations=5000

2. Generate samples:
   python -m scripts.diffusion_sample --checkpoint checkpoints/diffusion_latest.pt

3. Evaluate:
   python -m scripts.diffusion_eval --checkpoint checkpoints/diffusion_latest.pt

For more information, see DIFFUSION_README.md
""")

print0("="*80)
