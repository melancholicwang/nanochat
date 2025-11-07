# Diffusion Language Model in nanochat

This is an implementation of a **Diffusion Language Model** using **RoBERTa architecture** in the nanochat framework. Unlike autoregressive models (like GPT) that generate text token-by-token from left to right, diffusion models generate text through an iterative denoising process.

## Overview

### What is a Diffusion Language Model?

Diffusion models work by:
1. **Forward process (training)**: Gradually adding noise to clean text embeddings
2. **Reverse process (generation)**: Starting from random noise and iteratively denoising to generate coherent text

This approach offers several advantages:
- **Bidirectional context**: Can attend to both past and future tokens
- **Controllability**: Easier to condition on specific attributes
- **Parallel generation**: Potential for faster generation than autoregressive models

### Key Differences from GPT

| Aspect | GPT (Autoregressive) | Diffusion Model |
|--------|---------------------|-----------------|
| Architecture | Causal (decoder-only) | Bidirectional (encoder-like) |
| Training | Next token prediction | Denoising objective |
| Generation | Sequential (one token at a time) | Iterative refinement (all tokens together) |
| Context | Left-to-right only | Full bidirectional |

## Architecture

The implementation uses a **RoBERTa-style bidirectional transformer**:

- **Bidirectional Self-Attention**: No causal masking, can attend to entire sequence
- **Rotary Position Embeddings**: For positional information
- **Time Embeddings**: Encode the diffusion timestep
- **Cosine Noise Schedule**: For smooth diffusion process
- **Embedding-space Diffusion**: Operates on continuous embeddings rather than discrete tokens

### Model Configuration

```python
@dataclass
class DiffusionRoBERTaConfig:
    sequence_len: int = 1024        # Maximum sequence length
    vocab_size: int = 50304         # Vocabulary size
    n_layer: int = 12               # Number of transformer layers
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Embedding dimension
    diffusion_steps: int = 1000     # Number of diffusion timesteps
    noise_schedule: str = "cosine"  # Noise schedule type
```

## Usage

### 1. Training

Train a diffusion model from scratch:

```bash
# Single GPU
python -m scripts.diffusion_train --num_iterations 5000 --depth 12

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 -m scripts.diffusion_train --num_iterations 10000 --depth 12

# CPU/MPS (for testing)
python -m scripts.diffusion_train --depth=4 --max_seq_len=256 --device_batch_size=1 --num_iterations=100
```

**Key training parameters:**
- `--depth`: Model depth (number of layers)
- `--max_seq_len`: Maximum sequence length
- `--diffusion_steps`: Number of diffusion timesteps (default: 1000)
- `--noise_schedule`: Noise schedule type ("cosine" or "linear")
- `--num_iterations`: Number of training iterations
- `--device_batch_size`: Batch size per device
- `--total_batch_size`: Total batch size across all devices

### 2. Generating Samples

Generate text using a trained model:

```bash
python -m scripts.diffusion_sample \
    --checkpoint checkpoints/diffusion_step_005000.pt \
    --num_samples 5 \
    --seq_len 128 \
    --num_steps 100 \
    --temperature 1.0
```

**Parameters:**
- `--checkpoint`: Path to trained model checkpoint
- `--num_samples`: Number of samples to generate
- `--seq_len`: Length of generated sequences
- `--num_steps`: Number of denoising steps (fewer = faster but lower quality)
- `--temperature`: Sampling temperature (higher = more random)

### 3. Evaluation

Evaluate a trained model on validation data:

```bash
python -m scripts.diffusion_eval \
    --checkpoint checkpoints/diffusion_step_005000.pt \
    --eval_tokens 1048576 \
    --batch_size 8
```

## How it Works

### Forward Diffusion (Training)

1. Take clean token embeddings **x₀**
2. Sample a random timestep **t** ∈ [0, T]
3. Add noise according to the schedule:
   ```
   x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
   ```
   where ε ~ N(0, I) is Gaussian noise
4. Train the model to predict the noise **ε** from **x_t** and **t**

### Reverse Diffusion (Generation)

1. Start from random noise **x_T** ~ N(0, I)
2. For t = T, T-1, ..., 1:
   - Input **x_t** and timestep **t** to the model
   - Model predicts the noise **ε_θ(x_t, t)**
   - Compute denoised **x₀** prediction
   - Sample **x_{t-1}** using DDPM formula
3. Final **x₀** is converted to discrete tokens via argmax over vocabulary

### Noise Schedule

We use a **cosine schedule** (Nichol & Dhariwal, 2021) which provides:
- Smoother transitions between timesteps
- Better quality at both high and low noise levels
- More stable training

## Implementation Details

### Files Created

```
nanochat/
├── nanochat/
│   └── diffusion_roberta.py      # Model architecture
└── scripts/
    ├── diffusion_train.py         # Training script
    ├── diffusion_sample.py        # Sampling/generation script
    └── diffusion_eval.py          # Evaluation script
```

### Key Features

1. **Embedding-space Diffusion**: Operates in continuous embedding space, avoiding issues with discrete tokens
2. **Efficient Sampling**: Configurable number of denoising steps (can use fewer steps for faster generation)
3. **Bidirectional Architecture**: RoBERTa-style transformer with full attention
4. **Time Conditioning**: Sinusoidal time embeddings injected at each layer
5. **Distributed Training**: Supports multi-GPU training with DDP
6. **Muon + AdamW Optimization**: Following nanochat conventions

## Training Tips

1. **Start small**: Begin with `--depth=4` or `--depth=6` to verify the setup works
2. **Diffusion steps**: 1000 steps is standard, but you can train with fewer (e.g., 100) for faster iteration
3. **Sampling steps**: During generation, using 50-100 steps often gives good quality
4. **Learning rate**: The default rates (embedding_lr=0.2, matrix_lr=0.02) work well
5. **Sequence length**: Start with 256-512 tokens, increase as needed

## Experiments to Try

Following the spirit of Andrej Karpathy's recommendations, here are experiments to try:

### 1. **Basic Training Run**
```bash
python -m scripts.diffusion_train --depth=8 --num_iterations=5000 --max_seq_len=256
```

### 2. **Compare with Autoregressive GPT**
Train both a diffusion model and GPT on the same data, compare:
- Training speed (tokens/sec)
- Sample quality
- Controllability

### 3. **Ablation Studies**
- **Noise schedule**: Try "linear" vs "cosine"
- **Diffusion steps**: Train with 100, 500, 1000 steps
- **Architecture depth**: Compare depth 4, 8, 12, 16

### 4. **Generation Speed vs Quality**
- Vary `num_steps` during sampling: 10, 25, 50, 100, 250, 500
- Plot generation quality vs time

### 5. **Controllable Generation**
Modify the model to condition on attributes (sentiment, topic, style)

## Expected Results

With a small model (depth=8-12) trained for 5000-10000 iterations:
- **Training loss**: Should decrease to 0.01-0.05 range
- **Generation**: Will produce somewhat coherent text, though not perfect
- **Speed**: Training ~same as GPT, generation slower but parallelizable

## References

1. **Diffusion-LM**: Li et al., "Diffusion-LM Improves Controllable Text Generation", NeurIPS 2022
2. **Analog Bits**: Chen et al., "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning", ICLR 2023
3. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
4. **Improved DDPM**: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
5. **Spectral Autoregression**: Blog post by Sander Dieleman (referenced in the original request)

## Comparison with nanochat's GPT

| Feature | GPT (nanochat) | Diffusion Model |
|---------|----------------|-----------------|
| Architecture | Causal transformer | Bidirectional transformer |
| Parameters | ~1.9B (d32 model) | Comparable for same depth |
| Training time | ~4 hours (speedrun) | Similar |
| Generation | Fast, sequential | Slower, iterative |
| Controllability | Limited | Better |
| Use case | General chat | Research, controllable gen |

## Troubleshooting

### Out of Memory
- Reduce `--device_batch_size`
- Reduce `--max_seq_len`
- Reduce `--depth`

### Training is slow
- Check GPU utilization
- Try reducing `--diffusion_steps` to 100-500
- Ensure you're using CUDA, not CPU

### Poor sample quality
- Train longer (more iterations)
- Use more denoising steps during sampling
- Try cosine schedule if using linear
- Increase model size (depth)

## Future Improvements

Potential extensions to explore:
1. **Self-conditioning**: Feed previous prediction back into model
2. **Classifier-free guidance**: For better controllable generation
3. **Latent diffusion**: Operate in compressed latent space
4. **Continuous diffusion**: Other parameterizations (v-prediction, flow matching)
5. **Fast sampling**: DDIM, DPM-Solver for fewer steps

## License

Same as nanochat (MIT License)

## Acknowledgments

- **Andrej Karpathy** for nanochat and nanoGPT
- **Sander Dieleman** for the spectral autoregression blog post
- Research community for diffusion model papers

---

**Note**: This is a research implementation for educational purposes. For production use cases, consider well-tested libraries like Hugging Face Diffusers or stability.ai's implementations.
