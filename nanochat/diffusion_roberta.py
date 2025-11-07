"""
Diffusion Language Model with RoBERTa Architecture

Key features:
- Bidirectional self-attention (no causal masking)
- Continuous diffusion process for discrete tokens
- Embedding-based diffusion (operates in continuous embedding space)
- Cosine noise schedule
- Iterative denoising for generation

Based on concepts from:
- "Diffusion-LM Improves Controllable Text Generation" (Li et al., 2022)
- "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning" (Chen et al., 2022)
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class DiffusionRoBERTaConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12  # RoBERTa uses equal heads (no MQA)
    n_embd: int = 768
    diffusion_steps: int = 1000  # number of diffusion timesteps
    noise_schedule: str = "cosine"  # "cosine" or "linear"


def norm(x):
    """Purely functional rmsnorm with no learnable params"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary positional embeddings"""
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


class BidirectionalSelfAttention(nn.Module):
    """Bidirectional self-attention (no causal masking) for RoBERTa"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, attention_mask=None):
        B, T, C = x.size()

        # Project to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply Rotary Embeddings
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Bidirectional attention (no causal mask!)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False)

        # Re-assemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, attention_mask=None):
        x = x + self.attn(norm(x), cos_sin, attention_mask)
        x = x + self.mlp(norm(x))
        return x


class DiffusionRoBERTa(nn.Module):
    """
    Diffusion Language Model with RoBERTa architecture.

    The model operates in continuous embedding space and uses a diffusion process
    to generate text by iteratively denoising from random noise.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Time step embedding for diffusion
        self.time_embedding = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])

        # Output projection to predict denoised embeddings
        self.output_projection = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # For converting embeddings back to logits over vocabulary
        self.to_logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary embeddings setup
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Noise schedule
        self.register_buffer("sqrt_alphas_cumprod", self._get_noise_schedule(config.diffusion_steps, config.noise_schedule)[0])
        self.register_buffer("sqrt_one_minus_alphas_cumprod", self._get_noise_schedule(config.diffusion_steps, config.noise_schedule)[1])

    def _get_noise_schedule(self, steps, schedule_type="cosine"):
        """
        Create noise schedule for diffusion process.
        Returns sqrt_alphas_cumprod and sqrt_one_minus_alphas_cumprod.
        """
        if schedule_type == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            def alpha_bar(t):
                return math.cos((t / steps + 0.008) / 1.008 * math.pi / 2) ** 2

            betas = []
            for i in range(steps):
                t1 = i / steps
                t2 = (i + 1) / steps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
            betas = torch.tensor(betas)
        else:
            # Linear schedule
            betas = torch.linspace(0.0001, 0.02, steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """Precompute rotary embeddings"""
        if device is None:
            device = torch.device("cpu")

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def init_weights(self):
        """Initialize weights"""
        self.apply(self._init_weights)
        # Zero out output projections
        torch.nn.init.zeros_(self.output_projection.weight)
        torch.nn.init.zeros_(self.to_logits.weight)
        for block in self.transformer_blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        # Re-init rotary embeddings on proper device
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        if self.token_embedding.weight.device.type == "cuda":
            self.token_embedding.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        """Initialize weights for individual modules"""
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def get_device(self):
        return self.token_embedding.weight.device

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Create sinusoidal timestep embeddings.
        Args:
            timesteps: (B,) tensor of timestep indices
            embedding_dim: dimension of embedding
        Returns:
            (B, embedding_dim) tensor of timestep embeddings
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad if odd
            emb = F.pad(emb, (0, 1))
        return emb

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to embeddings.
        Args:
            x_start: (B, T, D) clean embeddings
            t: (B,) timestep indices
            noise: optional noise tensor, will be sampled if None
        Returns:
            noisy embeddings
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get noise schedule values for timesteps t
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[:, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[:, None, None]

        # Apply noise: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def forward(self, idx, timesteps=None, attention_mask=None):
        """
        Forward pass for diffusion model.
        Args:
            idx: (B, T) token indices
            timesteps: (B,) timestep indices for diffusion, or None for standard forward
            attention_mask: optional attention mask
        Returns:
            predicted embeddings or noise
        """
        B, T = idx.size()
        device = idx.device

        # Get token embeddings
        x = self.token_embedding(idx)  # (B, T, D)
        x_clean = x.detach()  # Store clean embeddings

        # If timesteps provided, we're training with diffusion
        if timesteps is not None:
            # Add noise to embeddings based on timesteps
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x, timesteps, noise)

            # Get timestep embeddings
            t_emb = self._get_timestep_embedding(timesteps, self.config.n_embd)  # (B, D)
            t_emb = self.time_embedding(t_emb)  # (B, D)

            # Add timestep embedding to all positions
            x = x_noisy + t_emb[:, None, :]  # (B, T, D)

        # Apply norm after embedding
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Forward through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, cos_sin, attention_mask)

        x = norm(x)

        # Project to predict denoised embeddings
        x_pred = self.output_projection(x)

        # During training, we predict the noise
        if timesteps is not None:
            # Predict noise instead of clean signal (epsilon parameterization)
            return x_pred, noise, x_clean
        else:
            # During inference, return predictions
            return x_pred

    def compute_loss(self, idx, attention_mask=None):
        """
        Compute diffusion loss for training.
        Args:
            idx: (B, T) token indices
            attention_mask: optional attention mask
        Returns:
            loss value
        """
        B, T = idx.size()
        device = idx.device

        # Sample random timesteps for each example in batch
        timesteps = torch.randint(0, self.config.diffusion_steps, (B,), device=device)

        # Forward pass
        x_pred, noise, x_clean = self.forward(idx, timesteps, attention_mask)

        # Loss: MSE between predicted noise and actual noise
        # (This is the epsilon parameterization used in DDPM)
        loss = F.mse_loss(x_pred, noise, reduction='mean')

        return loss

    @torch.inference_mode()
    def sample(self, batch_size=1, seq_len=None, temperature=1.0, num_steps=None):
        """
        Generate samples using iterative denoising.
        Args:
            batch_size: number of samples to generate
            seq_len: sequence length (defaults to config.sequence_len)
            temperature: sampling temperature
            num_steps: number of denoising steps (defaults to config.diffusion_steps)
        Returns:
            (B, T) tensor of generated token indices
        """
        if seq_len is None:
            seq_len = self.config.sequence_len
        if num_steps is None:
            num_steps = self.config.diffusion_steps

        device = self.get_device()

        # Start from random noise
        x = torch.randn(batch_size, seq_len, self.config.n_embd, device=device)

        # Iteratively denoise
        for t in reversed(range(num_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get timestep embeddings
            t_emb = self._get_timestep_embedding(timesteps, self.config.n_embd)
            t_emb = self.time_embedding(t_emb)

            # Add timestep info
            x_input = norm(x + t_emb[:, None, :])

            # Get rotary embeddings
            cos_sin = self.cos[:, :seq_len], self.sin[:, :seq_len]

            # Forward through transformer
            for block in self.transformer_blocks:
                x_input = block(x_input, cos_sin, None)

            x_input = norm(x_input)
            predicted_noise = self.output_projection(x_input)

            # Denoise using DDPM sampling formula
            alpha_t = self.sqrt_alphas_cumprod[t] ** 2
            alpha_t_minus_1 = self.sqrt_alphas_cumprod[t - 1] ** 2 if t > 0 else 1.0
            beta_t = 1 - alpha_t

            # Predict x_0 from x_t and predicted noise
            x_0_pred = (x - math.sqrt(beta_t) * predicted_noise) / math.sqrt(alpha_t)

            if t > 0:
                # Add noise for next step (not on last step)
                noise = torch.randn_like(x) * temperature
                x = math.sqrt(alpha_t_minus_1) * x_0_pred + math.sqrt(1 - alpha_t_minus_1) * noise
            else:
                x = x_0_pred

        # Convert final embeddings to tokens
        logits = self.to_logits(x)  # (B, T, vocab_size)
        tokens = torch.argmax(logits, dim=-1)  # (B, T)

        return tokens

    def setup_optimizers(self, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """Setup optimizers similar to GPT"""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters
        matrix_params = list(self.transformer_blocks.parameters())
        embedding_params = list(self.token_embedding.parameters())
        other_params = (list(self.time_embedding.parameters()) +
                       list(self.output_projection.parameters()) +
                       list(self.to_logits.parameters()))

        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling LR for AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        adam_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=other_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        # Muon for transformer matrices
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        return optimizers
