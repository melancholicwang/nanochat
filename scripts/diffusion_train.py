"""
Train Diffusion Language Model. Run as:

python -m scripts.diffusion_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.diffusion_train

For CPU/Macbook testing:
python -m scripts.diffusion_train --depth=4 --max_seq_len=512 --device_batch_size=1 --total_batch_size=512 --num_iterations=100
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.diffusion_roberta import DiffusionRoBERTa, DiffusionRoBERTaConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "diffusion_dummy"  # wandb run name
# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect)
# Model architecture
depth = 12  # transformer depth
max_seq_len = 512  # max sequence length
diffusion_steps = 1000  # number of diffusion timesteps
noise_schedule = "cosine"  # "cosine" or "linear"
# Training horizon
num_iterations = 5000  # number of training iterations
target_flops = -1.0
target_param_data_ratio = -1  # disabled for diffusion training
# Optimization
device_batch_size = 16  # per-device batch size
total_batch_size = 262144  # total batch size in tokens
embedding_lr = 0.2  # learning rate for embeddings
weight_decay = 0.0
matrix_lr = 0.02  # learning rate for transformer matrices
grad_clip = 1.0
warmup_ratio = 0.05
warmdown_ratio = 0.1
final_lr_frac = 0.1
# Evaluation
eval_every = 250
eval_tokens = 524288
sample_every = 500  # generate samples
# Output
model_tag = "diffusion"
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging
use_dummy_wandb = run.endswith("dummy") or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-diffusion", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model configuration (similar to nanochat's GPT setup)
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"diffusion_steps: {diffusion_steps}")

# Gradient accumulation
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Initialize Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_embd=model_dim,
    diffusion_steps=diffusion_steps,
    noise_schedule=noise_schedule,
)
with torch.device("meta"):
    model_config = DiffusionRoBERTaConfig(**model_config_kwargs)
    model = DiffusionRoBERTa(model_config)
model.to_empty(device=device)
model.init_weights()
orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# Calculate iterations
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (6 * num_params * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")

# Initialize Optimizer
optimizers = model.setup_optimizers(embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# Initialize DataLoaders
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)

# Learning rate schedule
def get_lr(it):
    warmup_iters = int(num_iterations * warmup_ratio)
    warmdown_iters = int(num_iterations * warmdown_ratio)
    # Warmup
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    # Warmdown
    elif it > num_iterations - warmdown_iters:
        progress = (num_iterations - it) / warmdown_iters
        return final_lr_frac + (1.0 - final_lr_frac) * progress
    # Constant
    else:
        return 1.0

# Wrap model with DDP if needed
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])

# Training loop
print0("\n" + "="*80)
print0("Starting Diffusion Model Training")
print0("="*80 + "\n")

train_losses = []
timings = []
step_time_ms = 0.0

for step in range(num_iterations):
    t0 = time.time()
    last_step = (step == num_iterations - 1)

    # Update learning rate
    lr_mult = get_lr(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult

    # Gradient accumulation
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        # Get batch
        x, y = next(train_loader)

        # Forward pass (compute diffusion loss)
        with autocast_ctx:
            loss = model.compute_loss(x)
            loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # Backward pass
        loss.backward()

    # All-reduce loss across processes
    if ddp:
        torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    for opt in optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    step_time_ms = (t1 - t0) * 1000
    tokens_per_sec = total_batch_size / (t1 - t0)

    # Logging
    if master_process:
        train_losses.append(loss_accum.item())
        timings.append(step_time_ms)

        if step % 10 == 0:
            print0(f"step {step:4d}/{num_iterations} | loss {loss_accum.item():.4f} | "
                  f"lr {optimizers[0].param_groups[0]['lr']:.6f} | "
                  f"time {step_time_ms:.1f}ms | tok/s {tokens_per_sec:.0f}")

        # Log to wandb
        wandb_run.log({
            "train/loss": loss_accum.item(),
            "train/lr": optimizers[0].param_groups[0]['lr'],
            "train/step_time_ms": step_time_ms,
            "train/tokens_per_sec": tokens_per_sec,
        }, step=step)

    # Evaluation
    if (eval_every > 0 and step % eval_every == 0) or last_step:
        if master_process:
            print0(f"\n{'='*60}")
            print0(f"Evaluation at step {step}")
            print0(f"{'='*60}")

            model.eval()
            with torch.inference_mode():
                # Validation loss
                val_loader = build_val_loader()
                val_loss_accum = 0.0
                val_steps = max(1, eval_tokens // (device_batch_size * max_seq_len))
                for _ in range(val_steps):
                    x, y = next(val_loader)
                    with autocast_ctx:
                        loss = model.compute_loss(x)
                    val_loss_accum += loss.item()
                val_loss = val_loss_accum / val_steps

                print0(f"Validation loss: {val_loss:.4f}")
                wandb_run.log({"val/loss": val_loss}, step=step)

            model.train()

    # Sample generation
    if (sample_every > 0 and step % sample_every == 0) or last_step:
        if master_process:
            print0(f"\n{'='*60}")
            print0(f"Generating samples at step {step}")
            print0(f"{'='*60}")

            model.eval()
            with torch.inference_mode():
                # Generate samples using diffusion
                sample_len = min(64, max_seq_len)
                tokens = model.sample(batch_size=1, seq_len=sample_len, temperature=1.0, num_steps=50)
                text = tokenizer.decode(tokens[0].tolist())
                print0(f"Generated sample:\n{text}\n")
                wandb_run.log({"samples/text": text}, step=step)

            model.train()

    # Save checkpoint
    if ((step > 0 and step % 1000 == 0) or last_step) and master_process:
        checkpoint_path = f"checkpoints/diffusion_step_{step:06d}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        save_checkpoint(
            orig_model,
            optimizers,
            step,
            checkpoint_path,
            extra_info={"loss": loss_accum.item()}
        )
        print0(f"Saved checkpoint to {checkpoint_path}")

# Training complete
if master_process:
    print0("\n" + "="*80)
    print0("Training Complete!")
    print0("="*80)
    print0(f"Final loss: {train_losses[-1]:.4f}")
    print0(f"Average step time: {sum(timings)/len(timings):.1f}ms")

# Cleanup
compute_cleanup()
