"""
GPT-2 / GPT-3 style Transformer training loop, implemented in plain PyTorch.

This is a Python port of the core training logic in train_gpt2.cu (single-GPU,
no custom CUDA kernels, no multi-GPU/ZeRO, no cuDNN attention, no HellaSwag eval).
"""
import argparse
import csv
import glob
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# GPT-2 / GPT-3 model definition

@dataclass
class GPT2Config:
    max_seq_len: int = 1024   # max sequence length
    vocab_size: int = 50257   # vocab size
    padded_vocab_size: int = 50304  # padded to a multiple of 128 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    channels: int = 768


def gpt2_set_hyperparameters(config: GPT2Config, depth: int) -> None:
    depth_to_shape = {
        6: (384, 6),    # (unofficial) gpt2-tiny (30M)
        12: (768, 12),  # gpt2 (124M)
        24: (1024, 16),  # gpt2-medium (350M)
        36: (1280, 20),  # gpt2-large (774M)
        48: (1600, 25),  # gpt2-xl (1558M)
        60: (1920, 30),  # (unofficial) 2.7B
        72: (2880, 30),  # (unofficial) 7.3B
        84: (3456, 36),  # (unofficial) 12.2B
    }
    if depth not in depth_to_shape:
        raise ValueError(f"Unsupported GPT-2 depth: {depth}")
    channels, num_heads = depth_to_shape[depth]
    config.num_layers = depth
    config.channels = channels
    config.num_heads = num_heads
    config.max_seq_len = 1024


def gpt3_set_hyperparameters(config: GPT2Config, channels: int) -> None:
    channels_to_shape = {
        384: (6, 64),     # gpt3-tiny (31M)
        768: (12, 64),    # gpt3-small (125M)
        1024: (24, 64),   # gpt3-medium (350M)
        1536: (24, 96),   # gpt3-large (760M)
        2048: (24, 128),  # gpt3-xl (1.3B)
        2560: (32, 80),   # gpt3-2.7B
        4096: (32, 128),  # gpt3-6.7B
        5140: (40, 128),  # gpt3-13B
        12288: (96, 128),  # gpt3 (175B)
    }
    if channels not in channels_to_shape:
        raise ValueError(f"Unsupported GPT-3 channels: {channels}")
    depth, head_size = channels_to_shape[channels]
    assert channels % head_size == 0
    config.num_layers = depth
    config.channels = channels
    config.num_heads = channels // head_size
    config.max_seq_len = 2048  # GPT-3 uses context length of 2048, up from 1024 in GPT-2


def config_from_descriptor(descriptor: str) -> GPT2Config:
    """
    Builds a GPT2Config from a model descriptor string:
      - legacy format "dX": GPT-2 with X layers, e.g. "d12"
      - "gpt2:dX": same as above, e.g. "gpt2:d48"
      - "gpt3:cX": GPT-3 with X channels, e.g. "gpt3:c768"
    """
    config = GPT2Config()
    if descriptor.startswith("gpt3:c"):
        gpt3_set_hyperparameters(config, int(descriptor[len("gpt3:c"):]))
    elif descriptor.startswith("gpt2:d"):
        gpt2_set_hyperparameters(config, int(descriptor[len("gpt2:d"):]))
    elif descriptor.startswith("d"):
        gpt2_set_hyperparameters(config, int(descriptor[1:]))
    else:
        raise ValueError(f"Unsupported model descriptor: {descriptor}")
    config.vocab_size = 50257
    config.padded_vocab_size = 50304
    return config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.channels % config.num_heads == 0
        self.num_heads = config.num_heads
        self.channels = config.channels
        self.qkv = nn.Linear(config.channels, 3 * config.channels)
        self.attproj = nn.Linear(config.channels, config.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.channels, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.attproj(y)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.fc = nn.Linear(config.channels, 4 * config.channels)
        self.fcproj = nn.Linear(4 * config.channels, config.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        return self.fcproj(x)


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.channels)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.channels)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.padded_vocab_size, config.channels)
        self.wpe = nn.Embedding(config.max_seq_len, config.channels)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.lnf = nn.LayerNorm(config.channels)
        # the classifier head shares weights with the token embedding, matching
        # the .cu code reusing params.wte for the final matmul_forward_cublaslt call
        self.lm_head = nn.Linear(config.channels, config.padded_vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # residual-stream projections are additionally scaled by 1/sqrt(2*L) for training stability
        residual_scale = 1.0 / math.sqrt(2.0 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.attproj.weight, mean=0.0, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp.fcproj.weight, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        B, T = inputs.shape
        assert T <= self.config.max_seq_len, f"sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        pos = torch.arange(0, T, dtype=torch.long, device=inputs.device)
        x = self.wte(inputs) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.lnf(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, weight_decay: float, learning_rate: float) -> torch.optim.Optimizer:
        # weight decay is only applied to the 2D matmul weight tensors (and embeddings),
        # never to biases or layernorm scales, matching gpt2_update()'s per-tensor `wd` logic
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)


# -----------------------------------------------------------------------------
# learning rate scheduler: linear warmup, then cosine decay to a final LR fraction

def get_learning_rate(step: int, base_lr: float, warmup_iters: int, max_steps: int, final_lr_frac: float) -> float:
    if step < warmup_iters:
        return base_lr * (step + 1) / warmup_iters
    if step >= max_steps:
        return base_lr * final_lr_frac
    decay_ratio = (step - warmup_iters) / max(1, max_steps - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = base_lr * final_lr_frac
    return min_lr + coeff * (base_lr - min_lr)


# -----------------------------------------------------------------------------
# outlier detection: tracks a running window of values and reports a z-score for
# each new value, mirroring llmc/outlier_detector.h's OutlierDetector

class OutlierDetector:
    def __init__(self, window_size: int = 128):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, new_value: float) -> float:
        if len(self.buffer) < self.window_size:
            # initial fill phase: not enough data yet to compute a z-score
            self.buffer.append(new_value)
            self.sum += new_value
            self.sum_sq += new_value * new_value
            return float("nan")
        mean = self.sum / self.window_size
        variance = max(0.0, (self.sum_sq / self.window_size) - (mean * mean))
        std = math.sqrt(variance)
        z = (new_value - mean) / std if std > 0 else float("nan")
        old_value = self.buffer[0]  # about to be evicted by the append below
        self.buffer.append(new_value)
        self.sum += new_value - old_value
        self.sum_sq += (new_value * new_value) - (old_value * old_value)
        return z


# -----------------------------------------------------------------------------
# MFU (matrix FLOPS utilization) estimation, mirroring gpt2_estimate_mfu().
# Peak bf16 TFLOPS are hardcoded per device since llmc/mfu.h's lookup table is
# not available here; unrecognized devices fall back to "unknown" (like the
# original's flops_promised < 0 => -1.f case).

GPU_PEAK_BF16_TFLOPS = {
    "H100": 989.0,
    "A100": 312.0,
    "V100": 125.0,
    "L40": 181.0,
    "L4": 121.0,
    "RTX 4090": 165.0,
    "RTX 3090": 71.0,
    "T4": 65.0,
}


def get_flops_promised(device_name: str) -> float:
    """Returns peak bf16 TFLOPS for a known GPU name substring, or None if unknown."""
    for name, tflops in GPU_PEAK_BF16_TFLOPS.items():
        if name in device_name:
            return tflops
    return None


def estimate_mfu(num_parameters: int, config: GPT2Config, seq_len: int, num_tokens: int,
                  dt: float, device_name: str):
    """Returns MFU as a fraction (e.g. 0.42), or None if the GPU's peak flops are unknown."""
    flops_per_token = 6 * num_parameters + 6 * config.num_layers * config.channels * seq_len
    flops_per_step = flops_per_token * num_tokens
    flops_achieved = flops_per_step / dt
    flops_promised = get_flops_promised(device_name)
    if flops_promised is None:
        return None
    return flops_achieved / (flops_promised * 1e12)


# -----------------------------------------------------------------------------
# data loading: tokenizes ClimbMix parquet shards into a flat token stream and
# yields (B, T) input/target batches

def discover_contiguous_shards(data_dir: str) -> list:
    """
    Returns paths to shard_00000.parquet, shard_00001.parquet, ... stopping at the
    first missing index. This ignores any out-of-sequence shard files that may be
    present in data_dir (e.g. a stray shard downloaded separately from a different
    part of the dataset), since those would otherwise break the assumption that
    consecutive shards were downloaded in order.
    """
    shard_paths = []
    index = 0
    while True:
        path = os.path.join(data_dir, shard_filename_from_index(index))
        if not os.path.exists(path):
            break
        shard_paths.append(path)
        index += 1
    return shard_paths


def shard_filename_from_index(index: int) -> str:
    return f"shard_{index:05d}.parquet"


class ParquetTokenDataLoader:
    def __init__(self, data_dir: str, batch_size: int, seq_len: int, split: str = "train"):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token

        shard_paths = discover_contiguous_shards(data_dir)
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_00000.parquet (and onward) found in {data_dir}. Run train_dataset.py first."
            )
        # hold out the last shard for validation, use the rest for training
        if split == "train":
            self.shard_paths = shard_paths[:-1] if len(shard_paths) > 1 else shard_paths
        elif split == "val":
            self.shard_paths = shard_paths[-1:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.shard_index = 0
        self.tokens = self._load_shard(self.shard_paths[self.shard_index])
        self.position = 0

    def _load_shard(self, path: str) -> np.ndarray:
        df = pd.read_parquet(path, columns=["text"])
        token_chunks = []
        for text in df["text"]:
            token_chunks.append(self.enc.encode_ordinary(text))
            token_chunks.append([self.eot])
        tokens = np.concatenate([np.array(chunk, dtype=np.uint16) for chunk in token_chunks])
        return tokens

    def reset(self) -> None:
        self.shard_index = 0
        self.tokens = self._load_shard(self.shard_paths[self.shard_index])
        self.position = 0

    def next_batch(self):
        needed = self.batch_size * self.seq_len + 1
        while self.position + needed > len(self.tokens):
            self.shard_index = (self.shard_index + 1) % len(self.shard_paths)
            self.tokens = self._load_shard(self.shard_paths[self.shard_index])
            self.position = 0
        buf = self.tokens[self.position: self.position + needed]
        inputs = torch.from_numpy(buf[:-1].astype(np.int64)).view(self.batch_size, self.seq_len)
        targets = torch.from_numpy(buf[1:].astype(np.int64)).view(self.batch_size, self.seq_len)
        self.position += self.batch_size * self.seq_len
        return inputs, targets


# -----------------------------------------------------------------------------
# checkpointing

def write_checkpoint(output_dir: str, step: int, model: GPT, optimizer: torch.optim.Optimizer) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"model_{step:08d}.pt")
    print(f"Writing checkpoint to {path}")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(model.config),
    }, path)


def find_latest_checkpoint(output_dir: str):
    if output_dir is None or not os.path.isdir(output_dir):
        return None
    paths = sorted(glob.glob(os.path.join(output_dir, "model_*.pt")))
    return paths[-1] if paths else None


def load_checkpoint(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    config = GPT2Config(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


# -----------------------------------------------------------------------------
# text sampling

@torch.no_grad()
def generate_sample(model: GPT, enc, device: str, gen_tokens: int = 64) -> str:
    model.eval()
    tokens = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    for _ in range(gen_tokens - 1):
        context = tokens[:, -model.config.max_seq_len:]
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :model.config.vocab_size], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
    model.train()
    return enc.decode(tokens[0].tolist())


# -----------------------------------------------------------------------------
# parameter/config summary table, mirroring the "+---+" ASCII table printed near
# the top of train_gpt2.cu's main()

def print_parameter_table(args, config: GPT2Config, num_parameters: int, device: str, precision_str: str) -> None:
    rows = [
        ("input dir", args.input_dir),
        ("output dir", args.output_dir if args.output_dir is not None else "NULL"),
        ("model descriptor", args.model),
        ("resume", args.resume),
        ("micro batch size B", args.batch_size),
        ("sequence length T", args.seq_len),
        ("total batch size", args.total_batch_size if args.total_batch_size != -1 else args.batch_size * args.seq_len),
        ("learning rate (LR)", args.learning_rate),
        ("warmup iterations", args.warmup_iters),
        ("final LR fraction", args.final_lr_frac),
        ("weight decay", args.weight_decay),
        ("grad clip", args.grad_clip),
        ("max_steps", args.max_steps),
        ("val_loss_every", args.val_loss_every),
        ("val_max_steps", args.val_max_steps),
        ("sample_every", args.sample_every),
        ("checkpoint_every", args.checkpoint_every),
        ("num_layers", config.num_layers),
        ("num_heads", config.num_heads),
        ("channels", config.channels),
        ("vocab_size", config.vocab_size),
        ("padded_vocab_size", config.padded_vocab_size),
        ("num_parameters", num_parameters),
        ("device", device),
        ("precision", precision_str),
    ]
    name_width = max(len(name) for name, _ in rows)
    value_width = max(50, max(len(str(value)) for _, value in rows))
    border = "+" + "-" * (name_width + 2) + "+" + "-" * (value_width + 2) + "+"
    print(border)
    print(f"| {'Parameter':<{name_width}} | {'Value':<{value_width}} |")
    print(border)
    for name, value in rows:
        print(f"| {name:<{name_width}} | {str(value):<{value_width}} |")
    print(border)


# -----------------------------------------------------------------------------
# main training loop

def main():
    parser = argparse.ArgumentParser(description="Train a GPT-2 / GPT-3 style model")
    # file system input / output
    parser.add_argument("--input-dir", type=str, default="data", help="directory of downloaded parquet shards")
    parser.add_argument("--output-dir", type=str, default=None, help="output log/checkpoint dir (default: no checkpointing)")
    parser.add_argument("--model", type=str, default="d12", help="model descriptor, e.g. d12, gpt3:c768")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --output-dir")
    # token layout for each step of the optimization
    parser.add_argument("--batch-size", type=int, default=4, help="micro batch size B")
    parser.add_argument("--seq-len", type=int, default=1024, help="sequence length T")
    parser.add_argument("--total-batch-size", type=int, default=-1, help="total desired batch size in tokens (default = batch_size * seq_len, no grad accumulation)")
    # workload
    parser.add_argument("--max-steps", type=int, default=100, help="max steps of optimization to run")
    # optimization
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--final-lr-frac", type=float, default=1.0, help="final LR as a fraction of the base LR")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    # evaluation
    parser.add_argument("--val-loss-every", type=int, default=20)
    parser.add_argument("--val-max-steps", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--gen-tokens", type=int, default=64)
    # checkpointing
    parser.add_argument("--checkpoint-every", type=int, default=0, help="checkpoint every how many steps (0 = disabled)")
    # metrics logging (consumed by visualize_training.ipynb)
    parser.add_argument("--log-dir", type=str, default="logs", help="directory to write metrics.csv for visualization (empty string = disabled)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokens_per_step = args.batch_size * args.seq_len
    total_batch_size = args.total_batch_size if args.total_batch_size != -1 else tokens_per_step
    assert total_batch_size % tokens_per_step == 0
    grad_accum_steps = total_batch_size // tokens_per_step
    print(f"total_batch_size={total_batch_size} => grad_accum_steps={grad_accum_steps}")

    step = 0
    resume_path = find_latest_checkpoint(args.output_dir) if args.resume else None
    if resume_path is not None:
        print(f"Resuming from checkpoint {resume_path}")
        model, checkpoint = load_checkpoint(resume_path, device)
        step = checkpoint["step"]
    else:
        config = config_from_descriptor(args.model)
        model = GPT(config).to(device)

    print(f"num_parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = model.configure_optimizer(args.weight_decay, args.learning_rate)
    if resume_path is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    enc = tiktoken.get_encoding("gpt2")
    train_loader = ParquetTokenDataLoader(args.input_dir, args.batch_size, args.seq_len, split="train")
    val_loader = ParquetTokenDataLoader(args.input_dir, args.batch_size, args.seq_len, split="val")

    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    autocast_ctx = torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda"))
    precision_str = "BF16" if device == "cuda" else "FP32"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    num_parameters = sum(p.numel() for p in model.parameters())
    print_parameter_table(args, model.config, num_parameters, device_name, precision_str)

    loss_outlier_detector = OutlierDetector()
    grad_norm_outlier_detector = OutlierDetector()

    metrics_writer = None
    metrics_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        metrics_path = os.path.join(args.log_dir, "metrics.csv")
        metrics_file = open(metrics_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([
            "step", "train_loss", "val_loss", "grad_norm", "loss_zscore",
            "grad_norm_zscore", "learning_rate", "tokens_per_second", "mfu",
        ])
        print(f"Logging metrics to {metrics_path}")

    last_val_loss = float("nan")

    while step <= args.max_steps:
        last_step = step == args.max_steps

        if step % args.val_loss_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(args.val_max_steps):
                    inputs, targets = val_loader.next_batch()
                    inputs, targets = inputs.to(device), targets.to(device)
                    with autocast_ctx:
                        _, loss = model(inputs, targets)
                    val_loss += loss.item()
            val_loss /= args.val_max_steps
            print(f"val loss {val_loss:.6f}")
            last_val_loss = val_loss
            model.train()

        if device == "cuda" and args.sample_every > 0 and (step > 0 and step % args.sample_every == 0 or last_step):
            print("generating:\n---")
            text = generate_sample(model, enc, device, args.gen_tokens)
            print(text)
            print("---")

        if args.checkpoint_every > 0 and args.output_dir is not None and (step > 0 and step % args.checkpoint_every == 0 or last_step):
            write_checkpoint(args.output_dir, step, model, optimizer)

        if last_step:
            break

        # --------------- training step ---------------
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            inputs, targets = train_loader.next_batch()
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_ctx:
                _, loss = model(inputs, targets)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_learning_rate(step, args.learning_rate, args.warmup_iters, args.max_steps, args.final_lr_frac)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_per_second = total_batch_size / dt
        zloss = loss_outlier_detector.update(loss_accum)
        zgrad = grad_norm_outlier_detector.update(float(grad_norm))
        mfu = estimate_mfu(num_parameters, model.config, args.seq_len, total_batch_size, dt, device_name)
        mfu_str = f"{100 * mfu:.1f}% {precision_str.lower()} MFU" if mfu is not None else "MFU n/a"
        print(f"step {step + 1:4d}/{args.max_steps} | loss {loss_accum:.6f} ({zloss:+.2f}z) | "
              f"norm {grad_norm:.4f} ({zgrad:+.2f}z) | lr {lr:.2e} | {dt * 1000:.2f} ms | "
              f"{mfu_str} | {tokens_per_second:.0f} tok/s")

        if metrics_writer is not None:
            metrics_writer.writerow([
                step + 1, loss_accum, last_val_loss, float(grad_norm), zloss,
                zgrad, lr, tokens_per_second, mfu if mfu is not None else "",
            ])
            metrics_file.flush()

        step += 1

    if metrics_file is not None:
        metrics_file.close()


if __name__ == "__main__":
    main()
