# baseGPT

A minimal, single-file PyTorch implementation of GPT-2 / GPT-3 style pretraining,
ported from [karpathy/llm.c](https://github.com/karpathy/llm.c)'s `train_gpt2.cu`.
No custom CUDA kernels, no multi-GPU/ZeRO, no cuDNN attention — just plain
`torch.nn` modules, `F.scaled_dot_product_attention`, and standard PyTorch
autograd, so it's easy to read end to end and hack on.

## Features

- GPT-2/GPT-3 model definitions selectable by a short descriptor string
  (`d12`, `d48`, `gpt3:c768`, ...), weight-tied embeddings, matching weight init.
- AdamW with selective weight decay (2D+ tensors only), cosine LR schedule with
  linear warmup.
- Gradient accumulation, gradient clipping, `bfloat16` autocast on CUDA.
- A parquet-shard data loader for the [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
  dataset, tokenized on the fly with `tiktoken`.
- Checkpointing/resume, periodic validation loss, and autoregressive text sampling.
- Console diagnostics matching `train_gpt2.cu`'s output: a parameter summary
  table, z-score outlier detection on loss/grad-norm, and MFU% (matrix FLOPS
  utilization) estimates for common GPUs.
- A metrics CSV logger (`--log-dir`) plus a [Jupyter notebook](visualize_training.ipynb)
  to plot loss, grad norm, z-scores, learning rate, throughput, and MFU% after a run.

## Requirements

- Python 3.10+
- `torch` (CUDA build recommended for real training; CPU works for smoke tests)
- `tiktoken`, `pandas`, `pyarrow`, `numpy`, `requests`, `tqdm`

```bash
pip install torch tiktoken pandas pyarrow numpy requests tqdm
```

## Quick start

### 1. Download the dataset

Downloads parquet shards of the ClimbMix dataset from HuggingFace into `data/`:

```bash
python train_dataset.py --num-shards 170 --data-dir data
```

### 2. Train

```bash
python train_base.py --model d12 --input-dir data --max-steps 1000 --log-dir logs
```

Model size is picked via `--model`:

| Descriptor | Layers | Channels | Heads | Params |
|---|---|---|---|---|
| `d6`  | 6  | 384  | 6  | ~30M   |
| `d12` | 12 | 768  | 12 | ~124M  |
| `d24` | 24 | 1024 | 16 | ~350M  |
| `d36` | 36 | 1280 | 20 | ~774M  |
| `d48` | 48 | 1600 | 25 | ~1.5B  |
| `gpt3:c768` | 12 | 768 | 12 | ~125M (GPT-3 shapes) |

Run `python train_base.py --help` for the full list of flags (batch size,
sequence length, learning rate schedule, checkpointing, sampling, etc).

### 3. Visualize results

After a run with `--log-dir logs`, open [visualize_training.ipynb](visualize_training.ipynb)
and run all cells to plot loss curves, gradient norm / z-score diagnostics,
the learning rate schedule, and throughput/MFU%.

## Project layout

```
train_base.py           # model, optimizer, training loop, diagnostics
train_dataset.py        # ClimbMix parquet shard downloader
visualize_training.ipynb  # post-training matplotlib visualization
data/                    # downloaded parquet shards (not tracked in git)
logs/                    # metrics.csv written by --log-dir (not tracked in git)
```

## Acknowledgements

This project is a Python/PyTorch port of the training loop in
[karpathy/llm.c](https://github.com/karpathy/llm.c), and uses the
[karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
dataset for pretraining.

## License

[MIT](LICENSE)
