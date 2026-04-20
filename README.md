# Evaluation Pipeline for Long Homophonic Substitution Ciphers

This repository contains the State-of-the-Art (SOTA) evaluation pipeline for deciphering long homophonic substitution ciphers (up to 10,000+ characters). It utilizes **vLLM** and **PagedAttention** to perform high-throughput, distributed generation across Ada Lovelace (L4) GPUs.

The pipeline executes a **Mandatory Training Objective**, forcing the model to complete the joint-sequence `[C1, C2...][SEP][P1, P2...]` format, collapsing the non-deterministic symbol space into strict plaintext predictions.

## Architecture & Hardware
* **Inference Engine:** vLLM (Dynamic Tensor Parallelism based on allocation)
* **Optimization:** FlashAttention-2 (Native CUDA 12 / PyTorch 2.9)
* **Metric:** Symbol Error Rate (SER)
* **Hardware Profile:** L4 GPUs (24GB VRAM)

## How to Run

Evaluations are triggered via SLURM. The SLURM script automatically synchronizes the `uv` environment and injects the pre-compiled FlashAttention-2 wheel to bypass compute node compilation bottlenecks.

> ⚠️ **IMPORTANT: Tensor Parallelism Constraints**
> Because vLLM distributes the model's architecture across the hardware, the number of GPUs you request **must evenly divide both** of the following model parameters:
> 1. The number of **Attention Heads**
> 2. The **Vocabulary Size**
> 
> *For example: If your model has 6 attention heads and a vocabulary size of 2560, you must allocate 2 GPUs (since 6 is divisible by 3, but 2560 is not).*

**Standard Execution:**
```bash
sbatch --gres=gpu:l4:<NUM_GPUS> eval.slurm --model_path /path/to/model
```

**Examples**
# 4 GPUs (e.g., for a model with 8 attention heads and 2560 vocab size)
```bash
sbatch --gres=gpu:l4:4 eval.slurm --model_path outputs/checkpoint-16750
```

# 4 GPUs with space character inclusion
```bash
sbatch --gres=gpu:l4:4 eval.slurm --model_path outputs/checkpoint-16750 --spaces
```

# 2 GPUs (e.g., for a model with 6 attention heads and 2560 vocab size)
```bash
sbatch --gres=gpu:l4:2 eval.slurm --model_path outputs/llama_no_spaces_ucloud
```