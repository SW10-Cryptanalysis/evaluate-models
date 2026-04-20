# Evaluation Pipeline for Long Homophonic Substitution Ciphers

This repository contains the State-of-the-Art (SOTA) evaluation pipeline for deciphering long homophonic substitution ciphers (up to 10,000+ characters). It utilizes **vLLM** and **PagedAttention** to perform high-throughput, distributed generation across Ada Lovelace (L4) GPUs.

The pipeline executes a **Mandatory Training Objective**, forcing the model to complete the joint-sequence `[C1, C2...][SEP][P1, P2...]` format, collapsing the non-deterministic symbol space into strict plaintext predictions.

## Architecture & Hardware
* **Inference Engine:** vLLM (Tensor Parallelism = 4)
* **Optimization:** FlashAttention-2 (Native CUDA 12 / PyTorch 2.9)
* **Metric:** Symbol Error Rate (SER)
* **Hardware Profile:** 4x L4 GPUs (24GB VRAM)

## How to Run

Evaluations are triggered via SLURM. The SLURM script automatically synchronizes the `uv` environment and injects the pre-compiled FlashAttention-2 wheel to bypass compute node compilation bottlenecks.

**Standard Execution:**
```bash
sbatch --gres=gpu:l4:4 eval.slurm --model_path /path/to/model
```

**Examples:**
```bash
sbatch --gres=gpu:l4:4 eval.slurm --model_path outputs/checkpoint-16750

sbatch --gres=gpu:l4:4 eval.slurm --model_path outputs/checkpoint-16750 --spaces

sbatch --gres=gpu:l4:3 eval.slurm --model_path outputs/llama_no_spaces_ucloud --spaces
```
