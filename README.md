# RED: Rényi-Guided Diverse User Simulation

This repository contains an research implementation for training and evaluating **LLM-based user simulators with improved response diversity**. The pipeline first obtains a user simulator with supervised fine-tuning (SFT), then applies a Rényi-guided Group Relative Policy Optimization (GRPO) objective to encourage semantically distinct responses under the same dialogue context while retaining a reference-model constraint.

## 1. Overview

A user simulator generates the next user response conditioned on the dialogue context, such as the user profile and dialogue history. Standard SFT can produce realistic responses but may concentrate on a small number of dominant response patterns.

This project focuses on **within-context semantic diversity**: when the same dialogue context is sampled repeatedly, the generated responses should cover more distinct semantic modes rather than being minor paraphrases of one another.

The training pipeline is:

```text
Dialogue context
      |
      v
SFT user simulator
      |
      v
Sample N responses for the same context
      |
      v
Rényi-2 group diversity / leave-one-out marginal reward
      |
      v
GRPO post-training with a reference-model KL constraint
```

The repository also includes an evaluation script that repeatedly samples each context and reports:

- **RED / 1-U**: a Rényi-inspired effective diversity statistic computed from normalized response embeddings.
- **Self-BLEU**: lexical overlap among repeated samples; lower values indicate greater surface-form diversity.
- **LLM-judge quality**: the proportion of generated responses accepted by a binary quality judge.

## 2. Repository Structure

```text
RED-Diverse-User-Simulation/
├── README.md
├── configs/
│   ├── deepspeed_zero3.yaml     # DeepSpeed ZeRO-3 configuration
│   ├── grpo.yaml                # Rényi-guided GRPO configuration
│   └── sft.yaml                 # SFT hyperparameters and paths
├── scripts/
│   ├── train_grpo.sh            # GRPO training launcher
│   └── train_sft.sh             # SFT training launcher
├── src/
    ├── common.py                # Shared utilities for I/O and reproducibility
    ├── rewards.py               # Rényi-based diversity reward and embedding utilities
    ├── train_grpo.py            # GRPO training entry point
    └── train_sft.py             # SFT training entry point
```

Training data, model checkpoints, and private evaluation data are **not included**. Configure local paths before running the code.


## 3. Supervised Fine-Tuning

Edit `configs/sft.yaml` as needed, then run:

```bash
bash scripts/train_sft.sh
```

Important parameters include:

```yaml
max_length: 2048
learning_rate: 0.0001
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
bf16: true
use_lora: true
```

This path is used as the default starting point for GRPO.

## 4. Rényi-Guided GRPO

Edit `configs/grpo.yaml`, especially model/data paths and GPU-sensitive batch sizes, then run:

```bash
bash scripts/train_grpo.sh
```

The default launcher uses:

```text
configs/deepspeed_zero3.yaml
```

with eight processes on one machine. Adjust `num_processes`, batch size, or DeepSpeed settings for your hardware.

Key GRPO parameters include:

```yaml
num_generations: 8
learning_rate: 0.000005
num_train_epochs: 1
num_iterations: 2
temperature: 0.9
top_p: 0.95
beta: 1.5
loss_type: dapo
```

`beta > 0` is required by the training script so that optimization remains constrained relative to the reference/SFT policy.

## 5. Reproducibility Notes

- Default random seeds are fixed in the training and evaluation scripts.
- Diversity metrics depend on the embedding model, embedding dimensionality, sampling temperature, top-p/top-k settings, and number of repeated samples. Keep these fixed when comparing systems.
- The GRPO implementation assumes `num_generations = 8` and checks that the effective batch size is divisible by this group size.
- The evaluation script is configured for a multi-GPU machine with one worker per GPU. Reduce `WORKER_GPU_IDS` for smaller machines.
- `local_files_only=True` is used throughout to prevent accidental network dependency during experiments.
- Checkpoint and result directories are intentionally relative to the repository root in this anonymized release.
