# RED: Rényi-Guided Diverse User Simulation

This repository contains an research implementation for training and evaluating **LLM-based user simulators with improved response diversity**. The pipeline first obtains a user simulator with supervised fine-tuning (SFT), then applies a Rényi-guided Group Relative Policy Optimization (GRPO) objective to encourage semantically distinct responses under the same dialogue context while retaining a reference-model constraint.

## 1. Overview

A user simulator generates the next user response conditioned on the dialogue context. Standard SFT can produce realistic responses but may concentrate on a small number of dominant response patterns.

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

Training data, model checkpoints, and evaluation data are not included in the current release. The relevant data will be released in a future update. Please configure local paths before running the code.

## 3. Data Format

Training data, model checkpoints, and evaluation data are not included in the current release. **The datasets used in this work will be released in a future update.** Please configure local paths before running the code.

The dataset is stored in JSONL format, with one training example per line. Each example contains the dialogue prompt, the target user response, and auxiliary metadata used during training and evaluation.

A simplified and anonymized example is shown below:

```json
{
  "id": "session_000106:2",
  "session_id": "session_000106",
  "prompt": [
    {
      "role": "system",
      "content": "You are simulating a real user in a telephone dialogue. Generate the user's next natural, concise, and conversational response based on the user profile and dialogue history."
    },
    {
      "role": "user",
      "content": "Hello, may I confirm that I am speaking with you?"
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "Yes, who is this?"
    }
  ],
  "ground_truth": "Yes, who is this?",
  "user_info": "Gender: male; overdue duration: long-term; region: ...; financial attributes: ...",
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

The main fields are:

- `id`: unique identifier of the current dialogue turn.
- `session_id`: identifier shared by turns from the same dialogue session.
- `prompt`: model input in chat format, including the system instruction and dialogue context.
- `completion`: target next-user response in chat-template format.
- `ground_truth`: plain-text target response used for evaluation or preprocessing.
- `user_info`: structured or serialized user-profile information associated with the dialogue.
- `chat_template_kwargs`: optional arguments passed to the model chat template.

For SFT, the model learns to generate `completion` conditioned on `prompt`.

For GRPO training, only the dialogue context is required as input. The current policy samples multiple responses for the same prompt during training, and the sampled responses are used to compute the group-level Rényi diversity reward. Therefore, multiple candidate responses do not need to be precomputed in the dataset.

The public dataset released in a future update will follow the same general JSONL structure and preprocessing conventions expected by the training scripts in this repository.


## 4. Supervised Fine-Tuning

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

## 5. Rényi-Guided GRPO

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

## 6. Reproducibility Notes

- Default random seeds are fixed in the training and evaluation scripts.
- Diversity metrics depend on the embedding model, embedding dimensionality, sampling temperature, top-p/top-k settings, and number of repeated samples. Keep these fixed when comparing systems.
- The GRPO implementation assumes `num_generations = 8` and checks that the effective batch size is divisible by this group size.
- The evaluation script is configured for a multi-GPU machine with one worker per GPU. Reduce `WORKER_GPU_IDS` for smaller machines.
- `local_files_only=True` is used throughout to prevent accidental network dependency during experiments.
- Checkpoint and result directories are intentionally relative to the repository root in this anonymized release.
