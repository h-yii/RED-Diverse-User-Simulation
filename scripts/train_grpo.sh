#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
accelerate launch --config_file configs/deepspeed_zero3.yaml -m src.train_grpo --config configs/grpo.yaml "$@"
