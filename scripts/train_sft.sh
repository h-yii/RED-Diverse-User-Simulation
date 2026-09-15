#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
accelerate launch -m src.train_sft --config configs/sft.yaml "$@"

