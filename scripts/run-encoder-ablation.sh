#!/usr/bin/env bash
# Encoder ablation on 8 GPUs: frozen vs LoRA vs full fine-tune LWM.
# Each arm is a full 8-GPU e2e run; outputs tagged dashboard/{metrics,eval}_<arm>.json.
#   bash scripts/run-encoder-ablation.sh [STEPS]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

STEPS="${1:-12000}"
DATA=data/act60k

for ARM in frozen lora full; do
  echo "==================== ENCODER ARM: $ARM ===================="
  torchrun --nproc_per_node=8 --master_port=29560 scripts/train-e2e.py \
     --data_dir "$DATA" --steps "$STEPS" --bs 96 --lr 3e-4 \
     --encoder "$ARM" --tag "$ARM"
  echo "==================== $ARM DONE ===================="
done
echo "ALL_ARMS_DONE"
