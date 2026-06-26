#!/usr/bin/env bash
#
# run-parallel-pretrain.sh — RUN ON THE BOX.
# Generates Sionna data across all 8 A100s in parallel, then pretrains the head.
#
#   bash scripts/run-parallel-pretrain.sh [N_PER_SHARD] [STEPS]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N_PER_SHARD="${1:-256}"
STEPS="${2:-4000}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DATA_DIR=data/sionna_shards
mkdir -p "$DATA_DIR"

echo "==== Parallel Sionna generation: $NGPU shards x $N_PER_SHARD seqs ===="
pids=()
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_shard.py \
     --shard "$g" --n "$N_PER_SHARD" --out "$DATA_DIR/shard_$g.pt" \
     > "$DATA_DIR/shard_$g.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} generation jobs: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a shard failed; see $DATA_DIR/*.log"; tail -5 "$DATA_DIR"/shard_*.log; exit 1; }
echo "==== generation done; shards: ===="
ls -la "$DATA_DIR"/*.pt

echo "==== Pretrain head on pooled data ($STEPS steps) ===="
python implementation/context_encoder/pretrain_head.py --data_dir "$DATA_DIR" --steps "$STEPS"
