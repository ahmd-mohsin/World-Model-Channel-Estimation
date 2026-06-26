#!/usr/bin/env bash
#
# run-large-scale.sh — RUN ON THE BOX. Large-scale Sionna generation on all 8 GPUs
# across multiple city scenes, then full SSWM JEPA training on the pooled dataset.
#
#   bash scripts/run-large-scale.sh [N_PER_SHARD] [STEPS]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N_PER_SHARD="${1:-500}"
STEPS="${2:-8000}"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DATA_DIR=data/large
mkdir -p "$DATA_DIR"
SCENES=(munich etoile florence san_francisco simple_street_canyon)

echo "==== Large-scale Sionna generation: $NGPU GPUs x $N_PER_SHARD seqs ===="
pids=()
for g in $(seq 0 $((NGPU-1))); do
  scene=${SCENES[$((g % ${#SCENES[@]}))]}
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_large.py \
     --shard "$g" --n "$N_PER_SHARD" --scene "$scene" --out "$DATA_DIR/shard_$g.pt" \
     > "$DATA_DIR/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "  GPU $g -> scene $scene (pid $!)"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a shard failed:"; tail -3 "$DATA_DIR"/shard_*.log; exit 1; }
echo "==== generation done ===="
ls -la "$DATA_DIR"/*.pt
python -c "
import torch,glob
n=sum(torch.load(f,map_location='cpu')['data'].shape[0] for f in glob.glob('$DATA_DIR/shard_*.pt'))
print(f'TOTAL sequences: {n}')"

echo "==== Full SSWM JEPA training ($STEPS steps) ===="
python scripts/train-sswm-large.py --data_dir "$DATA_DIR" --steps "$STEPS"
