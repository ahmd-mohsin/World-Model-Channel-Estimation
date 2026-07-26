#!/usr/bin/env bash
# Generate a richer multi-scene Sionna dataset for hard stress-testing.
#   bash scripts/gen-stress6.sh [N_PER_SHARD]
#
# 6 realistic scenes (adds simple_street_canyon_with_cars vs the old 5), wider speed range
# (0.3-2.0 vs 0.5-1.5) to stress under-represented dynamics. 8 shards over 8 GPUs.
# Each shard tags its own scene, so ShardDataset(holdout_scene=...) gives clean leave-one-out OOD.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N="${1:-8000}"        # 8000 x 8 = 64000 sequences
DATA=data/stress6
mkdir -p "$DATA"

# One scene per GPU; the two richest scenes (munich, san_francisco) get a 2nd shard (diff seed
# via --shard) to fill 8 GPUs with per-scene diversity rather than duplicating a small scene.
SCENES=(munich etoile florence san_francisco simple_street_canyon simple_street_canyon_with_cars munich san_francisco)

pids=()
for g in $(seq 0 7); do
  sc=${SCENES[$g]}
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_actions.py \
     --shard "$g" --n "$N" --scene "$sc" --step 0.15 \
     --speed_lo 0.3 --speed_hi 2.0 \
     --out "$DATA/shard_$g.pt" \
     > "$DATA/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "GPU $g -> $sc (pid $!)"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo FAIL; tail -3 "$DATA"/shard_*.log; exit 1; }
python -c "import torch,glob; fs=sorted(glob.glob('$DATA/shard_*.pt')); \
print('TOTAL', sum(torch.load(f,map_location='cpu')['data'].shape[0] for f in fs)); \
[print(torch.load(f,map_location='cpu')['scene'], torch.load(f,map_location='cpu')['data'].shape) for f in fs]"
echo GEN_DONE
