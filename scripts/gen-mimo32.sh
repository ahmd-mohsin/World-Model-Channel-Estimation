#!/usr/bin/env bash
# Generate massive-MIMO (Na=32) data for the scale-up experiment.
#   bash scripts/gen-mimo32.sh [N_PER_SHARD]
# Na=32, Ns=32. 8 shards over 8 GPUs, 6 scenes (2 richest get a 2nd shard). Wider speed 0.3-2.0.
# Larger channels -> slower ray-tracing per sample, so default N is smaller than the Na=8 run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N="${1:-6000}"          # 6000 x 8 = 48000 sequences
DATA=data/mimo32
mkdir -p "$DATA"
SCENES=(munich etoile florence san_francisco simple_street_canyon simple_street_canyon_with_cars munich san_francisco)

pids=()
for g in $(seq 0 7); do
  sc=${SCENES[$g]}
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_actions.py \
     --shard "$g" --n "$N" --scene "$sc" --step 0.15 --speed_lo 0.3 --speed_hi 2.0 \
     --n_ant 32 --n_sub 32 --out "$DATA/shard_$g.pt" \
     > "$DATA/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "GPU $g -> $sc (Na=32) pid $!"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo FAIL; tail -3 "$DATA"/shard_*.log; exit 1; }
python -c "import torch,glob; fs=sorted(glob.glob('$DATA/shard_*.pt')); \
print('TOTAL', sum(torch.load(f,map_location='cpu')['data'].shape[0] for f in fs)); \
print('shape', tuple(torch.load(fs[0],map_location='cpu')['data'].shape))"
echo GEN_DONE
