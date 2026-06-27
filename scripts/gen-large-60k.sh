#!/usr/bin/env bash
# Generate ~60k velocity-action Sionna sequences across all 8 GPUs.
#   bash scripts/gen-large-60k.sh [N_PER_SHARD]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N="${1:-7500}"   # 7500 x 8 = 60000
DATA=data/act60k
mkdir -p "$DATA"
SCENES=(munich etoile florence san_francisco simple_street_canyon)

pids=()
for g in $(seq 0 7); do
  sc=${SCENES[$((g % 5))]}
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_actions.py \
     --shard "$g" --n "$N" --scene "$sc" --step 0.15 --out "$DATA/shard_$g.pt" \
     > "$DATA/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "GPU $g -> $sc (pid $!)"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo FAIL; tail -3 "$DATA"/shard_*.log; exit 1; }
python -c "import torch,glob; print('TOTAL', sum(torch.load(f,map_location='cpu')['data'].shape[0] for f in glob.glob('$DATA/shard_*.pt')))"
echo GEN_DONE
