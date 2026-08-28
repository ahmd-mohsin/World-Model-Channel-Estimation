#!/usr/bin/env bash
# Generate the true-MIMO dataset (8 TX x 4 RX = 32 spatial channels) across 6 scenes, 8 GPUs.
#   bash scripts/gen-mimo.sh [N_PER_SHARD]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

N="${1:-6000}"          # 6000 x 8 = 48000 sequences
DATA=data/mimo
mkdir -p "$DATA"
SCENES=(munich etoile florence san_francisco simple_street_canyon simple_street_canyon_with_cars munich san_francisco)

pids=()
for g in $(seq 0 7); do
  sc=${SCENES[$g]}
  CUDA_VISIBLE_DEVICES=$g python scripts/gen_sionna_mimo.py \
     --shard "$g" --n "$N" --scene "$sc" --n_tx 8 --n_rx 4 --n_sub 32 \
     --step 0.15 --speed_lo 0.3 --speed_hi 2.0 --out "$DATA/shard_$g.pt" \
     > "$DATA/shard_$g.log" 2>&1 &
  pids+=($!)
  echo "GPU $g -> $sc (8tx x 4rx MIMO) pid $!"
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo FAIL; tail -3 "$DATA"/shard_*.log; exit 1; }
python -c "import torch,glob; fs=sorted(glob.glob('$DATA/shard_*.pt')); \
print('TOTAL', sum(torch.load(f,map_location='cpu',weights_only=False)['data'].shape[0] for f in fs)); \
s=torch.load(fs[0],map_location='cpu',weights_only=False); print('shape',tuple(s['data'].shape),'ntx',s['n_tx'],'nrx',s['n_rx'])"
echo GEN_DONE
