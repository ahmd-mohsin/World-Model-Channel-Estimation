#!/usr/bin/env bash
# Learned deep-baseline (ReEsNet) wave for sparse-pilot estimation — the apples-to-apples competitor
# to Beam-WM (same masked data, no oracle covariance, no temporal history).
#   bash scripts/deep-baseline-wave.sh [DATA_DIR] [STEPS]
#
# 4 runs (deep baseline at each density) + 4 extra Beam-WM seeds to firm up the comparison = 8 GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
STEPS="${2:-15000}"
OODSCENE="simple_street_canyon"
mkdir -p results/sparse

deep() {  # GPU TAG STRIDE [holdout]
  local gpu=$1 tag=$2 stride=$3 holdout=${4:-}
  local args="--data_dir $DATA --steps $STEPS --stride $stride --tag $tag --seed 0"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29900 + gpu)) \
    scripts/train-deep-baseline.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (deep, stride=$stride holdout=${holdout:-none}) pid $!"
}
wm() {    # GPU TAG STRIDE SEED [holdout]  (extra Beam-WM full runs for seed variance)
  local gpu=$1 tag=$2 stride=$3 seed=$4 holdout=${5:-}
  local args="--data_dir $DATA --steps $STEPS --stride $stride --tag $tag --ablate full --seed $seed"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29900 + gpu)) \
    scripts/train-beam-sparse.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (beam-wm seed=$seed stride=$stride) pid $!"
}

echo "==== DEEP-BASELINE WAVE ===="
pids=()
deep 0 deep_s2      2 ; pids+=($!)
deep 1 deep_s4      4 ; pids+=($!)
deep 2 deep_s8      8 ; pids+=($!)
deep 3 deep_ood_s4  4 "$OODSCENE"; pids+=($!)
wm   4 sp_s4_full_s1 4 1 ; pids+=($!)
wm   5 sp_s4_full_s2 4 2 ; pids+=($!)
wm   6 sp_s8_full_s1 8 1 ; pids+=($!)
wm   7 sp_s8_full_s2 8 2 ; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "DEEP_WAVE_DONE (fail=$fail)"
