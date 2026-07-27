#!/usr/bin/env bash
# Sparse-pilot wave: does the world-model prior win when pilots are sparse?
#   bash scripts/sparse-wave.sh [DATA_DIR] [STEPS]
#
# With comb pilots, unobserved subcarriers have NO observation -> the estimate must lean on the
# prior. If the WM prior is real, full >> noprior here (unlike the dense case where they tied).
# 8 runs / 8 GPUs, one wave:
#   in-dist: stride {2,4,8} x {full, noprior}         (GPU 0-5)
#   OOD:     stride 4 x {full, noprior} (held-out)    (GPU 6-7)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
STEPS="${2:-15000}"
OODSCENE="simple_street_canyon"
mkdir -p results/sparse

# run GPU TAG STRIDE ABLATE [holdout]
run() {
  local gpu=$1 tag=$2 stride=$3 ab=$4 holdout=${5:-}
  local args="--data_dir $DATA --steps $STEPS --bs 96 --lr 3e-4 --tag $tag --stride $stride --ablate $ab --seed 0"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29800 + gpu)) \
    scripts/train-beam-sparse.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (stride=$stride ablate=$ab holdout=${holdout:-none}) pid $!"
}

echo "==== SPARSE-PILOT WAVE ===="
pids=()
run 0 sp_s2_full    2 full    ; pids+=($!)
run 1 sp_s2_noprior 2 prior   ; pids+=($!)
run 2 sp_s4_full    4 full    ; pids+=($!)
run 3 sp_s4_noprior 4 prior   ; pids+=($!)
run 4 sp_s8_full    8 full    ; pids+=($!)
run 5 sp_s8_noprior 8 prior   ; pids+=($!)
run 6 sp_ood_s4_full    4 full  "$OODSCENE"; pids+=($!)
run 7 sp_ood_s4_noprior 4 prior "$OODSCENE"; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "SPARSE_WAVE_DONE (fail=$fail)"
