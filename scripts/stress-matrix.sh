#!/usr/bin/env bash
# Full stress-test matrix for the beamspace world model on 8x A100.
#   bash scripts/stress-matrix.sh [DATA_DIR] [STEPS]
#
# Each run trains a fresh Beam-WM (DDP is per-run; here we run 1 GPU/run in PARALLEL to fill 8 GPUs
# with independent configs) and emits estimation (eval_*.json) + prediction (pred_*.json).
#
# Matrix (8 runs -> 8 GPUs, one wave):
#   in-distribution, 3 seeds        -> robustness / error bars
#   leave-one-scene-out, 6 scenes   -> OOD (uses --holdout_scene; ties up 5 GPUs)
# Total 3 + 6 = 9 runs; we run 8 in wave 1 and 1 in wave 2. Each run is single-GPU (no DDP) so
# they are independent and we sidestep multi-job NCCL contention.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
STEPS="${2:-15000}"
mkdir -p results/stress dashboard

SCENES=(munich etoile florence san_francisco simple_street_canyon simple_street_canyon_with_cars)

# run_one GPU TAG [holdout_scene] [seed]
run_one() {
  local gpu=$1 tag=$2 holdout=${3:-} seed=${4:-0}
  local args="--data_dir $DATA --steps $STEPS --bs 96 --lr 3e-4 --tag $tag"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29500 + gpu)) \
    scripts/train-beam-wm.py $args --seed "$seed" \
    > "results/stress/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (holdout=${holdout:-none} seed=$seed) pid $!"
}

echo "==== WAVE 1: 6 OOD leave-one-out (GPU 0-5) + 2 in-dist seeds (GPU 6-7) ===="
pids=()
for i in "${!SCENES[@]}"; do
  run_one "$i" "ood_${SCENES[$i]}" "${SCENES[$i]}" 0; pids+=($!)
done
run_one 6 "indist_s0" "" 0; pids+=($!)
run_one 7 "indist_s1" "" 1; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "wave 1 done (fail=$fail)"

echo "==== WAVE 2: 3rd in-dist seed (GPU 0) ===="
run_one 0 "indist_s2" "" 2
wait
echo "STRESS_MATRIX_DONE"
