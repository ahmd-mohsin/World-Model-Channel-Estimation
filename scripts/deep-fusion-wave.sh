#!/usr/bin/env bash
# Deep-fusion wave: give the world model a ReEsNet-capacity fusion head FED THE PRIOR, and test
# whether it now beats the standalone ReEsNet (which it already ties on capacity but lacks the prior).
#   bash scripts/deep-fusion-wave.sh [DATA_DIR] [STEPS]
#
# Clean control: --deep_fusion --ablate prior = SAME capacity, prior zeroed (== ReEsNet+ourhead).
# So deep-fusion-full vs deep-fusion-noprior isolates the prior's value at identical capacity.
# 8 runs / 8 GPUs:
#   in-dist stride {2,4,8} x {full, noprior}        (GPU 0-5)
#   OOD stride 4 x {full, noprior}                  (GPU 6-7)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
STEPS="${2:-15000}"
OODSCENE="simple_street_canyon"
mkdir -p results/sparse

run() {  # GPU TAG STRIDE ABLATE [holdout]
  local gpu=$1 tag=$2 stride=$3 ab=$4 holdout=${5:-}
  local args="--data_dir $DATA --steps $STEPS --stride $stride --tag $tag --ablate $ab --deep_fusion --seed 0"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30000 + gpu)) \
    scripts/train-beam-sparse.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (deep_fusion stride=$stride ablate=$ab holdout=${holdout:-none}) pid $!"
}

echo "==== DEEP-FUSION WAVE ===="
pids=()
run 0 df_s2_full    2 full  ; pids+=($!)
run 1 df_s2_noprior 2 prior ; pids+=($!)
run 2 df_s4_full    4 full  ; pids+=($!)
run 3 df_s4_noprior 4 prior ; pids+=($!)
run 4 df_s8_full    8 full  ; pids+=($!)
run 5 df_s8_noprior 8 prior ; pids+=($!)
run 6 df_ood_s4_full    4 full  "$OODSCENE"; pids+=($!)
run 7 df_ood_s4_noprior 4 prior "$OODSCENE"; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "DEEP_FUSION_WAVE_DONE (fail=$fail)"
