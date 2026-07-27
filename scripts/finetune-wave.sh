#!/usr/bin/env bash
# Fine-tuned deep-fusion wave: joint train, then estimation-only fine-tune of the fusion head
# (encoder/SSM/predictor frozen -> prediction stays unbeaten). Goal: beat standalone ReEsNet at
# every density x SNR, removing the multitask-dilution handicap.
#   bash scripts/finetune-wave.sh [DATA_DIR]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
JOINT=12000
FT=4000
OODSCENE="simple_street_canyon"
mkdir -p results/sparse

run() {  # GPU TAG STRIDE SEED [holdout]
  local gpu=$1 tag=$2 stride=$3 seed=$4 holdout=${5:-}
  local args="--data_dir $DATA --steps $JOINT --finetune_steps $FT --stride $stride --tag $tag \
--ablate full --deep_fusion --seed $seed"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30100 + gpu)) \
    scripts/train-beam-sparse.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (ft deep_fusion stride=$stride seed=$seed holdout=${holdout:-none}) pid $!"
}

echo "==== FINE-TUNE WAVE ===="
pids=()
run 0  dft_s2       2 0 ; pids+=($!)
run 1 dft_s4       4 0 ; pids+=($!)
run 2 dft_s8       8 0 ; pids+=($!)
run 3 dft_ood_s4   4 0 "$OODSCENE"; pids+=($!)
run 4 dft_s4_s1    4 1 ; pids+=($!)
run 5 dft_s4_s2    4 2 ; pids+=($!)
run 6 dft_s8_s1    8 1 ; pids+=($!)
run 7 dft_s8_s2    8 2 ; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "FINETUNE_WAVE_DONE (fail=$fail)"
