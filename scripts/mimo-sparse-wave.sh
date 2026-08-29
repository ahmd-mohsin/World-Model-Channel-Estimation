#!/usr/bin/env bash
# Massive-MIMO (Na=32) sparse-pilot comparison: does the WM prior's advantage grow with antenna
# dimension and flip the SNR extremes where a lean CNN won at Na=8?
#   bash scripts/mimo32-wave.sh
# 8 runs / 8 GPUs: DeepFusion+finetune (stride 2/4/8 + OOD) vs standalone ReEsNet (stride 2/4/8 + OOD).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/mimo}"
JOINT=12000
FT=4000
OOD=simple_street_canyon
mkdir -p results/sparse

wm() {   # GPU TAG STRIDE [holdout]
  local gpu=$1 tag=$2 stride=$3 holdout=${4:-}
  local args="--data_dir $DATA --n_ant 32 --n_sub 32 --steps $JOINT --finetune_steps $FT \
--stride $stride --tag $tag --ablate full --deep_fusion --seed 0"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30200 + gpu)) \
    scripts/train-beam-sparse.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (WM+ft Na=32 stride=$stride holdout=${holdout:-none}) pid $!"
}
ree() {  # GPU TAG STRIDE [holdout]
  local gpu=$1 tag=$2 stride=$3 holdout=${4:-}
  local args="--data_dir $DATA --n_ant 32 --n_sub 32 --steps $JOINT --stride $stride --tag $tag --seed 0"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30200 + gpu)) \
    scripts/train-deep-baseline.py $args > "results/sparse/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (ReEsNet Na=32 stride=$stride holdout=${holdout:-none}) pid $!"
}

echo "==== MIMO 8x4 SPARSE+DEEP WAVE ===="
pids=()
wm  0 m32_dft_s2      2 ; pids+=($!)
wm  1 m32_dft_s4      4 ; pids+=($!)
wm  2 m32_dft_s8      8 ; pids+=($!)
wm  3 m32_dft_ood_s4  4 "$OOD"; pids+=($!)
ree 4 m32_deep_s2     2 ; pids+=($!)
ree 5 m32_deep_s4     4 ; pids+=($!)
ree 6 m32_deep_s8     8 ; pids+=($!)
ree 7 m32_deep_ood_s4 4 "$OOD"; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "MIMO_SPARSE_WAVE_DONE (fail=$fail)"
