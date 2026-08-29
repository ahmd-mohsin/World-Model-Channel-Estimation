#!/usr/bin/env bash
# Missing baseline (#3): train the learned per-snapshot CNN (ReEsNet) on FULL-GRID estimation
# (stride 1 = all subcarriers observed, no masking) so we can compare the world model vs a strong
# LEARNED estimator on full-grid — not just vs classical MMSE. 3 in-dist seeds + 5 OOD scenes.
#   bash scripts/fullgrid-cnn-wave.sh [DATA]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
DATA="${1:-data/mimo}"; STEPS=15000
mkdir -p results/fullgrid_cnn
run(){ local gpu=$1 tag=$2 seed=$3 holdout=${4:-}
  local args="--data_dir $DATA --n_ant 32 --n_sub 32 --stride 1 --steps $STEPS --seed $seed --tag $tag"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30400+gpu)) \
    scripts/train-deep-baseline.py $args > results/fullgrid_cnn/run_$tag.log 2>&1 &
  echo "GPU $gpu -> $tag (seed=$seed holdout=${holdout:-none})"; }
echo "==== FULL-GRID CNN WAVE ===="
pids=()
run 0 fg_cnn_s0 0; pids+=($!)
run 1 fg_cnn_s1 1; pids+=($!)
run 2 fg_cnn_s2 2; pids+=($!)
run 3 fg_cnn_ood_munich       0 munich; pids+=($!)
run 4 fg_cnn_ood_etoile       0 etoile; pids+=($!)
run 5 fg_cnn_ood_florence     0 florence; pids+=($!)
run 6 fg_cnn_ood_san_francisco 0 san_francisco; pids+=($!)
run 7 fg_cnn_ood_simple_street_canyon 0 simple_street_canyon; pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo FULLGRID_CNN_DONE
