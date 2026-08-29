#!/usr/bin/env bash
# Learned temporal predictor baselines vs the WM predictor: {gru,lstm,transformer} x {slow,fast}.
#   bash scripts/temporal-wave.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
STEPS=15000
run(){ local gpu=$1 data=$2 bk=$3 tag=$4
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((30510+gpu)) \
    scripts/train-temporal-predictor.py --data_dir $data --n_ant 32 --n_sub 32 \
    --backbone $bk --steps $STEPS --tag $tag > results/temporal/run_$tag.log 2>&1 &
  echo "GPU $gpu -> $tag ($bk on $data)"; }
mkdir -p results/temporal
echo "==== TEMPORAL PREDICTOR WAVE ===="
pids=()
run 0 data/mimo      gru         tp_gru_slow;   pids+=($!)
run 1 data/mimo      lstm        tp_lstm_slow;  pids+=($!)
run 2 data/mimo      transformer tp_tf_slow;    pids+=($!)
run 3 data/mimo_fast gru         tp_gru_fast;   pids+=($!)
run 4 data/mimo_fast lstm        tp_lstm_fast;  pids+=($!)
run 5 data/mimo_fast transformer tp_tf_fast;    pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo TEMPORAL_WAVE_DONE
