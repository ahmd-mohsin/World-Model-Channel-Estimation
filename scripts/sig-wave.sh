#!/usr/bin/env bash
# Multi-seed significance for the deep-SSM-vs-Transformer prediction tie. 2 extra seeds x {deepssm-L3,
# transformer} x {slow,fast} = 8 runs on 8 GPUs. Combined with seed-0 (gap-close-wave) -> 3 seeds.
#   bash scripts/sig-wave.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
STEPS=15000
r(){ local gpu=$1 data=$2 bk=$3 seed=$4 tag=$5 extra="${6:-}"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((32000+gpu)) \
    scripts/train-temporal-predictor.py --data_dir $data --n_ant 32 --n_sub 32 \
    --backbone $bk --seed $seed $extra --steps $STEPS --tag $tag > results/temporal/run_$tag.log 2>&1 &
  echo "GPU $gpu -> $tag"; }
mkdir -p results/temporal
echo "==== SIGNIFICANCE WAVE (seeds 1,2) ===="
pids=()
r 0 data/mimo      deepssm     1 ds3_slow_s1 "--n_layers 3"; pids+=($!)
r 1 data/mimo      transformer 1 tf_slow_s1  "";             pids+=($!)
r 2 data/mimo      deepssm     2 ds3_slow_s2 "--n_layers 3"; pids+=($!)
r 3 data/mimo      transformer 2 tf_slow_s2  "";             pids+=($!)
r 4 data/mimo_fast deepssm     1 ds3_fast_s1 "--n_layers 3"; pids+=($!)
r 5 data/mimo_fast transformer 1 tf_fast_s1  "";             pids+=($!)
r 6 data/mimo_fast deepssm     2 ds3_fast_s2 "--n_layers 3"; pids+=($!)
r 7 data/mimo_fast transformer 2 tf_fast_s2  "";             pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo SIG_WAVE_DONE > results/temporal/sig_done.flag
echo SIG_WAVE_DONE
