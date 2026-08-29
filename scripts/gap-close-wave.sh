#!/usr/bin/env bash
# Gap-closing prediction comparison: deep stacked selective-SSM (the fair strong SSM) vs Transformer /
# GRU / LSTM / single-layer-SSM-WM, ALL trained prediction-only on the SAME data. Isolates whether a
# PROPERLY DEEP selective-SSM matches attention on channel prediction.
#   bash scripts/gap-close-wave.sh <DATA_DIR> <TAGSUFFIX>     e.g. bash scripts/gap-close-wave.sh data/mimo slow
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
DATA="${1:-data/mimo}"; SFX="${2:-slow}"; STEPS=15000
mkdir -p results/temporal results/stress
tp(){ local gpu=$1 bk=$2 tag=$3 extra="${4:-}"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((31000+gpu)) \
    scripts/train-temporal-predictor.py --data_dir $DATA --n_ant 32 --n_sub 32 \
    --backbone $bk $extra --steps $STEPS --tag $tag > results/temporal/run_$tag.log 2>&1 &
  echo "GPU $gpu -> $tag ($bk)"; }
wm(){ local gpu=$1 tag=$2
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((31000+gpu)) \
    scripts/train-beam-wm.py --data_dir $DATA --n_ant 32 --n_sub 32 --pred_only \
    --steps $STEPS --tag $tag > results/stress/run_$tag.log 2>&1 &
  echo "GPU $gpu -> $tag (single-layer-SSM WM pred-only)"; }
echo "==== GAP-CLOSE WAVE ($DATA / $SFX) ===="
pids=()
tp 0 deepssm     ds4_$SFX "--n_layers 4"; pids+=($!)
tp 1 deepssm     ds5_$SFX "--n_layers 5"; pids+=($!)
tp 2 deepssm     ds3_$SFX "--n_layers 3"; pids+=($!)
tp 3 transformer tf_$SFX  "";             pids+=($!)
tp 4 gru         gru_$SFX "";             pids+=($!)
tp 5 lstm        lstm_$SFX "";            pids+=($!)
wm  6 po_$SFX;                            pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo "GAP_CLOSE_DONE_$SFX" > results/temporal/gapclose_${SFX}.flag
echo GAP_CLOSE_WAVE_DONE
