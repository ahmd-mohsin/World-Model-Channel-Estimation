#!/usr/bin/env bash
# Module-ablation wave: is the world-model prior load-bearing for estimation?
#   bash scripts/ablation-wave.sh [DATA_DIR] [STEPS]
#
# 8 runs / 8 GPUs, one wave:
#   GPU 0-4: in-distribution, ablate in {full, prior, action, ssm, beamspace}
#   GPU 5-7: OOD (held-out simple_street_canyon), ablate in {full, prior, ssm}
#            -> tests whether the prior helps MORE out-of-distribution (where temporal info matters)
# Each run emits eval_ab_*.json (estimation) + pred_ab_*.json (prediction).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate

DATA="${1:-data/stress6}"
STEPS="${2:-15000}"
OODSCENE="simple_street_canyon"
mkdir -p results/ablation dashboard

# run_one GPU TAG ABLATE [holdout]
run_one() {
  local gpu=$1 tag=$2 ab=$3 holdout=${4:-}
  local args="--data_dir $DATA --steps $STEPS --bs 96 --lr 3e-4 --tag $tag --ablate $ab --seed 0 --n_ant ${NANT:-8} --n_sub ${NSUB:-32}"
  [ -n "$holdout" ] && args="$args --holdout_scene $holdout"
  CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29600 + gpu)) \
    scripts/train-beam-wm.py $args \
    > "results/ablation/run_${tag}.log" 2>&1 &
  echo "GPU $gpu -> $tag (ablate=$ab holdout=${holdout:-none}) pid $!"
}

echo "==== ABLATION WAVE ===="
pids=()
run_one 0 ab_indist_full      full      "" ; pids+=($!)
run_one 1 ab_indist_prior     prior     "" ; pids+=($!)
run_one 2 ab_indist_action    action    "" ; pids+=($!)
run_one 3 ab_indist_ssm       ssm       "" ; pids+=($!)
run_one 4 ab_indist_beamspace beamspace "" ; pids+=($!)
run_one 5 ab_ood_full         full      "$OODSCENE"; pids+=($!)
run_one 6 ab_ood_prior        prior     "$OODSCENE"; pids+=($!)
run_one 7 ab_ood_ssm          ssm       "$OODSCENE"; pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "ABLATION_WAVE_DONE (fail=$fail)"
