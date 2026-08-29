#!/usr/bin/env bash
# Comprehensive benchmark campaign: sweep MOBILITY x PILOT-DENSITY x {in-dist, OOD} and train the
# world model + learned CNN baseline at each; classical baselines (LS, linear interp, oracle-MMSE,
# masked-MMSE, persistence, AR(1)) are computed at eval time. Produces per-scenario JSON that
# aggregate-benchmark.py rolls into the grand tables.
#
#   bash scripts/run-benchmark.sh           # runs the full grid on 8 GPUs, in waves (~2-3h)
#
# Mobility regimes (datasets must exist on box):
#   slow = data/mimo       (0.15m step, ~lambda)   -> low per-step channel change
#   fast = data/mimo_fast  (0.8m step,  ~5lambda)  -> high per-step channel change
# Outputs relocated into results/bench/<mob>/{stress,sparse}/ after each wave so nothing overwrites.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
export NANT=32 NSUB=32
STEPS=15000

relocate() {  # move current wave outputs into the per-mobility bench dir
  local mob=$1 kind=$2
  mkdir -p "results/bench/$mob/$kind"
  mv results/$kind/eval_*.json results/$kind/pred_*.json "results/bench/$mob/$kind/" 2>/dev/null || true
}

for MOB in slow fast; do
  DATA=data/mimo;      [ "$MOB" = fast ] && DATA=data/mimo_fast
  echo "############## MOBILITY=$MOB  ($DATA) ##############"

  # ---- Wave A: full-grid estimation (SNR sweep vs LS/MMSE) + prediction (horizon vs persist/AR1),
  #             3 in-dist seeds + 6 OOD leave-one-scene-out ----
  echo ">> [$MOB] Wave A: stress-matrix (full-grid est + prediction + OOD)"
  rm -f results/stress/eval_*.json results/stress/pred_*.json results/stress/run_*.log 2>/dev/null
  bash scripts/stress-matrix.sh "$DATA" "$STEPS"
  relocate "$MOB" stress

  # ---- Wave B: sparse-pilot — WM deep-fusion + ReEsNet, comb strides 2/4/8 (50/25/12.5% pilots)
  #             + OOD stride 4 & 8. Classical interp/masked-MMSE computed at eval. ----
  echo ">> [$MOB] Wave B: sparse (deep-fusion + ReEsNet, densities + OOD)"
  rm -f results/sparse/eval_*.json results/sparse/run_*.log 2>/dev/null
  bash scripts/mimo-sparse-wave.sh "$DATA"
  relocate "$MOB" sparse
done
echo "BENCHMARK_DONE"
