#!/usr/bin/env bash
# Architecture-experiment wave on current 24k data:
#  - 2-D vs 3-D beamspace full-grid estimator (limitation 5)
#  - sparse deep-fusion WM + ReEsNet at strides 8 & 4 (inputs for the LEARNED ROUTER, limitation 3)
#   bash scripts/arch-wave.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source wireless/bin/activate
D=data/mimo; S=15000
wm(){ CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=1 --master_port=$((35000+$1)) \
  scripts/train-beam-wm.py --data_dir $D --n_ant 32 --n_sub 32 $3 --steps $S --tag $2 \
  > results/stress/run_$2.log 2>&1 & echo "GPU $1 -> $2"; }
sp(){ CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=1 --master_port=$((35000+$1)) \
  scripts/train-beam-sparse.py --data_dir $D --n_ant 32 --n_sub 32 --deep_fusion --stride $3 --steps $S --tag $2 \
  > results/sparse/run_$2.log 2>&1 & echo "GPU $1 -> $2 (sparse s$3)"; }
dp(){ CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node=1 --master_port=$((35000+$1)) \
  scripts/train-deep-baseline.py --data_dir $D --n_ant 32 --n_sub 32 --stride $3 --steps $S --tag $2 \
  > results/sparse/run_$2.log 2>&1 & echo "GPU $1 -> $2 (ReEsNet s$3)"; }
mkdir -p results/stress results/sparse
echo "==== ARCH WAVE ===="
pids=()
wm 0 est2d "";        pids+=($!)
wm 1 est3d "--beam3d"; pids+=($!)
sp 2 sp8  8;          pids+=($!)
dp 3 deep8 8;         pids+=($!)
sp 4 sp4  4;          pids+=($!)
dp 5 deep4 4;         pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo ARCH_WAVE_DONE > results/stress/arch_done.flag
echo ARCH_WAVE_DONE
