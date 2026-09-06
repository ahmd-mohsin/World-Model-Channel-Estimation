# Beam-WM — A Beamspace Selective State-Space World Model for MIMO-OFDM Channels

A **world model** for wireless channels that learns the channel's *dynamics* in a sparse
angle–delay (beamspace) representation and uses them for two tasks with **one** model:
**channel estimation** (recover `H` from noisy, possibly sparse pilots) and **channel prediction**
(forecast `H` several slots ahead for CSI-aging-robust precoding). The temporal backbone is a
Mamba-style **selective state-space model (SSM)** conditioned on receiver motion; the representation
is an orthonormal 2-D DFT (beamspace). Everything is validated on **ray-traced Sionna RT channels**,
true **MIMO 8×4**, across six scenes and two mobility regimes, on 8× A100 GPUs.

> **Design note (important):** earlier iterations used a JEPA latent-predictive objective and a frozen
> LWM foundation encoder. Both were **dropped**. A frozen LWM collapsed on real Sionna channels, and a
> latent-invariance objective is the wrong target for a model that must reconstruct a *faithful* future
> channel. The current model predicts the **actual future channel in beamspace** (reconstruction), and
> that prediction doubles as a temporal **prior** for estimation. This README describes what the code
> actually does today.

---

## 1. What this is, in one paragraph

A MIMO-OFDM receiver must estimate the channel `H` (Nt×Nr antennas × subcarriers) and, under mobility,
predict where it is going so the transmitter can precode for the channel that *will* exist. Classical
estimators (LS, MMSE) treat each snapshot independently and ignore temporal structure. We instead
transform the channel into the sparse **beamspace** (a 2-D DFT over spatial × subcarrier), encode each
frame, roll a **selective SSM** forward under the motion action to predict the future beamspace channel,
and **fuse** that prediction with the noisy observation through a learned Kalman-style gate to estimate
the current channel. One representation, one backbone, both tasks.

## 2. Headline findings (honest)

We benchmark against classical **and** matched-capacity **learned** baselines, and report negatives.

**Estimation — the strong pillar.**
- **Full-grid:** beats MMSE at every SNR (−5…30 dB), in-distribution and on all six held-out OOD scenes
  (42/42 cells). MMSE's fixed covariance breaks at high SNR; the σ²-conditioned gate does not.
- **Sparse pilots:** a ReEsNet-capacity deep-fusion head fed the world-model prior wins the
  sparse-pilot / mid-SNR band; a **learned SNR-band router** picks the better of {world model, CNN}
  in **13/14** regimes (vs 9/14 for a hand rule), never worse than the best fixed estimator.
- **Cross-config:** a **spatial-size-agnostic decoder** lets a model trained on **8×4** transfer
  **zero-shot to 16×8** (128 spatial dims) at NMSE parity, beating LS/MMSE at every SNR (MMSE collapses).
- **Baselines it beats in-band:** LS, linear interpolation, oracle-covariance MMSE, ReEsNet CNN, and a
  conditional **DDPM diffusion** estimator (which is only high-SNR-robust and ~30× slower).

**Prediction — competitive and unified, not dominant.**
- A **deep (3-layer) selective SSM is statistically tied** with a matched-capacity Transformer at
  channel prediction on both slow and fast mobility (3-seed means, gap ≪ seed std). A *single* SSM
  layer trails ~6–8% — depth, not attention, closes the gap.
- All learned models crush classical persistence / AR(1), by a margin that widens with horizon.
- Honest caveats: the SSM is **less data-efficient** than attention (−9% at 6–12k sequences), and
  predictors are **mobility-specific** (57–97% degradation transferring across speed regimes).

**Downstream (predict-then-precode throughput).** Under CSI aging, prediction recovers up to ~24% of the
aging-induced spectral-efficiency loss at fast mobility / long horizon / low-mid SNR; under slow mobility
stale CSI is near-optimal and prediction is net counterproductive. A clean **crossover law**: a temporal
prior helps only in a specific mobility × horizon × SNR × pilot-density regime.

**The contribution is unification + characterization:** one ~7M-parameter model does estimation *and*
prediction, each within a few percent of a task-specialized specialist, at the parameter cost of one —
plus a precise map of *when* channel dynamics are worth their cost.

## 3. Architecture

```
o_t (noisy channel)  ──2-D DFT──▶  beam_t  ──BeamEncoder(conv)──▶  x_t
                                                                    │
                        a_t (motion action) ──SelectionNet──▶ (A,B,C,Δ)
                                                                    ▼
                                          x_t, a_t ──selective SSM──▶ (z_t, h_t)
                                                                    │  roll k steps under planned actions
                                                                    ▼
                                   BeamDecoder ◀── predictor ──▶  b̂_{t+k}   [reconstruction loss]
                                                                    │
   estimate:  b̂_t (history prior) + noisy beam obs + clean t-1 persistence
              ──learned Kalman gate──▶ beam_est_t ──inv DFT──▶  Ĥ_t        [channel loss]
```

- **Beamspace transform** (`implementation/wireless_data/beamspace.py`): orthonormal, invertible 2-D
  DFT; a separable 3-D angle-angle-delay variant is available (`beam3d`, tested — does not help
  estimation, kept for completeness).
- **Selective SSM** (`selection_net/`, `selective_ssm/`): diagonal, ZOH-discretized, HiPPO/S4-style
  init, action-conditioned params. A deep stacked variant lives in `task_heads/ssm_predictor.py`.
- **Persistence-augmented fusion head**: the estimation gate sees {obs, learned prior, clean t-1
  persistence} — a strict superset of every competitor, so it cannot lose to persistence by construction.
- **Spatial-agnostic decoder**: latent broadcast over the grid + coordinate channels, so one model
  decodes any (antenna, subcarrier) grid → cross-config transfer.

## 4. Repository structure

```
implementation/
  beam_wm/beam_world_model.py     # THE model: beamspace encode → selective SSM → predict → fuse/estimate
  selection_net/ selective_ssm/   # action-conditioned SSM params + ZOH scan
  context_encoder/ target_encoder/ predictor/   # legacy JEPA-era modules (superseded; kept for history)
  wireless_data/                  # Sionna dataset loaders + beamspace transforms (2-D and 3-D)
  task_heads/
    baselines.py                  # LS, MMSE, linear interpolation
    deep_baseline.py              # ReEsNet-style CNN + a-priori CSI router
    ssm_predictor.py              # deep stacked selective-SSM predictor (matched-capacity)
    temporal_baseline.py          # GRU / LSTM / Transformer predictors (matched-capacity)
    diffusion_estimator.py        # conditional DDPM channel estimator
    task_heads.py, unet_head.py   # estimation heads
  config.py                       # SSWMConfig (dims, SSM hyperparams)
  sswm.py, test_*.py              # full-model wiring + tests
scripts/
  gen_sionna_mimo.py, gen-mimo*.sh    # ray-traced MIMO data generation (8×4, 16×8, slow/fast)
  train-beam-wm.py                     # world model: full-grid + prediction (--pred_only, --beam3d, --agnostic_decoder)
  train-beam-sparse.py                 # sparse-pilot deep-fusion estimator
  train-deep-baseline.py               # ReEsNet baseline
  train-temporal-predictor.py          # GRU/LSTM/Transformer/deepssm predictors
  train-diffusion-estimator.py         # DDPM estimator
  gap-close-wave.sh sig-wave.sh arch-wave.sh temporal-wave.sh   # multi-GPU experiment launchers
  throughput-aging.py spectral-efficiency.py   # predict-then-precode / SVD-precoding throughput
  xconfig-eval.py xeval-predictor.py           # cross-config & cross-mobility generalization evals
  greenland-*.sh                       # 8×A100 box: auth, SSM tunnel, rsync
dashboard/                             # live HTTP training dashboard (Sionna 3D scene + loss curves)
docs/paper.tex, paper.pdf              # the paper (all results below)
results/                               # per-experiment NMSE / throughput JSONs
```

## 5. Results at a glance

| Task / experiment | Result |
|---|---|
| Full-grid estimation vs MMSE | wins every SNR, in-dist + 42/42 OOD |
| Sparse-pilot + learned router | 13/14 regimes correct; +2% over hand rule, never worse than best fixed |
| Cross-config 8×4 → 16×8 (zero-shot) | NMSE parity, beats LS/MMSE every SNR |
| Diffusion (DDPM) estimator | legit but loses to WM everywhere, 30× slower |
| Prediction: deep-SSM vs Transformer | statistically tied, both mobilities (3 seeds) |
| Prediction: single-layer SSM | trails 6–8% (depth is the enabler) |
| Data-efficiency (SSM vs Transformer) | SSM −9% at 6–12k, ties at 24k |
| Cross-mobility transfer | 57–97% degradation (both models) |
| Predict-then-precode throughput | +≤24% aging recovery fast/long-k/low-SNR; net-negative slow |

Raw numbers per experiment are in `results/` and written up in `docs/paper.tex`.

## 6. Data — real ray-traced channels (Sionna RT)

Sionna RT ray tracing at 3.5 GHz over six scenes (munich, etoile, florence, san_francisco,
simple_street_canyon, simple_street_canyon_with_cars). Each sequence is a receiver walking a straight
trajectory; the action is its velocity. Two mobility regimes: **slow** (≈λ step) and **fast** (≈5λ step).
MIMO 8×4 = 32 folded spatial × 32 subcarriers; a 16×8 (128 spatial) set is generated for cross-config.

```bash
bash scripts/gen-mimo.sh 3000        # slow, 8 scenes × 3000 = 24k sequences (8 GPUs)
bash scripts/gen-mimo-fast.sh 3000   # fast mobility
```

## 7. How to run

```bash
# world model (full-grid estimation + prediction)
torchrun --nproc_per_node=1 scripts/train-beam-wm.py --data_dir data/mimo --n_ant 32 --n_sub 32 --tag est
# prediction-only (fair vs learned temporal baselines)
torchrun ... scripts/train-beam-wm.py --data_dir data/mimo --pred_only --tag po_slow
# matched-capacity predictors: GRU / LSTM / Transformer / deep selective-SSM
torchrun ... scripts/train-temporal-predictor.py --data_dir data/mimo --backbone deepssm --n_layers 3 --tag ds3
# sparse-pilot deep-fusion estimator + ReEsNet baseline
torchrun ... scripts/train-beam-sparse.py --data_dir data/mimo --deep_fusion --stride 8 --tag sp8
torchrun ... scripts/train-deep-baseline.py --data_dir data/mimo --stride 8 --tag deep8
# downstream + generalization
python scripts/throughput-aging.py  --wm_ckpt ... --tf_ckpt ... --ds_ckpt ... --data data/mimo_fast
python scripts/xconfig-eval.py      --ckpt implementation/checkpoints/beam_est_agn.pt --data8 data/mimo --data16 data/mimo16x8
```

Multi-GPU experiment waves (8× A100): `scripts/gap-close-wave.sh` (prediction), `scripts/arch-wave.sh`
(estimation + 3-D + router inputs), `scripts/sig-wave.sh` (multi-seed).

## 8. Compute

Runs on a Greenland EKS `p4d.24xlarge` (8× A100). `scripts/greenland-*.sh` handle Isengard auth, the SSM
port-forward tunnel, and rsync. Data is regenerated on the box (Sionna RT is GPU-accelerated); only small
result JSONs and checkpoints are pulled back.

## 9. Status

All six modules built and validated; the paper (`docs/paper.tex`) compiles and reports every result
above — positives and negatives. The estimation framework is never worse than the best baseline in any
tested regime; the prediction claim is "deep selective-SSM competitive with attention, unified with
estimation," with the crossover law delineating when a learned dynamics model is worth its cost.
