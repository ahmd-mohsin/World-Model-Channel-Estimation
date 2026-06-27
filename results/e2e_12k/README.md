# End-to-end SSWM run — 12k Sionna dataset

Full joint training of all 6 modules (LoRA-frozen LWM), 4×A100 DDP, 12,000 channel sequences
across 5 scenes, 12,000 steps. Pulled from the Greenland box on 2026-06-27.

## Files
- `metrics.json` — per-step training metrics (total + JEPA/VICReg/channel losses, stds, lr). 61 points.
- `eval.json` — held-out evaluation (world-model predictor + channel-estimation NMSE vs SNR).
- `scenes.json` — 3D scene geometry + channel statistics for the dashboard.
- `sswm_e2e.pt` — trained model checkpoint (25.7 MB, gitignored).
- `train.log` — full training log.

## Headline results (held-out)

**Channel estimation (NMSE, lower better):**

| SNR | LS | MMSE | SSWM |
| --- | ---- | ---- | ---- |
| 0 dB  | 1.002 | 0.037 | 0.172 |
| 5 dB  | 0.317 | 0.016 | 0.041 |
| 10 dB | 0.100 | 0.008 | 0.013 |
| 15 dB | 0.032 | 0.004 | 0.0045 |
| 20 dB | 0.010 | 0.0026 | **0.0018** |

SSWM **beats MMSE at 20 dB** and is tied at 15 dB; beats LS 5–8× across the board. It is *not*
given the channel covariance or noise variance that MMSE uses. Low-SNR (0 dB) still trails MMSE —
expected to improve with the 100k dataset.

**World-model predictor:** NMSE 0.0503 vs persistence 0.0718 = **1.43×**.

**Convergence:** total loss 0.62 → 0.096; channel NMSE 0.099 → 0.012; no representation collapse.

## View
```bash
python3 dashboard/serve.py 8000   # then copy these json files into dashboard/, open localhost:8000
```
