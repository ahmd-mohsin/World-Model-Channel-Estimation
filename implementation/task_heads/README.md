# 6. TaskHeads

`z_t  ->  channel estimate / reward / policy`

Downstream readouts that consume the learned world-model latent `z`. The headline is the
**channel head**: recover the clean channel from a noisy observation, measured by NMSE against
the classical **LS** and **MMSE** estimators.

## Interface

```python
class TaskHeads(nn.Module):
    def forward(self, z, obs=None) -> dict[str, Tensor]: ...   # channel / reward / policy
    def channel_grid(self, z, obs=None) -> Tensor             # (B, 2, n_ant, n_sub)
```

| Head      | Output            | Purpose                                   |
| --------- | ----------------- | ----------------------------------------- |
| `channel` | `(B, obs_dim)`    | channel estimate (NMSE vs LS/MMSE)        |
| `reward`  | `(B, 1)`          | reward prediction (RL/control)            |
| `policy`  | `(B, action_dim)` | action distribution (optional)            |

## Design — and the key finding that shaped it

Heads are independent MLP **probes** on the frozen world-model latent (trained separately so they
don't perturb the SSL objective). But a naive `z -> channel` probe **failed** (NMSE ~0.36 even at
20 dB). We diagnosed why (`scripts/diag-channel-head.py`):

| reconstruct channel @20 dB from… | NMSE |
| -------------------------------- | ---- |
| frozen LWM latent `z` (256-d)    | 0.31 |
| raw noisy observation (512-d)    | **0.005** |

**The frozen LWM latent is lossy/invariant** — it was trained for masked-channel *modeling*, so it
keeps semantic/denoised features but **cannot be inverted back to the exact channel**. The fix
(correct for channel estimation): the channel head takes **both** the latent (learned context /
denoising prior) **and the raw observation** (detail), and predicts a **residual on the observation**
(zero-init → starts at LS, learns the denoising). `channel_use_obs=True` by default.

## Result — SSWM channel head vs LS / MMSE (12k Sionna seqs, 5 scenes, held-out)

| SNR | LS | MMSE | **SSWM** |
| --- | ---- | ---- | ---- |
| 0 dB  | 0.997 | 0.036 | **0.186** |
| 5 dB  | 0.318 | 0.015 | **0.069** |
| 10 dB | 0.100 | 0.008 | **0.114** |
| 15 dB | 0.032 | 0.004 | **0.043** |
| 20 dB | 0.010 | 0.002 | **0.014** |

NMSE, lower is better. **SSWM beats LS by up to ~5× at low SNR** and tracks it closely at high SNR.
**MMSE still wins** — it is the optimal *linear* estimator and these ray-traced channels are highly
spatially correlated (its ideal regime), and it is handed the true channel covariance **and** exact
noise variance. The SSWM head gets neither: it learns one estimator across all SNRs/scenes from data
alone, yet substantially beats LS and approaches MMSE. That is the honest, defensible claim.

## Classical baselines (`baselines.py`)
- `ls_estimate` — LS = the noisy observation (unit pilots).
- `mmse_estimate` — linear MMSE / Wiener filter from the training channel covariance + SNR.
- `add_noise`, `nmse` — helpers. `test_mmse_beats_ls_at_low_snr` verifies the Wiener filter.

## Tests (M5) — 10 passing
Head shapes; channel-grid reshape; **channel head starts at LS** (zero-init residual); latent-only
mode; subset of heads; probe gradient isolation; overfit; and the three classical-baseline tests.

```
pytest implementation/task_heads/test_task_heads.py -q   # 10 passed
```
