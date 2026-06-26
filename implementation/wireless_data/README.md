# Wireless data (Sionna RT)

Real ray-traced MIMO-OFDM channel sequences for the SSWM pipeline. **Sionna only — no
synthetic fallback.** Every module is exercised on real propagation channels.

## What it produces
- `WirelessDataset(config, n_samples, spec, seed)` → observation tensors shaped
  `(seq_len, 2, n_antennas, n_subcarriers)` (real/imag planes), `.batch(B)` adds the batch dim.
- `SionnaChannelGenerator` ray-traces a Sionna scene and computes the OFDM channel frequency
  response (`Paths.cfr`) for a receiver moved along a trajectory, giving temporally-correlated
  sequences (the smooth channel evolution a world model predicts).

## Key design points
- **Sub-wavelength steps**: `SionnaSpec.step_size_m = 0.015 m` (λ @ 3.5 GHz ≈ 8.6 cm). Larger
  steps fully decorrelate the channel between frames; sub-wavelength keeps adjacent frames
  correlated. Verified by `test_temporal_correlation`.
- **Receiver position** is set as a `np.float32` array (Mitsuba `Point3f` rejects python lists
  of numpy scalars).
- Sionna's `PathSolver` has tiny Monte-Carlo ray-sampling noise (~1e-3), so repeats are
  *near*-identical, not bit-exact (`test_repeatable_per_index` uses a tolerance).

## Requirements
`sionna` + `sionna-rt` (modern Dr.Jit/Mitsuba backend, no TensorFlow). Installed on the
Greenland box; runs on the A100. Tests `skip` where Sionna is absent (e.g. laptop).

## Tests — 6 passing on the box (real Sionna)
shape, normalized magnitude, repeatability, batch shape, temporal correlation, feeds the
ContextEncoder.

```
pytest implementation/wireless_data/test_wireless_data.py -q   # 6 passed (box)
```

## Normalization: match LWM's training scale (resolved)
LWM was trained on raw DeepMIMO channels scaled by `1e6` (`deepmimo_data_cleaning:
channel * 1e6`), which keeps the natural cross-antenna/subcarrier **amplitude variation**.
Our first version used Sionna's `normalize=True` + per-sample max-normalization, which threw
away absolute scale — and a scaling sweep (`scripts/validate-on-sionna.py`) measured that this
made the frozen LWM features **~32× less discriminative** than LWM's native `×1e6` scale.

**Fix:** use Sionna's RAW (`normalize=False`) CFR and apply a fixed global `channel_scale=1e6`.
Result on the box, comparing channels from two genuinely different receiver locations:

| input scaling          | feature separation (RMS) | vs noise floor |
| ---------------------- | ------------------------ | -------------- |
| per-sample max-norm    | 0.002                    | ~545×          |
| **×1e6 (LWM native)**  | **0.064**                | **~21,000×**   |

With the fix, ContextEncoder `x_t` std across a batch rose 0.0003 → 0.017, and the frozen LWM
cleanly distinguishes channels from different locations (separation ratio ~21,000×). The
earlier "near-identical embeddings" reading was an artifact of (a) wrong normalization **and**
(b) comparing sub-wavelength trajectory steps that are genuinely near-identical by design.

**Caveat that still stands:** the projection head is still randomly initialized — it only
becomes task-meaningful after end-to-end JEPA training. But the *frozen backbone now produces
discriminative features on real Sionna data*, which is what Option 1 set out to verify.
