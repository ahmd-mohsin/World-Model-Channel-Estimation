# 4. SelectiveSSM

`[x_t ; a_t]  ->  z_t`

The **temporal backbone**. A selective (Mamba-like) **diagonal** state-space model that rolls a
hidden state forward in time using the per-timestep parameters from `SelectionNet`, and emits the
latent `z_t`. This is the `A_t, B_t, C_t, Δ_t` block in `docs/sswm_fig.pdf`.

## Interface

| Field    | Shape                  | Notes                                   |
| -------- | ---------------------- | --------------------------------------- |
| `x`      | `(B, T, embed_dim)`    | context embedding from ContextEncoder   |
| `a`      | `(B, T, action_dim)`   | actions (concatenated as `[x_t; a_t]`)  |
| output   | `(B, T, latent_dim)`   | latent `z_t`                            |

```python
class SelectiveSSM(nn.Module):
    def __init__(self, config, selection_net=None): ...      # owns a SelectionNet (or shares one)
    def forward(self, x, a) -> Tensor: ...                    # full sequence (sequential scan)
    def step(self, x_t, a_t, h_prev) -> (z_t, h_t): ...       # single step, for Predictor rollout
```

## Design (diagonal SSM, ZOH discretization)

Per timestep, discretize the continuous diagonal system with step `Δ_t` (from SelectionNet):

```
Ā_t = exp(Δ_t · A_t)                 # elementwise; A_t < 0 => |Ā_t| < 1 (stable)
B̄_t = (Ā_t − 1) / A_t · B_t          # zero-order hold
h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t      # u_t = Linear([x_t ; a_t]) -> state_dim
z_t = out_proj( LN( C_t ⊙ h_t + D ⊙ u_t ) )
```

- **Sequential scan** (`for` loop over T): correctness first; a parallel/associative scan is a
  later optimization. Stability comes from `A_t < 0` (guaranteed by SelectionNet), so `Ā_t` is
  inside the unit circle for any action — verified to stay finite over T=200.
- `step()` reuses the *exact same* recurrence so the Predictor's rollout matches `forward`
  bit-for-bit (tested). It owns (or shares) a `SelectionNet`; pass one in to tie parameters.
- Input mixing `u_t = Linear([x_t ; a_t])`, skip term `D`, output `LayerNorm → Linear` to
  `latent_dim`.

## Tests (M2) — 8 passing

- Output shape `(B,T,latent_dim)`.
- **Scan matches a hand-computed reference recurrence** (constant params) to 1e-6.
- **`forward` == stepwise `step()` rollout** to 1e-5 (Predictor can reuse `step`).
- Stable & finite over T=200; discretized `|Ā| < 1` always.
- Gradients reach inputs, `in_proj`, and the SelectionNet.
- Selectivity: different actions change the output.
- Causality: perturbing `x` at t≥7 leaves z[:7] unchanged, changes z[7:].

```
pytest implementation/selective_ssm/test_selective_ssm.py -q   # 8 passed
```
