# 3. SelectionNet

`a_t  ->  (A_t, B_t, C_t, Δ_t)`

Generates the **input-dependent (selective) parameters** of the diagonal SSM block. This is
what makes the SSM *selective* (Mamba-style): the state-space dynamics are conditioned on the
action `a_t` rather than being fixed in time. All four parameters are action-dependent, exactly
as drawn in `docs/sswm_fig.pdf` (SelectionNet → `A_t, B_t, C_t, Δ_t`).

## Interface

| Output | Shape                  | Constraint                                |
| ------ | ---------------------- | ----------------------------------------- |
| `A_t`  | `(B, T, state_dim)`    | strictly negative (stable): `A = -softplus` |
| `B_t`  | `(B, T, state_dim)`    | unconstrained                             |
| `C_t`  | `(B, T, state_dim)`    | unconstrained                             |
| `Δ_t`  | `(B, T, state_dim)`    | strictly positive: `softplus`             |

Also accepts a flat `(B, action_dim)` and returns `(B, state_dim)` tensors.

```python
class SelectionNet(nn.Module):
    def forward(self, a: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # a: (B,T,action_dim) -> (A_t, B_t, C_t, Δ_t)
```

## Design

- **Trunk**: `Linear → LayerNorm → GELU → Linear → GELU` from `action_dim` to `selection_hidden`,
  then four linear heads (A, B, C, Δ). Diagonal state space ⇒ each is a length-`state_dim` vector.
- **Stability**: `A_t = -softplus(head_A(h))` guarantees negative continuous-time poles, so the
  discretized recurrence `exp(Δ·A)` stays inside the unit circle for any action.
- **Positivity**: `Δ_t = softplus(head_dt(h))` keeps the discretization step positive.
- **Mamba/S4-style init** (this is what makes the module strong standalone, not a naive MLP):
  - `Δ` bias is set via the inverse-softplus trick so that at init (zero action) `Δ` is
    **log-uniform in `[dt_min, dt_max]`** — a spread of timescales.
  - `A` bias targets `-1, -2, …, -state_dim` (inverse-softplus), giving a HiPPO-like spread of
    decay rates so different state channels capture different temporal horizons.
  - The A/Δ head **weights** are small but non-zero (`std=1e-3`), so the parameters remain genuinely
    action-selective while the bias controls the init range.

## Tests (M2) — 10 passing

- Output shapes `(B,T,state_dim)` for all four params; flat `(B, action_dim)` input supported.
- `A_t < 0` everywhere; `Δ_t > 0` everywhere.
- `Δ` init lands in `[dt_min, dt_max]`; `A` init shows a spread of timescales.
- **Selectivity**: different actions produce different A, B, C, Δ.
- Gradients reach every parameter and are finite; deterministic in eval; no NaNs on extreme actions.

```
pytest implementation/selection_net/test_selection_net.py -q   # 10 passed
```
