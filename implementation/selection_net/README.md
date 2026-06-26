# 3. SelectionNet

`a_t  ->  (A_t, B_t, C_t, Δ_t)`

Generates the **input-dependent (selective) parameters** of the SSM block. This is what
makes the SSM *selective* (Mamba-style): the state-space dynamics are conditioned on the
action `a_t` rather than being fixed in time.

## Interface

| Output | Shape                        | Constraint                          |
| ------ | ---------------------------- | ----------------------------------- |
| `A_t`  | `(B, T, state_dim)`          | negative (stable): `A = -softplus`  |
| `B_t`  | `(B, T, state_dim)`          | —                                   |
| `C_t`  | `(B, T, state_dim)`          | —                                   |
| `Δ_t`  | `(B, T, 1)` or `(B,T,state)` | positive: `softplus`                |

```python
class SelectionNet(nn.Module):
    def forward(self, a: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # a: (B,T,action_dim) -> (A_t, B_t, C_t, Δ_t)
```

## Baseline design

- 2-layer MLP from `action_dim` → hidden → params, split into A/B/C/Δ heads.
- `A_t = -softplus(headA)` keeps the continuous-time pole stable.
- `Δ_t = softplus(headΔ)` keeps the discretization step positive.
- Diagonal state space ⇒ A/B/C are vectors of size `state_dim`, not matrices.

## Test (M2)

- `A_t < 0` everywhere, `Δ_t > 0` everywhere.
- Output shapes match `state_dim`.
- Gradients reach SelectionNet from a downstream loss.
