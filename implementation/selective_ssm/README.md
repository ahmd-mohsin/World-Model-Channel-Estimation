# 4. SelectiveSSM

`[x_t ; a_t]  ->  z_t`

The **temporal backbone**. A selective (Mamba-like) **diagonal** state-space model that
rolls a hidden state forward in time using the per-timestep parameters from `SelectionNet`,
and emits the latent representation `z_t`. This is the box labelled `A_t, B_t, C_t, Δ_t`
in the figure.

## Interface

| Field    | Shape                          | Notes                                    |
| -------- | ------------------------------ | ---------------------------------------- |
| `x`      | `(B, T, embed_dim)`            | context embedding from ContextEncoder    |
| `a`      | `(B, T, action_dim)`           | actions (concatenated: `[x_t; a_t]`)     |
| params   | from SelectionNet              | `A_t, B_t, C_t, Δ_t`                      |
| output   | `(B, T, latent_dim)`           | latent `z_t`                             |

```python
class SelectiveSSM(nn.Module):
    def forward(self, x: Tensor, a: Tensor) -> Tensor: ...
    def step(self, x_t, a_t, h_prev) -> tuple[z_t, h_t]: ...   # single step, for rollout
```

## Baseline design (diagonal SSM, ZOH discretization)

For each timestep, discretize the continuous system with step `Δ_t`:

```
Ā_t = exp(Δ_t · A_t)                  # elementwise, A_t diagonal
B̄_t = (Ā_t − 1) / A_t · B_t           # ZOH
h_t  = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t      # u_t = projection of [x_t; a_t]
z_t  = C_t ⊙ h_t  (+ D ⊙ u_t)         # readout
```

- **Baseline runs this as a sequential scan** (a plain Python/`for` loop over `T`).
  Correctness first; a parallel/associative scan is a later optimization.
- `step()` is exposed so the `Predictor` can reuse the exact same recurrence for rollout.
- Input mixing: `u_t = Linear([x_t ; a_t])` to `state_dim`.

## Test (M2)

- With constant `A,B,C,Δ`, output matches a hand-computed linear recurrence reference.
- State stays finite over long `T` (stability from `A<0`).
- Gradients flow through the scan to inputs and SelectionNet.
