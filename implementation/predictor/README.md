# 5. Predictor

`(z_t, a_t, ..., a_{t+k-1})  ->  ẑ_{t+k}`

Predicts the **future latent representation** `k` steps ahead from the current latent `z_t` and a
window of **planned actions**. This is the imagination/rollout step of the world model. Its output
is compared against the target encoder's `z̃_{t+k}` by the JEPA loss.

## Interface

| Field          | Shape                  | Notes                              |
| -------------- | ---------------------- | ---------------------------------- |
| `z_t`          | `(B, latent_dim)`      | current latent from SelectiveSSM   |
| `planned_acts` | `(B, k, action_dim)`   | `a_t … a_{t+k-1}`                  |
| output         | `(B, embed_dim)`       | predicted future rep `ẑ_{t+k}`     |

```python
class Predictor(nn.Module):
    def forward(self, z_t, planned_acts) -> Tensor: ...      # ẑ at horizon k
    def rollout(self, z_t, planned_acts) -> Tensor: ...      # ẑ at every 1..k, (B,k,embed_dim)
```

## Design — and two mistakes we deliberately avoided

**1. Observation-free rollout (no information leak).** The README's literal "reuse
`SelectiveSSM.step()`" is a trap: `step(x_t, a_t, h)` needs the *observation embedding* `x_t`, but
**the future is not observed** during prediction. Feeding future `x` would be a catastrophic leak
(the model would cheat by seeing the answer). So the Predictor rolls forward using the **planned
actions only**, with `z_t` mapped to the initial recurrent state. It reuses the SSM's *dynamics
primitives* (ZOH `discretize`, the shared SelectionNet) but not its observation-driven `step`.

**2. Output lives in `embed_dim`, not `latent_dim`.** The JEPA loss compares `ẑ_{t+k}` to
`z̃_{t+k} = target_encoder(o_{t+k})`, which is in **encoder-output space (`embed_dim`)**. The
predictor therefore projects to `embed_dim`. These are equal by default (256), so a silent bug
could hide if they ever differ — `test_output_in_embed_space_not_latent` sets them unequal to
guard this.

Rollout recurrence (per planned step, `A,B,C,Δ` from the shared SelectionNet on action `a_j`):
```
h_0 = z_to_state(z_t)
Ā,B̄ = discretize(A_j, B_j, Δ_j);  u = act_proj(a_j)
h_j = Ā ⊙ h_{j-1} + B̄ ⊙ u
ẑ   = out_proj( LN( C_j ⊙ h_j + D ⊙ u ) )      # at j = k
```

## Tests (M3) — 11 passing
Shape; **embed-space (not latent) output**; `forward == rollout()[:, -1]`; depends on both `z_t`
and actions; horizon changes the prediction; **uses only the action prefix (no future-action
leak)**; `k=0` edge case; gradient flow into the shared SelectionNet; stability over horizon 100;
overfits an AR(1)-style target (loss halves).

## Cross-check on real Sionna channels (`scripts/validate-predictor-sionna.py`)
Trained the full SSWM JEPA objective on 256 real Sionna sequences (600 steps, A100), then
compared the predictor against trivial baselines in the same embedding space:

| predictor | persistence (copy present) | batch-mean |
| --------- | -------------------------- | ---------- |
| **NMSE 0.0077** | 0.0237 (3.1× worse) | 0.0108 (1.4× worse) |

The predictor **beats persistence 3.1×** — a world model that hadn't learned dynamics would only
tie persistence (just echo `z_t`). It also beats batch-mean, so it isn't collapsing to the mean.

*Caveat (honest):* this is a small-scale sanity run on one scene; embedding std is modest
(pred ≈ 0.012, target ≈ 0.024) and the JEPA loss is very low, so partial-collapse pressure exists.
The nonzero std + beating batch-mean argue against full collapse, but scaling data diversity and
adding the VICReg regularizer to the full objective is future work, not a closed result.
