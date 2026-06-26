# 5. Predictor

`(z_t, a_t, ..., a_{t+k-1})  ->  ẑ_{t+k}`

Predicts the **future latent representation** `k` steps ahead from the current latent `z_t`
and a window of **planned actions**. This is the imagination/rollout step of the world model.
Its output is compared against the target encoder's `z̃_{t+k}` by the JEPA loss.

## Interface

| Field           | Shape                          | Notes                              |
| --------------- | ------------------------------ | ---------------------------------- |
| `z_t`           | `(B, latent_dim)`              | current latent from SelectiveSSM   |
| `planned_acts`  | `(B, k, action_dim)`           | `a_t … a_{t+k-1}`                  |
| output          | `(B, latent_dim)`              | predicted future rep `ẑ_{t+k}`     |

```python
class Predictor(nn.Module):
    def forward(self, z_t: Tensor, planned_acts: Tensor) -> Tensor: ...
```

## Baseline design

- **Rollout variant (preferred):** reuse `SelectiveSSM.step()` `k` times, feeding the
  planned actions, then a small projection head → `latent_dim`. Keeps dynamics consistent
  with the encoder path.
- **Simple variant (fastest to stand up):** MLP/GRU that ingests `z_t` and the flattened
  action window and regresses `ẑ_{t+k}` directly.
- Start with the simple variant for M3, swap to rollout once M2 SSM is verified.

## Test (M3)

- Output shape == `(B, latent_dim)`.
- On a toy AR(1) latent sequence, predicted `ẑ_{t+k}` tracks the true future latent
  (loss decreases during overfitting on one batch).
