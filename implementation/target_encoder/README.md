# 2. TargetEncoder

`o_{t+k}  ->  z̃_{t+k}`   (EMA weights, stop-gradient)

Produces the **target representation** the predictor is trained to match. Architecturally
identical to `ContextEncoder`, but its weights are an **exponential moving average** of the
context encoder's weights and it receives **no gradient** (stop-grad). This is the standard
JEPA/BYOL mechanism that prevents representation collapse.

## Interface

| Field   | Shape                 | Notes                              |
| ------- | --------------------- | ---------------------------------- |
| input   | `(B, T, obs_dim)`     | future observation `o_{t+k}`       |
| output  | `(B, T, embed_dim)`   | target rep `z̃_{t+k}` (detached)   |

```python
class TargetEncoder(nn.Module):
    @torch.no_grad()
    def forward(self, o: Tensor) -> Tensor: ...

    @torch.no_grad()
    def ema_update(self, source: ContextEncoder, m: float = 0.996): ...
```

## Baseline design

- Initialized as a hard copy of `ContextEncoder` (`load_state_dict`).
- `requires_grad_(False)` on all params; forward wrapped in `torch.no_grad()`.
- `ema_update`: `θ_t ← m·θ_t + (1−m)·θ_c` called once per training step.

## Test (M1)

- After an optimizer step on the context encoder, target params changed by EMA only.
- Output tensor has `requires_grad == False`.
