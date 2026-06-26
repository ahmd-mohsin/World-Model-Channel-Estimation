# 1. ContextEncoder

`o_t  ->  x_t`

The **online encoder**. Maps a current observation (received pilot / channel measurement)
into the latent embedding `x_t` that feeds the selective SSM. Trained by backpropagation
from the JEPA loss.

## Interface

| Field   | Shape                         | Notes                                  |
| ------- | ----------------------------- | -------------------------------------- |
| input   | `(B, T, obs_dim)`             | obs_dim = 2·N_subcarriers (real+imag)  |
| output  | `(B, T, embed_dim)`           | embedding `x_t`                        |

```python
class ContextEncoder(nn.Module):
    def forward(self, o: Tensor) -> Tensor:  # (B,T,obs_dim) -> (B,T,embed_dim)
```

## Baseline design

- LayerNorm → 1-D conv (across subcarriers) → MLP → `embed_dim` projection.
- No temporal mixing here — that is the SSM's job. Encoder is per-timestep.
- Shares architecture with `TargetEncoder`; weights are coupled via EMA in the trainer.

## Test (M1)

- Output shape == `(B, T, embed_dim)`.
- Gradients are non-`None` (this encoder *is* trained).
