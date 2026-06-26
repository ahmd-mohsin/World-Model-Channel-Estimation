# 6. TaskHeads

`z_t  ->  reward / policy / channel estimate / ...`

Downstream readouts that consume the learned latent `z_t` (or `ẑ_{t+k}`). In the figure
these are the "task heads" producing reward, policy, etc. For channel estimation the
primary head reconstructs/predicts the channel and reports NMSE against classical baselines.

## Interface

| Head              | Output shape          | Purpose                                   |
| ----------------- | --------------------- | ----------------------------------------- |
| `channel_head`    | `(B, obs_dim)`        | reconstruct/predict channel (NMSE eval)   |
| `reward_head`     | `(B, 1)`              | reward prediction (RL/control)            |
| `policy_head`     | `(B, action_dim)`     | action distribution (optional)            |

```python
class TaskHeads(nn.Module):
    def forward(self, z: Tensor) -> dict[str, Tensor]: ...
```

## Baseline design

- Each head is a small MLP probe on top of `z`.
- **Heads are trained separately / as linear probes** in the baseline so they don't
  interfere with the self-supervised world-model objective. This isolates "how good is the
  representation" from "how good is the head."
- Channel head is the headline metric: **NMSE vs. LS and MMSE** estimators.

## Test (M5)

- `channel_head` NMSE on synthetic data is reported and compared to LS/MMSE.
- Heads train without backpropagating into the frozen encoder (probe setting).
