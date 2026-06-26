# 2. TargetEncoder

`o_{t+k}  ->  z̃_{t+k}`   (EMA weights, stop-gradient)

Produces the **target representation** the predictor is trained to match. Architecturally
identical to `ContextEncoder` (by construction — it is a deep copy), but its weights are an
**exponential moving average** of the context encoder's weights and it receives **no gradient**
(stop-grad). This is the standard JEPA/BYOL mechanism that prevents representation collapse.

## No pretrained download needed

There is no separate "pretrained target encoder" to fetch: in JEPA the target encoder *is* a
slow copy of the online encoder. The only pretrained component is the **shared frozen LWM
backbone**, which `ContextEncoder` already loads. `TargetEncoder` deep-copies that encoder
(LWM weights included) at construction, so it inherits the pretrained backbone for free.

## Interface

| Field   | Shape                        | Notes                              |
| ------- | ---------------------------- | ---------------------------------- |
| input   | `(B, T, 2, N_sub, N_ant)`    | future observation `o_{t+k}`       |
| output  | `(B, T, embed_dim)`          | target rep `z̃_{t+k}` (detached)   |

```python
class TargetEncoder(nn.Module):
    def __init__(self, source: ContextEncoder, config: SSWMConfig): ...
    @torch.no_grad()
    def forward(self, o: Tensor) -> Tensor: ...           # detached
    @torch.no_grad()
    def ema_update(self, source: ContextEncoder, m=None): ...
```

## Design

- **Init**: `deepcopy(ContextEncoder)`, then a hard sync of params + buffers from the source.
- **Frozen**: all params `requires_grad_(False)`; forward wrapped in `no_grad` and `.detach()`.
  `train()` is overridden to keep the inner encoder in `eval()` (stable BN/dropout-free targets).
- **EMA**: `θ_t ← m·θ_t + (1−m)·θ_c`, default `m = config.ema_momentum (0.996)`. **Only params
  that are trainable in the source are EMA-updated** — the frozen LWM backbone is identical in
  both encoders and never changes, so it is skipped (EMA touches just the ~99K head params, not
  the ~2.5M LWM weights). Buffers are copied straight through.

## Rigorous review outcome

Two real bugs were found by scrutiny (not by the original tests) and fixed in `ContextEncoder`:
- **Frozen-dropout bug:** `ctx.train()` recursively re-enabled the frozen LWM's `dropout=0.1`,
  making the encoder stochastic in train mode. Fixed: `train()` keeps frozen modules in `eval()`.
- **deepcopy-fragility bug:** `trainable_parameters()` used `id(p)` captured at init; after the
  target's `deepcopy`, ids change and the method misreported all params as trainable. Fixed:
  filter on `p.requires_grad` (EMA was already safe — it keys off the source's `requires_grad`).

Coordination is verified empirically (`verify_coordination.py`, fig `figures/05_coordination.png`):
self-distillation loss decreases, embedding std stays > 0 (no collapse), EMA gap converges.

## Tests (M1) — 6 unit + 9 coordination = 15 passing (26 incl. context encoder)

- Output shape `(B,T,embed_dim)` and `requires_grad == False`.
- All target params frozen.
- Hard init makes target output == source output.
- `ema_update` moves a trainable param to exactly `m·old + (1−m)·new`.
- Target stays grad-free after a source backward + EMA step.
- Frozen LWM backbone is untouched by EMA.

```
pytest implementation/target_encoder/test_target_encoder.py -q   # 6 passed
```
