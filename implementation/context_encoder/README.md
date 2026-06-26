# 1. ContextEncoder  (pre-trained I-JEPA backbone)

`o_t  ->  x_t`

The **online encoder**. Maps a current observation (a channel measurement) into the latent
embedding `x_t` that feeds the selective SSM. Built around a **frozen pre-trained I-JEPA
ViT-H/14**; only a small input adapter and projection head are trained.

## Why I-JEPA (research summary)

We surveyed Meta's public JEPA checkpoints on HuggingFace `transformers`:

| Model        | Checkpoint                       | Domain | Hidden | Input             | Predictor |
| ------------ | -------------------------------- | ------ | ------ | ----------------- | --------- |
| **I-JEPA**   | `facebook/ijepa_vith14_1k`       | images | 1280   | `(B,3,224,224)`   | no (enc)  |
| I-JEPA (big) | `facebook/ijepa_vitg16_22k`      | images | 1408   | `(B,3,224,224)`   | no (enc)  |
| **V-JEPA 2** | `facebook/vjepa2-vitl-fpc64-256` | video  | 1024   | `(B,T,3,256,256)` | yes (+AC) |

**Decision:** baseline uses **I-JEPA ViT-H/14**, **frozen**. It is precisely the "encoder"
role in `sswm_fig.pdf`; our own `SelectiveSSM` + `Predictor` provide the temporal and
action-conditioned world-model dynamics (the novel part). V-JEPA 2 is the documented
upgrade path — it is a native video world model whose `-AC` variant already has an
action-conditioned predictor that could later replace ours.

## Domain bridge: channel data -> ViT image space

A channel snapshot `H_t ∈ C^{N_sub × N_ant}` is image-like (subcarrier × antenna, or
delay × Doppler grid). We bridge it to the ViT's expected `(3, 224, 224)` input:

```
H_t (complex)
  → stack real/imag                 → (2, N_sub, N_ant)
  → ChannelAdapter (conv stem)      → (3, 224, 224)        [TRAINABLE]
  → frozen I-JEPA ViT-H/14          → (256 tokens, 1280)   [FROZEN]
  → mean-pool over tokens           → (1280,)
  → projection head (MLP)           → x_t (embed_dim)       [TRAINABLE]
```

## Interface

| Field   | Shape                         | Notes                                   |
| ------- | ----------------------------- | --------------------------------------- |
| input   | `(B, T, 2, N_sub, N_ant)`     | real/imag channel; T = sequence length  |
| output  | `(B, T, embed_dim)`           | embedding `x_t`                         |

```python
class ContextEncoder(nn.Module):
    def forward(self, o: Tensor) -> Tensor:   # (B,T,2,Nsub,Nant) -> (B,T,embed_dim)
    def trainable_parameters(self): ...        # adapter + head only (for EMA + optim)
```

## Baseline design

- **Backbone**: `IJepaModel.from_pretrained("facebook/ijepa_vith14_1k")`, `requires_grad_(False)`,
  `.eval()`. Loaded lazily.
- **Offline fallback**: if `transformers`/weights are unavailable, a small random ViT-shaped
  stub with the same 1280-d output is used so the module stays importable and testable.
  Controlled by `SSWMConfig.use_pretrained`.
- **ChannelAdapter**: conv stem `2→3` channels + bilinear resize to 224×224. Trainable.
- **Projection head**: LayerNorm → Linear(1280→embed_dim) → GELU → Linear. Trainable.
- **Time handling**: encoder is per-timestep; we fold `(B,T)` into the batch, run the ViT
  once, then unfold. SSM does the temporal mixing downstream.

## Test (M1)

- Output shape == `(B, T, embed_dim)`.
- Backbone params have `requires_grad == False`; adapter+head params have `requires_grad == True`.
- `trainable_parameters()` excludes the backbone (so EMA/optimizer only touch adapter+head).
- Runs in fallback (stub) mode with no network access.
