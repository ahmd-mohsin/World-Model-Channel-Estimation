# 1. ContextEncoder  (pluggable pre-trained backbone)

`o_t  ->  x_t`

The **online encoder**. Maps a channel measurement into the latent embedding `x_t` that
feeds the selective SSM. It is **backbone-agnostic**: a pluggable pretrained encoder turns
each channel snapshot into patch tokens, which we pool and project to the pipeline-wide
`embed_dim`. Only the projection head (and any trainable adapter inside the backbone) is
trained; pretrained weights stay frozen.

## Which pretrained encoder? (research summary)

We compared general-vision JEPA models against **domain-native wireless foundation models**:

| Backbone        | Checkpoint                  | Pretrained on        | Input it ingests        | Hidden | Verdict                          |
| --------------- | --------------------------- | -------------------- | ----------------------- | ------ | -------------------------------- |
| **LWM** (default) | `wi-lab/lwm-v1.1`         | **DeepMIMO channels**| channel matrices direct | 128    | **Domain-native — best fit**     |
| LWM 1.0         | `wi-lab/lwm`                | DeepMIMO + Sionna    | 32×32 channel           | 64     | Simpler fallback                 |
| I-JEPA          | `facebook/ijepa_vith14_1k`  | ImageNet (photos)    | 3×224×224 image         | 1280   | Transfer via channel→image adapter |
| V-JEPA 2        | `facebook/vjepa2-vitl-...`  | internet video       | video clips             | 1024   | Upgrade path (has own predictor) |

**Decision:** default to **LWM** — it is pretrained on **DeepMIMO**, the same data domain as
channel estimation, and ingests channel matrices *directly* (no image hack). It is the only
wireless foundation model with **publicly downloadable weights** and a documented
embedding-extraction API. I-JEPA is kept as an image-transfer option (it needs a trainable
channel→image conv stem); the stub keeps everything testable offline. **License note:** LWM's
HF repo declares no license — resolve before any redistribution.

> **Real LWM is wired in.** The `wi-lab/lwm-v1.1` source (`lwm_model.py`, `inference.py`,
> `input_preprocess.py`, `utils.py`) is vendored under `lwm/`. `LWMBackbone` reproduces the
> v1.1 tokenization (interleave real/imag → 4×4 patches → length-32 tokens → prepend `[CLS]`)
> and loads `models/model.pth`, stripping the `module.` `DataParallel` prefix. The weights
> (~10 MB) are gitignored; if absent they auto-download via `huggingface_hub`. If anything is
> unavailable offline, it falls back to the stub so the module stays importable.
>
> **Setup:** create the env and deps with `python3 -m venv wireless && source wireless/bin/activate
> && pip install torch numpy transformers huggingface_hub pytest`.

## Architecture

```
channel (B,T,2,Nsub,Nant)
  → [fold time into batch]                                  (B*T, 2, Nsub, Nant)
  → backbone  (LWM / I-JEPA+adapter / stub)  → tokens       (B*T, P, hidden)   [frozen]
  → mean-pool over tokens                                   (B*T, hidden)
  → ProjectionHead (LN→Linear→GELU→Linear)   → x_t          (B*T, embed_dim)   [trainable]
  → [unfold time]                                           (B, T, embed_dim)
```

Backbones live in `backbones.py` behind a uniform contract
`forward(channel) -> tokens (N, P, hidden)` and a `frozen_modules()` hook, so swapping
backbones never changes the rest of the pipeline. Selected via `SSWMConfig.backbone`
(`"lwm" | "ijepa" | "stub"`); `use_pretrained=False` forces the stub.

## Interface

| Field   | Shape                         | Notes                                   |
| ------- | ----------------------------- | --------------------------------------- |
| input   | `(B, T, 2, N_sub, N_ant)`     | real/imag channel; T = sequence length  |
| output  | `(B, T, embed_dim)`           | embedding `x_t`                         |

```python
class ContextEncoder(nn.Module):
    def forward(self, o: Tensor) -> Tensor:   # (B,T,2,Nsub,Nant) -> (B,T,embed_dim)
    def trainable_parameters(self): ...        # everything except frozen pretrained weights
```

## Freezing semantics

- Pretrained sub-modules a backbone declares via `frozen_modules()` are set `requires_grad=False`
  and `.eval()`. For I-JEPA only the ViT is frozen; its channel→image adapter stays trainable.
- We deliberately do **not** wrap the backbone forward in `no_grad`: gradients must flow
  *through* the frozen ViT activations to the trainable adapter below it.
- `trainable_parameters()` yields exactly the non-frozen params — what the optimizer and the
  EMA target encoder operate on.

## Head SSL pretraining (making the head non-random)

The projection head starts randomly initialized. A random *linear* projection already preserves
the (excellent) frozen-LWM features, so on **clean** channels a random head classifies location
near-perfectly — but it has no reason to be robust to noise. We pretrain the head (LWM frozen,
~99K params) with a **VICReg** objective whose positive pairs are **two noise-augmented views of
the same channel** — i.e. enforce noise-invariance, the prior behind channel estimation.

- Data: **2048 Sionna sequences** generated in parallel across all 8 A100s
  (`scripts/run-parallel-pretrain.sh` → `gen_sionna_shard.py` per GPU, ~1 min total).
- Train: `pretrain_head.py` (AdamW + cosine, 4000 steps). Loss 22.6 → 12.3; variance term
  0.90 → 0.29 (no collapse). Curve: `checkpoints/06_pretrain_loss.png`. Head:
  `checkpoints/head_vicreg.pt`.
- Verify: `probe_head.py` (linear probe, 6 location clusters):

  | probe | random head | trained head | Δ |
  | ----- | ----------- | ------------ | - |
  | clean location acc | 0.967 | 0.906 | −0.061 |
  | **noise-robust (train clean, test @10 dB)** | 0.517 | **0.911** | **+0.394** |

  The trained head is far more **noise-robust** (+0.39 absolute), exactly what the objective
  targets, at a small cost on the already-saturated clean metric. So the head is now genuinely
  task-tuned, not random. (For the full pipeline it can be further refined end-to-end by JEPA.)

## Tests (M1) — 9 passing

- Output shape `(B, T, embed_dim)` across `lwm` / `ijepa` / `stub` backbones.
- Variable channel grid sizes accepted with no code change.
- Head trainable; gradients flow to it.
- Real LWM loads + freezes; `trainable_parameters()` excludes frozen LWM weights.
- LWM tokenization shape = `(N, n_patches + 1 CLS, 32)`.
- Gradients reach the head, never the frozen LWM.

```
pytest implementation/context_encoder/test_context_encoder.py -q   # 9 passed
```

## Demo — verified output

`python implementation/context_encoder/demo_encoder.py` runs the encoder on a synthetic
multipath MIMO-OFDM channel and shows the real `o_t → x_t` mapping:

```
observations o_t   : (2, 8, 2, 32, 32)   (B, T, [real/imag], N_ant, N_sub)
embedding x_t      : (2, 8, 256)         (B, T, embed_dim)  --> feeds the SSM block
trainable params   : 99,072 / 2,569,376  (rest = frozen LWM)
pretrained loaded  : True
```

Sanity on the pretrained LWM channel embeddings: small perturbations stay close
(RMS 0.0005) while distinct channels separate ~16×, confirming the frozen DeepMIMO
representation is meaningful before any task training.
