# SSWM Baseline Implementation Plan

**Selective Structured-State-Space World Model (SSWM)** for channel estimation.

This document describes how we build a **baseline** implementation of the architecture
shown in `docs/sswm_fig.pdf`. The model is a JEPA-style self-supervised world model whose
temporal backbone is a **selective state-space model (SSM, Mamba-like)**. We learn to
predict *future latent representations* from past observations and actions, entirely in
representation space (no raw-signal reconstruction).

> **Baseline backbone decision (researched).** The context/target encoders are built
> around a **frozen pre-trained I-JEPA ViT-H/14** (`facebook/ijepa_vith14_1k`,
> hidden dim **1280**, 224×224 input, 256 patch tokens). I-JEPA is Meta's image
> Joint-Embedding Predictive Architecture — exactly the "encoder" role in the figure.
> We keep the ViT **frozen** and train only a small **channel adapter** (maps complex
> channel tensors into the ViT's 3×224×224 image space) plus a **projection head**.
> The EMA target encoder is an EMA of the *trainable* parts only.
> See `context_encoder/README.md` for the full rationale and the V-JEPA 2 upgrade path.

For channel estimation, the mapping is:

| SSWM concept            | Channel-estimation meaning                                          |
| ----------------------- | ------------------------------------------------------------------- |
| Observation `o_t`       | Received pilot / channel measurement at time `t` (e.g. LS estimate) |
| Action `a_t`            | Tx/Rx configuration: pilot pattern, beam index, MCS, scheduling     |
| Latent `z_t`            | Compact channel-state representation                                |
| Future `o_{t+k}`        | Channel at a later slot/frame to be predicted                       |
| Task heads              | Channel reconstruction, SNR/Doppler prediction, beam selection      |

---

## 1. Architecture decomposition (the 6 models)

Each box in the figure is implemented as a self-contained module under its own folder.
Every module exposes a clean `forward()` and is independently unit-testable.

| # | Folder              | Module                | Role in the figure                                             |
| - | ------------------- | --------------------- | -------------------------------------------------------------- |
| 1 | `context_encoder/`  | **ContextEncoder**    | `o_t -> x_t`. Online encoder, trained by backprop.             |
| 2 | `target_encoder/`   | **TargetEncoder**     | `o_{t+k} -> z̃_{t+k}`. EMA copy of context encoder; stop-grad. |
| 3 | `selection_net/`    | **SelectionNet**      | `a_t -> (A_t, B_t, C_t, Δ_t)`. Input-dependent SSM params.     |
| 4 | `selective_ssm/`    | **SelectiveSSM**      | `[x_t; a_t] -> z_t`. Selective state-space recurrence.         |
| 5 | `predictor/`        | **Predictor**         | `(z_t, a_{t..t+k-1}) -> ẑ_{t+k}`. Rolls latent into future.    |
| 6 | `task_heads/`       | **TaskHeads**         | `z_t -> reward / policy / channel estimate`. Downstream.       |

The two remaining pieces of the figure are **not** separate models but are part of the
training loop, implemented in the top-level trainer:
- **JEPA loss** `L_JEPA` — distance between `ẑ_{t+k}` and `z̃_{t+k}`.
- **EMA update** + **stop-gradient** — collapse-prevention plumbing between encoders.

---

## 2. Data flow (one training step)

```
o_t  --ContextEncoder-->  x_t
                          [x_t ; a_t]  --+
a_t  --SelectionNet--> (A_t,B_t,C_t,Δ_t) |--SelectiveSSM--> z_t
                                                              |
              (a_t, ..., a_{t+k-1}) --Predictor------------>  ẑ_{t+k}
                                                              |
o_{t+k} --TargetEncoder(EMA, stop-grad)--> z̃_{t+k}           |
                                                              v
                              L_JEPA = D( ẑ_{t+k} , z̃_{t+k} )
```

Gradients flow into ContextEncoder, SelectionNet, SelectiveSSM, Predictor.
TargetEncoder is updated only by EMA of ContextEncoder weights.

---

## 3. Baseline design choices (keep it simple first)

The goal of the baseline is a **correct, end-to-end-trainable** pipeline — not SOTA.
We deliberately pick the simplest defensible option for each block:

- **Encoders**: small MLP / 1-D conv stack over the (real+imag) channel vector.
  Context and target share architecture; target starts as a hard copy.
- **SelectionNet**: 2-layer MLP from `a_t` to the SSM parameters; `Δ_t` via softplus
  to stay positive; `A_t` parameterized as negative (stable) diagonal.
- **SelectiveSSM**: **diagonal** state-space (S4D/Mamba-simplified) with a discretized
  recurrence (ZOH). Baseline runs the recurrence as a sequential scan — clarity over speed;
  a parallel/associative scan is a later optimization.
- **Predictor**: lightweight recurrent rollout — apply the SSM step `k` times feeding
  planned actions, then a projection head. Baseline may use an MLP that ingests `z_t` plus
  the action window.
- **JEPA loss**: smooth-L1 / MSE in latent space + **VICReg-style variance-covariance
  regularizer** on the target embeddings to prevent collapse (cheap insurance beyond EMA).
- **EMA**: `θ_target ← m·θ_target + (1−m)·θ_context`, `m≈0.99–0.999`.

All dimensions live in a single config so blocks compose without shape surprises.

---

## 4. Build order (milestones)

1. **M0 — Skeleton & config.** Shape contracts + dummy tensors flow end-to-end.
   `pytest` checks every module's output shape.
2. **M1 — Encoders + EMA.** Context/target encoders, EMA update, stop-grad verified
   (target grads are `None`).
3. **M2 — SelectionNet + SelectiveSSM.** Recurrence is numerically stable; gradients
   reach SelectionNet. Sanity: a fixed `A,B,C` matches a hand-checked linear recurrence.
4. **M3 — Predictor + JEPA loss.** Full forward; loss decreases on a toy synthetic
   sequence (e.g. AR(1) channel).
5. **M4 — Trainer + data.** Synthetic channel generator (Jakes/AR Rayleigh) → training
   loop with logging; representation does not collapse (monitor embedding variance).
6. **M5 — Task heads + eval.** Probe `z_t`/`ẑ_{t+k}` for channel-estimation NMSE vs. a
   classical LS/MMSE baseline.

---

## 5. Repository layout

```
implementation/
├── implementation.md          # this file
├── context_encoder/           # 1. ContextEncoder  (online, backprop)
├── target_encoder/            # 2. TargetEncoder    (EMA, stop-grad)
├── selection_net/             # 3. SelectionNet     (action -> SSM params)
├── selective_ssm/             # 4. SelectiveSSM     (temporal backbone)
├── predictor/                 # 5. Predictor        (future latent rollout)
└── task_heads/                # 6. TaskHeads        (downstream readouts)
```

Each folder contains its own `README.md` specifying inputs, outputs, baseline design,
and the test that proves it works. Cross-cutting code (trainer, JEPA loss, EMA helper,
config, synthetic data) is added at the top level once the six modules pass M0–M2.

---

## 6. Tech stack

- **PyTorch** for all modules (autograd handles the recurrence backprop in the baseline).
- **NumPy** for the synthetic channel generator.
- **pytest** for shape/gradient/collapse tests.
- Config via a single dataclass (`SSWMConfig`) shared by all modules.

---

## 7. Definition of done (baseline)

- [ ] All 6 modules importable and shape-correct (M0).
- [ ] Stop-gradient + EMA verified (M1).
- [ ] SSM recurrence matches reference linear recurrence (M2).
- [ ] JEPA loss decreases on synthetic AR(1) channel; no collapse (M3–M4).
- [ ] Channel-estimation NMSE reported against LS/MMSE baseline (M5).
