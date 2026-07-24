"""Shared configuration for the SSWM channel-estimation pipeline.

A single dataclass that every module imports so blocks compose without shape
surprises. See implementation.md for the architecture overview.
"""

from __future__ import annotations

from dataclasses import dataclass


def scale_embedding(x, config):
    """Project an embedding onto a fixed-radius sphere so all module outputs share one scale.

    L2-normalize the last dim then rescale to `embed_scale`. No-op when normalize_embeddings
    is False. Import in every module that emits a pipeline embedding (x_t, z_t, ẑ, z̃).
    """
    if not getattr(config, "normalize_embeddings", False):
        return x
    import torch.nn.functional as _F
    return _F.normalize(x, dim=-1) * config.embed_scale


@dataclass
class SSWMConfig:
    # ---- channel observation geometry ----
    n_subcarriers: int = 64          # N_sub  (frequency axis of the channel grid)
    n_antennas: int = 64             # N_ant  (spatial/antenna axis)
    obs_channels: int = 2            # real + imaginary planes
    seq_len: int = 16                # T: timesteps per training sequence
    horizon_k: int = 4               # k: prediction horizon (predict z_{t+k})

    # ---- action geometry ----
    # Velocity action [vx, vy, speed, theta] — the physical control driving channel evolution.
    # (Earlier a power-proxy action gave the predictor nothing to learn from; velocity fixed it.)
    action_dim: int = 4

    # ---- latent dimensions ----
    embed_dim: int = 256             # x_t / z_t / z̃ dimensionality (pipeline-wide)
    state_dim: int = 64              # SSM hidden state size (diagonal)
    latent_dim: int = 256            # SSM output latent (kept == embed_dim by default)

    # ---- embedding scale contract ----
    # Every module that emits a pipeline embedding (x_t, z_t, ẑ, z̃) L2-normalizes its output
    # to this fixed radius, so all module boundaries live at the SAME magnitude. Removes the
    # scale drift (e.g. pred_std 2.4 vs target_std 5.4) that made JEPA MSE/NMSE unreliable.
    normalize_embeddings: bool = True
    embed_scale: float = 16.0        # sqrt(embed_dim=256) -> unit-RMS features at this radius

    # ---- TaskHeads / channel head ----
    # "mlp"  : flat MLP on obs+latent (baseline).
    # "unet" : conv U-Net over the antenna x subcarrier grid, FiLM-conditioned on the latent and
    #          on the noise variance (leveling the field with MMSE, which gets the noise power).
    #          Attacks the low-SNR (0 dB) regime where the MLP head loses to MMSE.
    channel_head: str = "unet"
    unet_base_ch: int = 48           # base conv width of the U-Net

    # ---- SelectionNet (input-dependent SSM params A,B,C,Δ from actions) ----
    selection_hidden: int = 128      # hidden width of the SelectionNet trunk
    dt_min: float = 1e-3             # Δ init range lower bound (log-uniform)
    dt_max: float = 1e-1             # Δ init range upper bound (log-uniform)

    # ---- pretrained encoder backbone ----
    # "lwm"   : Large Wireless Model — DeepMIMO-pretrained, domain-native (RECOMMENDED).
    # "ijepa" : Meta I-JEPA ViT — image-domain transfer via a channel->image adapter.
    # "stub"  : offline random encoder, same output contract (for tests / no network).
    backbone: str = "lwm"
    use_pretrained: bool = True              # False -> force stub regardless of `backbone`
    # freeze_backbone=True keeps LWM frozen (probe/LoRA). Set False to FULLY fine-tune LWM
    # end-to-end (all backbone weights trainable) -- the "go full deep" setting. Use a smaller
    # LR for the backbone than for fresh heads (see trainer's param groups).
    freeze_backbone: bool = True
    # LoRA-unfreeze the backbone: inject trainable low-rank adapters into LWM attention while
    # keeping base weights frozen. Gives the frozen encoder capacity to adapt to Sionna channels
    # for large-scale training (zero cost when False).
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16

    # LWM (wireless) backbone
    lwm_checkpoint: str = "model.pth"
    lwm_hidden: int = 128                    # v1.1 D_MODEL (v1.0 = 64)
    lwm_patch_rows: int = 4
    lwm_patch_cols: int = 4
    lwm_element_length: int = 32             # patch_rows * patch_cols * 2

    # I-JEPA (image) backbone
    jepa_checkpoint: str = "facebook/ijepa_vith14_1k"
    jepa_hidden: int = 1280                  # ViT-H/14 hidden size
    jepa_image_size: int = 224               # backbone expected H == W

    @property
    def backbone_hidden(self) -> int:
        """Token/feature width emitted by the selected backbone."""
        return {
            "lwm": self.lwm_hidden,
            "ijepa": self.jepa_hidden,
            "stub": self.lwm_hidden,
        }[self.backbone]

    # ---- pretrained projection head (optional, from head SSL pretraining) ----
    # If set and the file exists, ContextEncoder loads these head weights at init.
    # None or missing file -> random head (graceful). Checkpoint is gitignored, so this
    # is opt-in rather than a forced default.
    head_checkpoint: str | None = None

    # ---- predictor ----
    # Predict the future embedding as present_embedding + learned_residual, so persistence
    # (residual=0) is the prior. Without this, predicting ẑ from scratch loses to persistence
    # on slowly-varying channels (measured: NMSE 0.99 vs 0.14 across multi-scene data).
    residual_prediction: bool = True

    # ---- EMA target encoder ----
    ema_momentum: float = 0.996

    def __post_init__(self) -> None:
        assert self.horizon_k < self.seq_len, "horizon_k must be < seq_len"
        assert self.obs_channels == 2, "baseline expects real/imag = 2 planes"
