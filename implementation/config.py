"""Shared configuration for the SSWM channel-estimation pipeline.

A single dataclass that every module imports so blocks compose without shape
surprises. See implementation.md for the architecture overview.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SSWMConfig:
    # ---- channel observation geometry ----
    n_subcarriers: int = 64          # N_sub  (frequency axis of the channel grid)
    n_antennas: int = 64             # N_ant  (spatial/antenna axis)
    obs_channels: int = 2            # real + imaginary planes
    seq_len: int = 16                # T: timesteps per training sequence
    horizon_k: int = 4               # k: prediction horizon (predict z_{t+k})

    # ---- action geometry ----
    action_dim: int = 8              # Tx/Rx config: pilot pattern, beam idx, MCS, ...

    # ---- latent dimensions ----
    embed_dim: int = 256             # x_t / z_t / z̃ dimensionality (pipeline-wide)
    state_dim: int = 64              # SSM hidden state size (diagonal)
    latent_dim: int = 256            # SSM output latent (kept == embed_dim by default)

    # ---- pretrained I-JEPA backbone ----
    use_pretrained: bool = True              # False -> offline random ViT stub
    jepa_checkpoint: str = "facebook/ijepa_vith14_1k"
    jepa_hidden: int = 1280                  # ViT-H/14 hidden size
    jepa_image_size: int = 224               # backbone expected H == W
    freeze_backbone: bool = True

    # ---- EMA target encoder ----
    ema_momentum: float = 0.996

    def __post_init__(self) -> None:
        assert self.horizon_k < self.seq_len, "horizon_k must be < seq_len"
        assert self.obs_channels == 2, "baseline expects real/imag = 2 planes"
