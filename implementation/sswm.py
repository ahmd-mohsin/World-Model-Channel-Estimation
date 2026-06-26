"""SSWM — full world model wiring all modules together (Modules 1-5 + JEPA loss).

Implements one self-supervised training step exactly as docs/sswm_fig.pdf:

    o_{0:T}  --context encoder-->  x_{0:T}
    (x, a)   --SelectiveSSM----->  z_{0:T}            (SelectionNet inside)
    z_t, a_{t:t+k}  --Predictor->  ẑ_{t+k}            (embed_dim space)
    o_{t+k}  --target encoder--->  z̃_{t+k}  (EMA, stop-grad, embed_dim space)
    L_JEPA = || ẑ_{t+k} - sg(z̃_{t+k}) ||

The SelectionNet is SHARED between the SSM and the Predictor so the action-conditioned
dynamics are consistent across the encode path and the imagination path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import SSWMConfig
    from .context_encoder import ContextEncoder
    from .target_encoder import TargetEncoder
    from .selection_net import SelectionNet
    from .selective_ssm import SelectiveSSM
    from .predictor import Predictor
except ImportError:
    from config import SSWMConfig
    from context_encoder import ContextEncoder
    from target_encoder import TargetEncoder
    from selection_net import SelectionNet
    from selective_ssm import SelectiveSSM
    from predictor import Predictor


class SSWM(nn.Module):
    def __init__(self, config: SSWMConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = ContextEncoder(config)
        self.target_encoder = TargetEncoder(self.context_encoder, config)
        # One SelectionNet shared by both the encode-path SSM and the imagination Predictor.
        self.selection = SelectionNet(config)
        self.ssm = SelectiveSSM(config, selection_net=self.selection)
        self.predictor = Predictor(config, selection_net=self.selection)

    def encode_sequence(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """o (B,T,2,Nant,Nsub), a (B,T,action_dim) -> z (B,T,latent_dim)."""
        x = self.context_encoder(o)
        return self.ssm(x, a)

    def forward(self, o: torch.Tensor, a: torch.Tensor, anchor: int | None = None):
        cfg = self.config
        b, t = o.shape[:2]
        k = cfg.horizon_k
        if anchor is None:
            anchor = t - 1 - k
        if not (0 <= anchor and anchor + k < t):
            raise ValueError(f"anchor {anchor} + horizon {k} must be < seq_len {t}")

        z = self.encode_sequence(o, a)                       # (B,T,latent)
        z_t = z[:, anchor]                                   # (B,latent)
        planned = a[:, anchor:anchor + k]                    # (B,k,action_dim)
        z_hat = self.predictor(z_t, planned)                 # (B,embed_dim)

        # Target: future observation through the EMA encoder (stop-grad), embed_dim space.
        o_future = o[:, anchor + k]                          # (B,2,Nant,Nsub)
        z_tilde = self.target_encoder(o_future.unsqueeze(1))[:, 0]  # (B,embed_dim), detached

        if z_hat.shape != z_tilde.shape:
            raise RuntimeError(f"predictor/target shape mismatch: {z_hat.shape} vs {z_tilde.shape}")
        return z_hat, z_tilde

    def jepa_loss(self, o: torch.Tensor, a: torch.Tensor, anchor: int | None = None):
        z_hat, z_tilde = self(o, a, anchor)
        loss = F.smooth_l1_loss(z_hat, z_tilde)
        with torch.no_grad():
            metrics = {
                "loss": loss.item(),
                "pred_std": z_hat.std(0).mean().item(),
                "target_std": z_tilde.std(0).mean().item(),
            }
        return loss, metrics

    def trainable_parameters(self):
        seen = set()
        for m in (self.context_encoder, self.selection, self.ssm, self.predictor):
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield p

    @torch.no_grad()
    def update_target(self) -> None:
        self.target_encoder.ema_update(self.context_encoder)
