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
    from .task_heads import TaskHeads
except ImportError:
    from config import SSWMConfig
    from context_encoder import ContextEncoder
    from target_encoder import TargetEncoder
    from selection_net import SelectionNet
    from selective_ssm import SelectiveSSM
    from predictor import Predictor
    from task_heads import TaskHeads


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
        # Downstream readouts on z (channel estimate / reward / policy). Probe-style: trained
        # separately so they don't perturb the self-supervised world-model objective.
        self.task_heads = TaskHeads(config, in_dim=config.latent_dim)

    def encode_sequence(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """o (B,T,2,Nant,Nsub), a (B,T,action_dim) -> z (B,T,latent_dim)."""
        x = self.context_encoder(o)
        return self.ssm(x, a)

    def forward(self, o: torch.Tensor, a: torch.Tensor, anchor: int | None = None,
                loss: bool = False, **loss_kw):
        # When loss=True, dispatch to the full multi-task loss THROUGH this forward so DDP's
        # gradient sync hooks fire (calling all_losses on .module directly would skip DDP sync).
        if loss:
            return self.all_losses(o, a, anchor=anchor, **loss_kw)
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
        delta = self.predictor(z_t, planned)                 # (B,embed_dim)

        # Target: future observation through the EMA encoder (stop-grad), embed_dim space.
        o_future = o[:, anchor + k]                          # (B,2,Nant,Nsub)
        z_tilde = self.target_encoder(o_future.unsqueeze(1))[:, 0]  # (B,embed_dim), detached

        # Residual prediction: the predictor learns the CHANGE from the present embedding,
        # so persistence (delta=0) is the built-in prior. The present embedding is taken
        # from the (stop-grad) target encoder so the prior is on the same scale as z_tilde.
        if self.config.residual_prediction:
            with torch.no_grad():
                z_present = self.target_encoder(o[:, anchor].unsqueeze(1))[:, 0]
            z_hat = z_present + delta
        else:
            z_hat = delta

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

    def all_losses(self, o, a, snr_db=10.0, anchor=None,
                   w_jepa=1.0, w_vic=0.05, w_chan=5.0, noise_gen=None):
        """End-to-end multi-task loss with SEPARATE components for logging.

        - jepa : predictor matches EMA target of the future channel (world-model objective)
        - vic  : VICReg variance+covariance on z_hat (anti-collapse)
        - chan : channel head denoises a NOISY observation back to the clean channel (task)
        Returns (total, dict_of_component_floats).
        """
        cfg = self.config
        t = o.shape[1]
        k = cfg.horizon_k
        anchor = (t - 1 - k) if anchor is None else anchor

        # --- world-model (JEPA) path ---
        z = self.encode_sequence(o, a)
        z_t = z[:, anchor]
        delta = self.predictor(z_t, a[:, anchor:anchor + k])
        z_tilde = self.target_encoder(o[:, anchor + k].unsqueeze(1))[:, 0]
        if cfg.residual_prediction:
            with torch.no_grad():
                z_present = self.target_encoder(o[:, anchor].unsqueeze(1))[:, 0]
            z_hat = z_present + delta
        else:
            z_hat = delta
        jepa = F.mse_loss(z_hat, z_tilde) + 0.05 * (1.0 - F.cosine_similarity(z_hat, z_tilde, -1)).mean()

        # --- VICReg anti-collapse on z_hat ---
        zc = z_hat - z_hat.mean(0)
        std = torch.sqrt(zc.var(0) + 1e-4)
        var_term = torch.relu(1.0 - std).mean()
        n, d = zc.shape
        cov = (zc.T @ zc) / (n - 1)
        cov_term = (cov.fill_diagonal_(0.0) ** 2).sum() / d
        vic = var_term + cov_term

        # --- channel-estimation task (denoise a noisy obs at the anchor) ---
        H_clean = o[:, anchor]                                   # (B,2,ant,sub)
        power = H_clean.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        noise_p = power / (10 ** (snr_db / 10))
        noise = torch.randn(H_clean.shape, generator=noise_gen, device=H_clean.device, dtype=H_clean.dtype)
        Y = H_clean + noise * noise_p.sqrt()
        seq_noisy = o.clone(); seq_noisy[:, anchor] = Y
        z_obs = self.encode_sequence(seq_noisy, a)[:, anchor]
        est = self.task_heads(z_obs, Y)["channel"]
        chan = F.mse_loss(est, H_clean.reshape(H_clean.shape[0], -1))

        total = w_jepa * jepa + w_vic * vic + w_chan * chan
        with torch.no_grad():
            chan_nmse = (F.mse_loss(est, H_clean.reshape(H_clean.shape[0], -1))
                         / H_clean.pow(2).mean()).item()
            ls_nmse = ((Y - H_clean).pow(2).mean() / H_clean.pow(2).mean()).item()
            metrics = {
                "total": total.item(), "jepa": jepa.item(), "vic": vic.item(),
                "chan": chan.item(), "chan_nmse": chan_nmse, "ls_nmse": ls_nmse,
                "pred_std": z_hat.std(0).mean().item(), "target_std": z_tilde.std(0).mean().item(),
            }
        return total, metrics

    def trainable_parameters(self):
        seen = set()
        for m in (self.context_encoder, self.selection, self.ssm, self.predictor):
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def all_trainable_parameters(self):
        """World-model params PLUS task heads — for end-to-end joint training."""
        seen = set()
        for m in (self.context_encoder, self.selection, self.ssm, self.predictor, self.task_heads):
            for p in m.parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        """Two LR groups: pretrained LWM backbone (smaller LR) vs everything else.

        Used when fully fine-tuning LWM end-to-end so the pretrained weights adapt gently while
        fresh heads/SSM/predictor learn fast. Returns a list for torch.optim.
        """
        backbone_ids = {id(p) for p in self.context_encoder.backbone.parameters()}
        bb, rest, seen = [], [], set()
        for p in self.all_trainable_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            (bb if id(p) in backbone_ids else rest).append(p)
        groups = [{"params": rest, "lr": base_lr}]
        if bb:
            groups.append({"params": bb, "lr": base_lr * backbone_lr_mult})
        return groups

    @torch.no_grad()
    def update_target(self) -> None:
        self.target_encoder.ema_update(self.context_encoder)
