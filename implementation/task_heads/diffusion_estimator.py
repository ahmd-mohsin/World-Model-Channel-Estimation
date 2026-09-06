"""Conditional diffusion (DDPM) channel estimator --- a strong generative learned baseline for the
ESTIMATION task, to sit alongside LS / MMSE / ReEsNet / world-model in the comparison.

Formulation: learn to denoise the clean beamspace channel b0 conditioned on the noisy observation. A
conv eps-network predicts the diffusion noise from [x_t, cond_obs, snr_map, t_embed]; at inference we
DDIM-sample b0 from x_T~N(0,I) conditioned on the observation, then inverse-DFT to the channel. This is
the conditional-generation route (robust and simple) rather than score-guided posterior sampling.

    torchrun ... scripts/train-diffusion-estimator.py --data_dir data/mimo --n_ant 32 --n_sub 32
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..wireless_data.beamspace import to_beamspace, from_beamspace
except ImportError:
    from wireless_data.beamspace import to_beamspace, from_beamspace


def _t_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class _Block(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.temb = nn.Linear(temb, cout); self.norm = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = F.gelu(self.norm(self.c1(x))) + self.temb(te)[:, :, None, None]
        h = F.gelu(self.c2(h))
        return h + self.skip(x)


class EpsNet(nn.Module):
    """Predicts diffusion noise on the beamspace channel, conditioned on obs + snr."""

    def __init__(self, base=64, temb=128):
        super().__init__()
        self.temb = temb
        # in: x_t(2) + cond obs(2) + snr map(1) = 5
        self.stem = nn.Conv2d(5, base, 3, padding=1)
        self.b1 = _Block(base, base, temb); self.b2 = _Block(base, base * 2, temb)
        self.b3 = _Block(base * 2, base * 2, temb); self.b4 = _Block(base * 2, base, temb)
        self.out = nn.Conv2d(base, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x_t, cond, snr_map, t):
        te = _t_embed(t, self.temb)
        h = self.stem(torch.cat([x_t, cond, snr_map], 1))
        h = self.b1(h, te); h = self.b2(h, te); h = self.b3(h, te); h = self.b4(h, te)
        return self.out(h)


class DiffusionEstimator(nn.Module):
    def __init__(self, config, T=1000, base=64):
        super().__init__()
        self.config = config; self.T = T
        self.net = EpsNet(base=base)
        betas = torch.linspace(1e-4, 0.02, T)
        acp = torch.cumprod(1 - betas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("acp", acp)

    def _cond(self, o, snr_db, noise_gen=None):
        """noisy observation in beamspace + snr map, at the given SNR."""
        power = o.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        nvar = power / (10 ** (snr_db / 10))
        y = o + torch.randn(o.shape, generator=noise_gen, device=o.device) * nvar.sqrt()
        cond = to_beamspace(y)
        snr_map = torch.full_like(cond[:, :1], 0.0) + (snr_db / 30.0)
        return cond, snr_map

    def losses(self, o, a=None, snr_range=(-5.0, 30.0), noise_gen=None, **kw):
        H = o[:, -1] if o.dim() == 5 else o                       # use last frame (B,2,A,S)
        b0 = to_beamspace(H)
        B = b0.shape[0]
        snr = torch.empty(B, 1, 1, 1, device=H.device).uniform_(*snr_range)
        cond, snr_map = self._cond(H, snr, noise_gen)
        t = torch.randint(0, self.T, (B,), device=H.device)
        ac = self.acp[t][:, None, None, None]
        eps = torch.randn(b0.shape, generator=noise_gen, device=H.device)
        x_t = ac.sqrt() * b0 + (1 - ac).sqrt() * eps
        pred = self.net(x_t, cond, snr_map, t)
        loss = F.mse_loss(pred, eps)
        return loss, {"total": loss.item(), "pred": loss.item(), "chan": 0.0, "chan_nmse": 0.0}

    @torch.no_grad()
    def estimate(self, H, snr_db, steps=20, noise_gen=None):
        """DDIM sampling conditioned on the noisy obs of H at snr_db -> channel-domain estimate."""
        B = H.shape[0]
        snr = torch.full((B, 1, 1, 1), float(snr_db), device=H.device)
        cond, snr_map = self._cond(H, snr, noise_gen)
        x = torch.randn(cond.shape, generator=noise_gen, device=H.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=H.device).long()
        for i in range(steps):
            t = ts[i].repeat(B)
            ac = self.acp[t][:, None, None, None]
            eps = self.net(x, cond, snr_map, t)
            b0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
            if i < steps - 1:
                ac_next = self.acp[ts[i + 1]][None, None, None, None]
                x = ac_next.sqrt() * b0 + (1 - ac_next).sqrt() * eps
            else:
                x = b0
        return from_beamspace(x)

    def forward(self, o, a=None, **kw):
        return self.losses(o, a, **kw)
