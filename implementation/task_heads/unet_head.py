from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation: condition conv features on a context vector."""

    def __init__(self, ctx_dim: int, ch: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(ctx_dim, 2 * ch)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(ctx)
        gamma, beta = gb.chunk(2, dim=-1)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetChannelHead(nn.Module):
    """Denoise a noisy channel grid (B, 2, N_ant, N_sub) -> clean channel, as a residual on the obs.

    - Operates on the 2D antenna x subcarrier grid (conv U-Net, skip connections) so spatial
      channel structure is exploited (unlike the flat MLP head).
    - Conditioned on: the world-model latent z (FiLM at the bottleneck) AND the per-sample noise
      variance (broadcast as an extra input channel + FiLM) -- leveling the field with MMSE, which
      is handed the noise power.
    - Predicts a RESIDUAL on the observation; final conv is zero-init so it STARTS at LS.
    """

    def __init__(self, latent_dim: int, n_ant: int, n_sub: int, base: int = 48,
                 use_prior: bool = False) -> None:
        super().__init__()
        self.n_ant, self.n_sub = n_ant, n_sub
        self.use_prior = use_prior
        ctx_dim = latent_dim + 16  # latent + noise-embedding
        self.noise_embed = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16))

        # input: 2 (obs re/im) + 1 (noise-var map) [+ 2 (prior re/im) if predictive] channels
        in_ch = 3 + (2 if use_prior else 0)
        self.inc = _ConvBlock(in_ch, base)
        self.down1 = _ConvBlock(base, base * 2)
        self.down2 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.AvgPool2d(2)
        self.film = FiLM(ctx_dim, base * 4)
        self.up2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.up1 = _ConvBlock(base * 2 + base, base)
        self.outc = nn.Conv2d(base, 2, 1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)  # start at LS
        if use_prior:
            # Learned per-element Kalman gain g in [0,1]: fuse obs and prior as
            # base = g*obs + (1-g)*prior, then add the conv residual. g is produced from
            # [obs, prior, noise] so the network can trust the prior more when noise is high.
            self.gate = nn.Sequential(
                nn.Conv2d(5, base, 3, padding=1), nn.GELU(),
                nn.Conv2d(base, 2, 3, padding=1), nn.Sigmoid())

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise_var: torch.Tensor,
                prior: torch.Tensor | None = None) -> torch.Tensor:
        b = obs.shape[0]
        nv = noise_var.reshape(b, 1, 1, 1).expand(b, 1, self.n_ant, self.n_sub)
        if self.use_prior:
            if prior is None:
                prior = torch.zeros_like(obs)
            x = torch.cat([obs, nv, prior], dim=1)               # (B,5,A,S)
        else:
            x = torch.cat([obs, nv], dim=1)                      # (B,3,A,S)
        ctx = torch.cat([z, self.noise_embed(noise_var.reshape(b, 1))], dim=-1)

        x0 = self.inc(x)
        x1 = self.down1(self.pool(x0))
        x2 = self.down2(self.pool(x1))
        x2 = self.film(x2, ctx)
        u2 = self.up2(torch.cat([F.interpolate(x2, size=x1.shape[-2:], mode="nearest"), x1], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=x0.shape[-2:], mode="nearest"), x0], 1))
        residual = self.outc(u1)

        if self.use_prior:
            g = self.gate(torch.cat([obs, prior, nv], dim=1))    # (B,2,A,S) in [0,1]
            base = g * obs + (1 - g) * prior                     # Kalman-style fusion
            return base + residual
        return obs + residual
