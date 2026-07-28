"""Learned deep channel-estimation baselines for the sparse-pilot regime.

The apples-to-apples competitor to the Beam-WM: a single-snapshot CNN that sees ONLY the current
noisy sparse-pilot observation (+ mask + noise level) and regresses the full channel. NO temporal
history, NO world-model prior, NO oracle covariance -- unlike masked-MMSE which is handed the true
frequency covariance. This isolates "does the temporal world model beat a strong learned per-snapshot
denoiser given the same information."

ReEsNet-style: residual CNN over the antenna x subcarrier grid, deep stack of residual blocks,
global skip; the observation is the input and the net predicts the (masked-hole-filling) residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ReEsNet(nn.Module):
    """Residual CNN sparse-pilot channel estimator (single snapshot).

    Input channels: [Y_masked (2), mask (1), noise-var map (1)] = 4.
    Output: full channel estimate (B,2,A,S). Interpolation is learned; the net fills the unobserved
    subcarriers from the pilot pattern. No oracle statistics, no temporal context.
    """

    def __init__(self, n_ant: int, n_sub: int, base: int = 64, n_blocks: int = 8) -> None:
        super().__init__()
        self.n_ant, self.n_sub = n_ant, n_sub
        self.inc = nn.Conv2d(4, base, 3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(base) for _ in range(n_blocks)])
        self.mid = nn.Conv2d(base, base, 3, padding=1)
        self.outc = nn.Conv2d(base, 2, 3, padding=1)

    def forward(self, Y_masked: torch.Tensor, mask: torch.Tensor,
                noise_var: torch.Tensor) -> torch.Tensor:
        b = Y_masked.shape[0]
        nv = noise_var.reshape(b, 1, 1, 1).expand(b, 1, self.n_ant, self.n_sub)
        mfull = mask.expand(b, 1, self.n_ant, self.n_sub)
        x = torch.cat([Y_masked, mfull, nv], dim=1)              # (B,4,A,S)
        h = self.inc(x)
        h = self.mid(self.body(h)) + h                          # global residual (ReEsNet skip)
        return self.outc(h)                                     # full channel estimate


def route_to_world_model(pilot_fraction: float, snr_db: float,
                         sparse_thresh: float = 0.125,
                         snr_lo: float = 0.0, snr_hi: float = 15.0) -> bool:
    """CSI-adaptive router grounded in the PHYSICS of when a temporal prior can help.

    The world-model prior is a prediction rolled from recent channel history. It is useful only when
    the current observation is degraded AND the history is informative:
      - Very LOW SNR: the history frames driving the prior are themselves at the same low SNR, so the
        prior is built on noise and carries little reliable signal -> route to the per-snapshot CNN,
        which exploits instantaneous spatial correlation instead.
      - Very HIGH SNR: the observation is already near-perfect, so the prior is redundant and only
        adds overhead -> route to the per-snapshot CNN.
      - MODERATE SNR with SPARSE pilots: the observation is missing many subcarriers yet the history
        is clean enough to form a useful prior -> route to the world-model deep-fusion estimator.
    Uses only inference-known CSI (pilot density set by the system, SNR estimated); no oracle
    statistics and no channel ground truth. The moderate-SNR window narrows as the antenna count
    grows (methodology Table tab:est), so at scale the router falls back to the CNN more often.
    """
    return (pilot_fraction <= sparse_thresh + 1e-9) and (snr_lo <= snr_db <= snr_hi)
