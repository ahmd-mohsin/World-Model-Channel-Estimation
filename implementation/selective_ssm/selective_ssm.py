from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..config import SSWMConfig
    from ..selection_net import SelectionNet
except ImportError:
    from config import SSWMConfig
    from selection_net import SelectionNet


def discretize(A, B, dt):
    dA = torch.exp(dt * A)
    dB = (dA - 1.0) / A * B
    return dA, dB


class SelectiveSSM(nn.Module):
    def __init__(self, config: SSWMConfig, selection_net: SelectionNet | None = None) -> None:
        super().__init__()
        self.config = config
        d = config.state_dim
        self.selection = selection_net if selection_net is not None else SelectionNet(config)
        self.in_proj = nn.Linear(config.embed_dim + config.action_dim, d)
        self.D = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Linear(d, config.latent_dim)
        self.norm = nn.LayerNorm(d)

    def _scan(self, u, A, B, C, dt):
        b, t, d = u.shape
        dA, dB = discretize(A, B, dt)
        h = torch.zeros(b, d, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(t):
            h = dA[:, i] * h + dB[:, i] * u[:, i]
            ys.append(C[:, i] * h + self.D * u[:, i])
        y = torch.stack(ys, dim=1)
        return y

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        A, B, C, dt = self.selection(a)
        u = self.in_proj(torch.cat([x, a], dim=-1))
        y = self._scan(u, A, B, C, dt)
        y = self.norm(y)
        return self.out_proj(y)

    def step(self, x_t: torch.Tensor, a_t: torch.Tensor, h_prev: torch.Tensor | None = None):
        A, B, C, dt = self.selection(a_t)
        u = self.in_proj(torch.cat([x_t, a_t], dim=-1))
        if h_prev is None:
            h_prev = torch.zeros(x_t.shape[0], self.config.state_dim, device=x_t.device, dtype=x_t.dtype)
        dA, dB = discretize(A, B, dt)
        h = dA * h_prev + dB * u
        y = C * h + self.D * u
        z = self.out_proj(self.norm(y))
        return z, h
