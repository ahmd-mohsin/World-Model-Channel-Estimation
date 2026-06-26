from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..config import SSWMConfig
except ImportError:
    from config import SSWMConfig


class SelectionNet(nn.Module):
    def __init__(self, config: SSWMConfig) -> None:
        super().__init__()
        self.config = config
        d = config.state_dim
        h = config.selection_hidden

        self.trunk = nn.Sequential(
            nn.Linear(config.action_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.head_A = nn.Linear(h, d)
        self.head_B = nn.Linear(h, d)
        self.head_C = nn.Linear(h, d)
        self.head_dt = nn.Linear(h, d)

        self._init_params()

    def _init_params(self) -> None:
        cfg = self.config

        nn.init.normal_(self.head_dt.weight, std=1e-3)
        dt = torch.exp(
            torch.rand(cfg.state_dim) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.head_dt.bias.copy_(inv_softplus_dt)

        a_targets = torch.arange(1, cfg.state_dim + 1, dtype=torch.float32)
        inv_softplus_a = torch.log(torch.expm1(a_targets))
        nn.init.normal_(self.head_A.weight, std=1e-3)
        with torch.no_grad():
            self.head_A.bias.copy_(inv_softplus_a)

    def forward(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(a)
        A = -F.softplus(self.head_A(h))
        B = self.head_B(h)
        C = self.head_C(h)
        dt = F.softplus(self.head_dt(h))
        return A, B, C, dt
