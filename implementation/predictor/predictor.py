from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..config import SSWMConfig
    from ..selection_net import SelectionNet
    from ..selective_ssm import discretize
except ImportError:
    from config import SSWMConfig
    from selection_net import SelectionNet
    from selective_ssm import discretize


class Predictor(nn.Module):
    """(z_t, a_t..a_{t+k-1}) -> ẑ_{t+k}.

    Observation-free latent rollout. The future is NOT observed, so the rollout is
    driven by planned ACTIONS only (plus the current latent z_t as the initial state).
    Output lives in embed_dim (the target-encoder space the JEPA loss compares against),
    NOT in SSM latent space.
    """

    def __init__(self, config: SSWMConfig, selection_net: SelectionNet | None = None) -> None:
        super().__init__()
        self.config = config
        d = config.state_dim
        self.selection = selection_net if selection_net is not None else SelectionNet(config)
        self.z_to_state = nn.Linear(config.latent_dim, d)
        self.act_proj = nn.Linear(config.action_dim, d)
        self.D = nn.Parameter(torch.ones(d))
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, config.embed_dim)

    def forward(self, z_t: torch.Tensor, planned_acts: torch.Tensor) -> torch.Tensor:
        if planned_acts.dim() != 3:
            raise ValueError(f"planned_acts must be (B, k, action_dim), got {tuple(planned_acts.shape)}")
        b, k, _ = planned_acts.shape
        h = self.z_to_state(z_t)
        if k == 0:
            return self.out_proj(self.norm(h))
        for j in range(k):
            a_j = planned_acts[:, j]
            A, B, C, dt = self.selection(a_j)
            u = self.act_proj(a_j)
            dA, dB = discretize(A, B, dt)
            h = dA * h + dB * u
            y = C * h + self.D * u
        return self.out_proj(self.norm(y))

    def rollout(self, z_t: torch.Tensor, planned_acts: torch.Tensor) -> torch.Tensor:
        """Return predictions at every horizon 1..k as (B, k, embed_dim)."""
        b, k, _ = planned_acts.shape
        h = self.z_to_state(z_t)
        outs = []
        for j in range(k):
            a_j = planned_acts[:, j]
            A, B, C, dt = self.selection(a_j)
            u = self.act_proj(a_j)
            dA, dB = discretize(A, B, dt)
            h = dA * h + dB * u
            y = C * h + self.D * u
            outs.append(self.out_proj(self.norm(y)))
        return torch.stack(outs, dim=1)
