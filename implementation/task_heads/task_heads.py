from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..config import SSWMConfig
except ImportError:
    from config import SSWMConfig


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512, depth: int = 2) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskHeads(nn.Module):
    """Downstream readouts on the world-model latent z (or predicted ẑ).

    Heads are independent MLP probes. The channel head is the headline: it reconstructs the
    channel grid from the latent, evaluated by NMSE against classical LS / MMSE estimators.
    `in_dim` defaults to latent_dim; pass embed_dim when probing the predictor's output ẑ.
    """

    def __init__(self, config: SSWMConfig, in_dim: int | None = None,
                 heads=("channel", "reward", "policy"), channel_use_obs: bool = True) -> None:
        super().__init__()
        self.config = config
        self.in_dim = in_dim if in_dim is not None else config.latent_dim
        self.obs_dim = config.obs_channels * config.n_subcarriers * config.n_antennas
        # The frozen LWM latent is lossy/invariant and cannot be inverted back to the exact
        # channel (verified: latent->channel NMSE ~0.31 even at 20 dB, vs ~0.005 from the raw
        # obs). So the channel head also takes the raw observation: the latent supplies learned
        # context (denoising prior), the obs supplies the detail to reconstruct. Predicts a
        # RESIDUAL on the observation (zero-init) -> starts at LS, learns the denoising.
        self.channel_use_obs = channel_use_obs
        self.heads = nn.ModuleDict()
        if "channel" in heads:
            cin = self.in_dim + (self.obs_dim if channel_use_obs else 0)
            ch = _MLP(cin, self.obs_dim, hidden=512, depth=2)
            if channel_use_obs:
                nn.init.zeros_(ch.net[-1].weight); nn.init.zeros_(ch.net[-1].bias)
            self.heads["channel"] = ch
        if "reward" in heads:
            self.heads["reward"] = _MLP(self.in_dim, 1, hidden=128, depth=2)
        if "policy" in heads:
            self.heads["policy"] = _MLP(self.in_dim, config.action_dim, hidden=128, depth=2)

    def forward(self, z: torch.Tensor, obs: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = {}
        for name, head in self.heads.items():
            if name == "channel" and self.channel_use_obs:
                o_flat = obs.reshape(obs.shape[0], -1)
                out[name] = o_flat + head(torch.cat([z, o_flat], dim=-1))
            else:
                out[name] = head(z)
        return out

    def channel_grid(self, z: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """Channel estimate reshaped to (B, 2, n_antennas, n_subcarriers)."""
        flat = self.forward(z, obs)["channel"]
        return flat.reshape(z.shape[0], self.config.obs_channels,
                            self.config.n_antennas, self.config.n_subcarriers)
