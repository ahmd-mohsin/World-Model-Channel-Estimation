from __future__ import annotations

import copy

import torch
import torch.nn as nn

try:
    from ..config import SSWMConfig
    from ..context_encoder import ContextEncoder
except ImportError:
    from config import SSWMConfig
    from context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    def __init__(self, source: ContextEncoder, config: SSWMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = copy.deepcopy(source)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self._sync_hard(source)

    @torch.no_grad()
    def _sync_hard(self, source: ContextEncoder) -> None:
        for tp, sp in zip(self.encoder.parameters(), source.parameters()):
            tp.data.copy_(sp.data)
        for tb, sb in zip(self.encoder.buffers(), source.buffers()):
            tb.data.copy_(sb.data)

    @torch.no_grad()
    def ema_update(self, source: ContextEncoder, m: float | None = None) -> None:
        m = self.config.ema_momentum if m is None else m
        for tp, sp in zip(self.encoder.parameters(), source.parameters()):
            if sp.requires_grad:
                tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)
        for tb, sb in zip(self.encoder.buffers(), source.buffers()):
            tb.data.copy_(sb.data)

    @torch.no_grad()
    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.encoder(o).detach()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self
