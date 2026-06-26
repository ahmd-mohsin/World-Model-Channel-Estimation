from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ..config import SSWMConfig
    from .backbones import build_backbone
except ImportError:
    from config import SSWMConfig
    from backbones import build_backbone


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextEncoder(nn.Module):
    def __init__(self, config: SSWMConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = build_backbone(config)
        self.head = ProjectionHead(self.backbone.hidden, config.embed_dim)

        self._frozen_param_ids: set[int] = set()
        if config.freeze_backbone:
            self._freeze_pretrained()

    def _freeze_pretrained(self) -> None:
        frozen = (
            self.backbone.frozen_modules()
            if hasattr(self.backbone, "frozen_modules")
            else [self.backbone]
        )
        for module in frozen:
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
                self._frozen_param_ids.add(id(p))

    def trainable_parameters(self):
        for p in self.parameters():
            if id(p) not in self._frozen_param_ids:
                yield p

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        b, t = o.shape[:2]
        x = o.reshape(b * t, *o.shape[2:])
        tokens = self.backbone(x)
        pooled = tokens.mean(dim=1)
        x = self.head(pooled)
        return x.reshape(b, t, self.config.embed_dim)
