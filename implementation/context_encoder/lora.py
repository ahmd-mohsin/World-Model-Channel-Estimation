from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank update: y = W0 x + (B A) x * scale.

    Base weights stay frozen; only A, B train. B is zero-init so the adapter starts as identity
    (output == frozen linear) and learns a correction — same principle as our residual predictor.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(module: nn.Module, r: int = 8, alpha: int = 16, targets=("W_Q", "W_V")) -> int:
    """Replace named nn.Linear children matching `targets` with LoRALinear. Returns count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, r, alpha, targets)
    return count
