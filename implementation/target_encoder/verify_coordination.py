from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.target_encoder import TargetEncoder

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SSWMConfig(n_subcarriers=32, n_antennas=32, seq_len=4, horizon_k=2,
                     embed_dim=256, backbone="lwm", use_pretrained=True,
                     ema_momentum=0.99)
    ctx = ContextEncoder(cfg).to(dev)
    tgt = TargetEncoder(ctx, cfg).to(dev)
    ctx.train()

    print("=" * 68)
    print("Context <-> Target coordination (JEPA self-distillation)")
    print("=" * 68)
    print(f"device                      : {dev}")
    print(f"backbone pretrained         : {ctx.backbone.is_pretrained}")
    print(f"online trainable params     : {sum(p.numel() for p in ctx.trainable_parameters()):,}")
    print(f"target trainable params     : {sum(p.numel() for p in tgt.parameters() if p.requires_grad):,} (must be 0)")
    print(f"shared frozen LWM weights   : {sum(p.numel() for p in ctx.backbone.parameters()):,}")
    print(f"ema momentum                : {cfg.ema_momentum}")

    torch.manual_seed(0)
    o = torch.randn(16, cfg.seq_len, 2, 32, 32, device=dev)
    proxy = torch.randn(16, cfg.seq_len, cfg.embed_dim, device=dev)
    opt = torch.optim.Adam(ctx.trainable_parameters(), lr=5e-3)

    losses, online_var, ema_gap = [], [], []
    theory_gap = []
    g0 = None
    m = cfg.ema_momentum
    for step in range(120):
        z_online = ctx(o)
        loss = (z_online - proxy).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tgt.ema_update(ctx)

        losses.append(loss.item())
        with torch.no_grad():
            online_var.append(z_online.reshape(-1, cfg.embed_dim).std(0).mean().item())
            g = (ctx(o) - tgt(o)).pow(2).mean().item()
            ema_gap.append(g)
            if g0 is None and g > 0:
                g0 = g

    print(f"\nproxy-task loss      : {losses[0]:.4f} -> {losses[-1]:.4f}  ({'DOWN (head learns through frozen LWM)' if losses[-1] < losses[0] else 'UP!'})")
    print(f"online embedding std : {online_var[0]:.4f} -> {online_var[-1]:.4f}  (> 0 => no collapse)")
    print(f"online/target gap    : peaked then -> {ema_gap[-1]:.5f}  (target tracks online via EMA)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(losses)
    axes[0].set_title("Proxy-task loss (online head trained\nthrough FROZEN LWM) -> head is learnable")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("MSE"); axes[0].grid(alpha=0.3)

    axes[1].plot(online_var, label="online z std")
    axes[1].axhline(0, ls="--", c="r", alpha=0.5, label="collapse floor")
    axes[1].set_ylim(bottom=-0.05)
    axes[1].set_title("Mean per-dim embedding std\n(stays > 0 => NOT collapsed)")
    axes[1].set_xlabel("step"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ema_gap, label="measured ||online - target||^2")
    axes[2].set_title(f"EMA tracking gap (m={m})\ntarget slow-follows the moving online net")
    axes[2].set_xlabel("step"); axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.suptitle("ContextEncoder + TargetEncoder coordination (frozen LWM + EMA self-distillation)", fontsize=12)
    p = OUT / "05_coordination.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {p}")
    print("=" * 68)


if __name__ == "__main__":
    main()
