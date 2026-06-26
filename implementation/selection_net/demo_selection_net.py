from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.config import SSWMConfig
from implementation.selection_net import SelectionNet


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SSWMConfig(action_dim=8, state_dim=64, seq_len=8, horizon_k=2, use_pretrained=False)
    net = SelectionNet(cfg).to(dev).eval()

    print("=" * 66)
    print("SelectionNet  ::  a_t -> (A_t, B_t, C_t, Δ_t)  [selective SSM params]")
    print("=" * 66)
    print(f"device              : {dev}")
    print(f"action_dim          : {cfg.action_dim}")
    print(f"state_dim           : {cfg.state_dim}")
    print(f"params              : {sum(p.numel() for p in net.parameters()):,}")

    a = torch.randn(4, cfg.seq_len, cfg.action_dim, device=dev)
    A, B, C, dt = net(a)
    print(f"\nactions a_t         : {tuple(a.shape)}")
    for name, t in [("A_t", A), ("B_t", B), ("C_t", C), ("Δ_t", dt)]:
        print(f"{name:5s} {tuple(t.shape)}  min={t.min().item():+.4f} max={t.max().item():+.4f} mean={t.mean().item():+.4f}")

    a0 = torch.zeros(1, 1, cfg.action_dim, device=dev)
    A0, _, _, dt0 = net(a0)
    print("\n--- init properties (zero action) ---")
    print(f"A timescales (-A) range : {(-A0[0,0]).min().item():.3f} .. {(-A0[0,0]).max().item():.3f}")
    print(f"Δ range                 : {dt0.min().item():.4f} .. {dt0.max().item():.4f}  (cfg {cfg.dt_min}..{cfg.dt_max})")
    print(f"all A<0: {(A<0).all().item()} | all Δ>0: {(dt>0).all().item()}")

    a1 = torch.zeros(1, 1, cfg.action_dim, device=dev)
    a2 = torch.ones(1, 1, cfg.action_dim, device=dev)
    d = sum((x - y).abs().mean().item() for x, y in zip(net(a1), net(a2)))
    print(f"\nselectivity (|params(0) - params(1)|): {d:.4f}  (> 0 => action-dependent)")
    print("=" * 66)


if __name__ == "__main__":
    main()
