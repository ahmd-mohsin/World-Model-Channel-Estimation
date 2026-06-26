from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.target_encoder import TargetEncoder


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32,
        n_antennas=32,
        seq_len=4,
        horizon_k=2,
        embed_dim=64,
        use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _obs(cfg: SSWMConfig, b: int = 2) -> "torch.Tensor":
    return torch.randn(b, cfg.seq_len, cfg.obs_channels, cfg.n_subcarriers, cfg.n_antennas)


def test_output_shape_and_detached():
    cfg = _cfg()
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    out = tgt(_obs(cfg, b=2))
    assert out.shape == (2, cfg.seq_len, cfg.embed_dim)
    assert out.requires_grad is False


def test_all_params_frozen():
    cfg = _cfg()
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    assert all(not p.requires_grad for p in tgt.parameters())


def test_hard_init_matches_source():
    cfg = _cfg()
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    o = _obs(cfg, b=2)
    ctx.eval()
    with torch.no_grad():
        a = ctx(o)
    b = tgt(o)
    assert torch.allclose(a, b, atol=1e-6)


def test_ema_moves_toward_source():
    cfg = _cfg(ema_momentum=0.9)
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)

    head = next(iter(ctx.head.parameters()))
    tgt_head = next(iter(tgt.encoder.head.parameters()))
    before = tgt_head.clone()

    with torch.no_grad():
        head.add_(torch.ones_like(head))

    tgt.ema_update(ctx)
    expected = 0.9 * before + 0.1 * head
    assert torch.allclose(tgt_head, expected, atol=1e-6)
    assert not torch.allclose(tgt_head, before)


def test_ema_leaves_target_grad_free():
    cfg = _cfg()
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    out = ctx(_obs(cfg, b=2))
    out.pow(2).mean().backward()
    tgt.ema_update(ctx)
    assert all(p.grad is None for p in tgt.parameters())


def test_frozen_backbone_unchanged_by_ema():
    cfg = _cfg()
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    snap = [p.clone() for p in tgt.encoder.backbone.parameters()]
    with torch.no_grad():
        for p in ctx.backbone.parameters():
            p.add_(torch.ones_like(p))
    tgt.ema_update(ctx)
    for p, s in zip(tgt.encoder.backbone.parameters(), snap):
        assert torch.allclose(p, s, atol=1e-6)
