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


def _obs(cfg: SSWMConfig, b: int = 4) -> "torch.Tensor":
    return torch.randn(b, cfg.seq_len, cfg.obs_channels, cfg.n_subcarriers, cfg.n_antennas)


def _pair(cfg):
    ctx = ContextEncoder(cfg)
    tgt = TargetEncoder(ctx, cfg)
    return ctx, tgt


def test_identical_architecture_and_param_count():
    cfg = _cfg()
    ctx, tgt = _pair(cfg)
    cp = [tuple(p.shape) for p in ctx.parameters()]
    tp = [tuple(p.shape) for p in tgt.encoder.parameters()]
    assert cp == tp


def test_same_input_same_output_at_init():
    cfg = _cfg()
    ctx, tgt = _pair(cfg)
    ctx.eval()
    o = _obs(cfg)
    with torch.no_grad():
        assert torch.allclose(ctx(o), tgt(o), atol=1e-6)


def test_stop_gradient_isolation():
    cfg = _cfg()
    ctx, tgt = _pair(cfg)
    o_ctx = _obs(cfg)
    o_tgt = _obs(cfg)
    z_online = ctx(o_ctx)
    z_target = tgt(o_tgt)
    loss = (z_online - z_target).pow(2).mean()
    loss.backward()
    assert all(p.grad is None for p in tgt.parameters())
    assert any(p.grad is not None for p in ctx.trainable_parameters())


def test_target_lags_then_tracks_online():
    cfg = _cfg(ema_momentum=0.9)
    ctx, tgt = _pair(cfg)
    o = _obs(cfg)
    ctx.eval()

    def gap():
        with torch.no_grad():
            return (ctx(o) - tgt(o)).pow(2).mean().item()

    assert gap() < 1e-10
    opt = torch.optim.SGD(ctx.trainable_parameters(), lr=0.5)
    for _ in range(3):
        loss = ctx(o).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    gap_no_ema = gap()
    assert gap_no_ema > 1e-6
    prev = gap_no_ema
    for _ in range(40):
        tgt.ema_update(ctx)
        g = gap()
        assert g <= prev + 1e-9
        prev = g
    assert prev < gap_no_ema


def test_full_jepa_step_reduces_distance():
    cfg = _cfg(ema_momentum=0.99)
    ctx, tgt = _pair(cfg)
    ctx.train()
    o_ctx = _obs(cfg, b=8)
    o_tgt = _obs(cfg, b=8)
    opt = torch.optim.Adam(ctx.trainable_parameters(), lr=1e-2)

    first = None
    last = None
    for step in range(30):
        z_online = ctx(o_ctx)
        with torch.no_grad():
            z_target = tgt(o_tgt)
        loss = (z_online - z_target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tgt.ema_update(ctx)
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


def test_no_representation_collapse_under_self_distillation():
    cfg = _cfg(ema_momentum=0.99)
    ctx, tgt = _pair(cfg)
    ctx.train()
    o = _obs(cfg, b=16)
    opt = torch.optim.Adam(ctx.trainable_parameters(), lr=1e-2)
    for _ in range(50):
        z_online = ctx(o)
        with torch.no_grad():
            z_target = tgt(o)
        loss = (z_online - z_target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tgt.ema_update(ctx)
    with torch.no_grad():
        z = ctx(o).reshape(-1, cfg.embed_dim)
    per_dim_std = z.std(dim=0)
    assert (per_dim_std > 1e-3).float().mean() > 0.5


def test_ema_momentum_one_freezes_target():
    cfg = _cfg(ema_momentum=1.0)
    ctx, tgt = _pair(cfg)
    snap = [p.clone() for p in tgt.encoder.parameters()]
    with torch.no_grad():
        for p in ctx.trainable_parameters():
            p.add_(torch.randn_like(p))
    tgt.ema_update(ctx)
    for p, s in zip(tgt.encoder.parameters(), snap):
        assert torch.allclose(p, s)


def test_ema_momentum_zero_copies_online():
    cfg = _cfg(ema_momentum=0.0)
    ctx, tgt = _pair(cfg)
    with torch.no_grad():
        for p in ctx.trainable_parameters():
            p.add_(torch.randn_like(p))
    tgt.ema_update(ctx)
    o = _obs(cfg)
    ctx.eval()
    with torch.no_grad():
        assert torch.allclose(ctx(o), tgt(o), atol=1e-6)


def test_target_and_online_share_no_storage():
    cfg = _cfg()
    ctx, tgt = _pair(cfg)
    online_ptrs = {p.data_ptr() for p in ctx.parameters()}
    assert all(p.data_ptr() not in online_ptrs for p in tgt.encoder.parameters())
