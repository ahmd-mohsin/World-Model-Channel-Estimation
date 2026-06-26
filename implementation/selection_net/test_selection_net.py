from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.selection_net import SelectionNet


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32,
        n_antennas=32,
        seq_len=8,
        horizon_k=2,
        action_dim=8,
        state_dim=64,
        use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _actions(cfg, b=4):
    return torch.randn(b, cfg.seq_len, cfg.action_dim)


def test_output_shapes():
    cfg = _cfg()
    net = SelectionNet(cfg)
    A, B, C, dt = net(_actions(cfg, b=4))
    for t in (A, B, C, dt):
        assert t.shape == (4, cfg.seq_len, cfg.state_dim)


def test_A_strictly_negative():
    cfg = _cfg()
    net = SelectionNet(cfg)
    A, *_ = net(_actions(cfg, b=8))
    assert (A < 0).all()


def test_dt_strictly_positive():
    cfg = _cfg()
    net = SelectionNet(cfg)
    *_, dt = net(_actions(cfg, b=8))
    assert (dt > 0).all()


def test_dt_init_in_configured_range():
    cfg = _cfg(dt_min=1e-3, dt_max=1e-1)
    net = SelectionNet(cfg)
    a = torch.zeros(1, 1, cfg.action_dim)
    *_, dt = net(a)
    assert dt.min().item() >= cfg.dt_min * 0.5
    assert dt.max().item() <= cfg.dt_max * 2.0


def test_A_init_spread_of_timescales():
    cfg = _cfg()
    net = SelectionNet(cfg)
    a = torch.zeros(1, 1, cfg.action_dim)
    A, *_ = net(a)
    vals = (-A[0, 0]).sort().values
    assert vals[0] < vals[-1]
    assert vals[-1] > 5.0


def test_selectivity_different_actions_give_different_params():
    cfg = _cfg()
    net = SelectionNet(cfg)
    a1 = torch.zeros(1, 1, cfg.action_dim)
    a2 = torch.ones(1, 1, cfg.action_dim)
    p1 = net(a1)
    p2 = net(a2)
    diffs = [not torch.allclose(x, y) for x, y in zip(p1, p2)]
    assert all(diffs)


def test_gradients_reach_all_params():
    cfg = _cfg()
    net = SelectionNet(cfg)
    A, B, C, dt = net(_actions(cfg, b=4))
    loss = A.pow(2).mean() + B.pow(2).mean() + C.pow(2).mean() + dt.pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_deterministic_in_eval():
    cfg = _cfg()
    net = SelectionNet(cfg).eval()
    a = _actions(cfg, b=4)
    p1 = net(a)
    p2 = net(a)
    for x, y in zip(p1, p2):
        assert torch.allclose(x, y)


def test_handles_flat_batch_shape():
    cfg = _cfg()
    net = SelectionNet(cfg)
    a = torch.randn(5, cfg.action_dim)
    A, B, C, dt = net(a)
    for t in (A, B, C, dt):
        assert t.shape == (5, cfg.state_dim)


def test_no_nan_on_extreme_actions():
    cfg = _cfg()
    net = SelectionNet(cfg)
    a = torch.full((2, cfg.seq_len, cfg.action_dim), 1e4)
    out = net(a)
    for t in out:
        assert torch.isfinite(t).all()
