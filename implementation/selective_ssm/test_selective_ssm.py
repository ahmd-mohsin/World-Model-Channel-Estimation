from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.selective_ssm import SelectiveSSM, discretize


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32, n_antennas=32, seq_len=8, horizon_k=2,
        action_dim=8, state_dim=16, embed_dim=32, latent_dim=32, use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _xa(cfg, b=4):
    return (torch.randn(b, cfg.seq_len, cfg.embed_dim),
            torch.randn(b, cfg.seq_len, cfg.action_dim))


def test_output_shape():
    cfg = _cfg()
    ssm = SelectiveSSM(cfg)
    x, a = _xa(cfg, b=4)
    z = ssm(x, a)
    assert z.shape == (4, cfg.seq_len, cfg.latent_dim)


def test_scan_matches_reference_recurrence():
    cfg = _cfg(state_dim=8)
    ssm = SelectiveSSM(cfg).eval()
    b, t, d = 2, 6, cfg.state_dim
    u = torch.randn(b, t, d)
    A = -torch.rand(b, t, d) - 0.5
    B = torch.randn(b, t, d)
    C = torch.randn(b, t, d)
    dt = torch.rand(b, t, d) * 0.1 + 0.01

    y = ssm._scan(u, A, B, C, dt)

    dA = torch.exp(dt * A)
    dB = (dA - 1.0) / A * B
    h = torch.zeros(b, d)
    ref = []
    for i in range(t):
        h = dA[:, i] * h + dB[:, i] * u[:, i]
        ref.append(C[:, i] * h + ssm.D * u[:, i])
    ref = torch.stack(ref, dim=1)
    assert torch.allclose(y, ref, atol=1e-6)


def test_forward_equals_stepwise():
    cfg = _cfg()
    ssm = SelectiveSSM(cfg).eval()
    x, a = _xa(cfg, b=3)
    z_full = ssm(x, a)

    h = None
    zs = []
    for i in range(cfg.seq_len):
        z_i, h = ssm.step(x[:, i], a[:, i], h)
        zs.append(z_i)
    z_step = torch.stack(zs, dim=1)
    assert torch.allclose(z_full, z_step, atol=1e-5)


def test_stable_over_long_sequence():
    cfg = _cfg(seq_len=200)
    ssm = SelectiveSSM(cfg).eval()
    x, a = _xa(cfg, b=2)
    z = ssm(x, a)
    assert torch.isfinite(z).all()
    assert z.abs().max().item() < 1e4


def test_discretize_stability():
    A = -torch.rand(100) - 0.1
    B = torch.randn(100)
    dt = torch.rand(100) * 0.2 + 0.01
    dA, dB = discretize(A, B, dt)
    assert (dA.abs() < 1.0).all()
    assert torch.isfinite(dB).all()


def test_gradients_flow_to_inputs_and_selection():
    cfg = _cfg()
    ssm = SelectiveSSM(cfg)
    x, a = _xa(cfg, b=4)
    x.requires_grad_(True)
    z = ssm(x, a)
    z.pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in ssm.selection.parameters())
    assert all(p.grad is not None for p in ssm.in_proj.parameters())


def test_selectivity_actions_change_output():
    cfg = _cfg()
    ssm = SelectiveSSM(cfg).eval()
    x = torch.randn(1, cfg.seq_len, cfg.embed_dim)
    a1 = torch.zeros(1, cfg.seq_len, cfg.action_dim)
    a2 = torch.ones(1, cfg.seq_len, cfg.action_dim)
    assert not torch.allclose(ssm(x, a1), ssm(x, a2))


def test_temporal_causality():
    cfg = _cfg(seq_len=10)
    ssm = SelectiveSSM(cfg).eval()
    x, a = _xa(cfg, b=1)
    z_full = ssm(x, a)
    x2 = x.clone()
    x2[:, 7:] += 5.0
    z_pert = ssm(x2, a)
    assert torch.allclose(z_full[:, :7], z_pert[:, :7], atol=1e-5)
    assert not torch.allclose(z_full[:, 7:], z_pert[:, 7:], atol=1e-5)
