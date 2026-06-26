from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.predictor import Predictor


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32, n_antennas=32, seq_len=8, horizon_k=4,
        action_dim=8, state_dim=16, embed_dim=32, latent_dim=32, use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _inputs(cfg, b=4, k=None):
    k = cfg.horizon_k if k is None else k
    return torch.randn(b, cfg.latent_dim), torch.randn(b, k, cfg.action_dim)


def test_output_shape():
    cfg = _cfg()
    pred = Predictor(cfg)
    z, acts = _inputs(cfg, b=4)
    out = pred(z, acts)
    assert out.shape == (4, cfg.embed_dim)


def test_output_in_embed_space_not_latent():
    # JEPA compares against target-encoder output (embed_dim). Guard against silently
    # using latent_dim by making them differ.
    cfg = _cfg(embed_dim=24, latent_dim=40)
    pred = Predictor(cfg)
    z = torch.randn(3, cfg.latent_dim)
    acts = torch.randn(3, cfg.horizon_k, cfg.action_dim)
    assert pred(z, acts).shape == (3, cfg.embed_dim)


def test_forward_matches_rollout_last_step():
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z, acts = _inputs(cfg, b=5)
    last = pred(z, acts)
    traj = pred.rollout(z, acts)
    assert traj.shape == (5, cfg.horizon_k, cfg.embed_dim)
    assert torch.allclose(last, traj[:, -1], atol=1e-6)


def test_depends_on_actions():
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z = torch.randn(2, cfg.latent_dim)
    a1 = torch.zeros(2, cfg.horizon_k, cfg.action_dim)
    a2 = torch.ones(2, cfg.horizon_k, cfg.action_dim)
    assert not torch.allclose(pred(z, a1), pred(z, a2))


def test_depends_on_z():
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    acts = torch.randn(2, cfg.horizon_k, cfg.action_dim)
    z1 = torch.zeros(2, cfg.latent_dim)
    z2 = torch.randn(2, cfg.latent_dim)
    assert not torch.allclose(pred(z1, acts), pred(z2, acts))


def test_horizon_changes_prediction():
    # Predicting k=1 vs k=3 ahead should differ (different number of rollout steps).
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z = torch.randn(2, cfg.latent_dim)
    acts = torch.randn(2, 3, cfg.action_dim)
    p1 = pred(z, acts[:, :1])
    p3 = pred(z, acts[:, :3])
    assert not torch.allclose(p1, p3)


def test_only_uses_provided_action_prefix():
    # Prediction at horizon j must NOT depend on actions a_{>=j} (causality / no leak).
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z = torch.randn(2, cfg.latent_dim)
    acts = torch.randn(2, 4, cfg.action_dim)
    acts_mod = acts.clone()
    acts_mod[:, 2:] += 10.0  # change only future actions beyond horizon 2
    p_short = pred(z, acts[:, :2])
    p_short_mod = pred(z, acts_mod[:, :2])
    assert torch.allclose(p_short, p_short_mod, atol=1e-6)


def test_k_zero_returns_projection_of_z():
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z = torch.randn(3, cfg.latent_dim)
    acts = torch.randn(3, 0, cfg.action_dim)
    out = pred(z, acts)
    assert out.shape == (3, cfg.embed_dim)
    assert torch.isfinite(out).all()


def test_gradients_flow():
    cfg = _cfg()
    pred = Predictor(cfg)
    z, acts = _inputs(cfg, b=4)
    z.requires_grad_(True)
    pred(z, acts).pow(2).mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert all(p.grad is not None for p in pred.selection.parameters())
    assert all(p.grad is not None for p in pred.out_proj.parameters())


def test_stable_over_long_horizon():
    cfg = _cfg()
    pred = Predictor(cfg).eval()
    z = torch.randn(2, cfg.latent_dim)
    acts = torch.randn(2, 100, cfg.action_dim)
    out = pred(z, acts)
    assert torch.isfinite(out).all() and out.abs().max().item() < 1e4


def test_overfits_ar1_target():
    # README M3 test: predicted ẑ_{t+k} tracks a learnable function of (z_t, actions).
    torch.manual_seed(0)
    cfg = _cfg(state_dim=32)
    pred = Predictor(cfg)
    z = torch.randn(16, cfg.latent_dim)
    acts = torch.randn(16, cfg.horizon_k, cfg.action_dim)
    target = torch.randn(16, cfg.embed_dim)
    opt = torch.optim.Adam(pred.parameters(), lr=1e-2)
    first = last = None
    for step in range(200):
        loss = (pred(z, acts) - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0: first = loss.item()
        last = loss.item()
    assert last < first * 0.5
