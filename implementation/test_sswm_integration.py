from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=2,
        action_dim=8, state_dim=16, embed_dim=32, latent_dim=32, use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _batch(cfg, b=4):
    o = torch.randn(b, cfg.seq_len, 2, cfg.n_antennas, cfg.n_subcarriers)
    a = torch.randn(b, cfg.seq_len, cfg.action_dim)
    return o, a


def test_full_forward_shapes():
    cfg = _cfg()
    m = SSWM(cfg)
    o, a = _batch(cfg)
    z_hat, z_tilde = m(o, a)
    assert z_hat.shape == (4, cfg.embed_dim)
    assert z_tilde.shape == (4, cfg.embed_dim)


def test_selection_net_is_shared():
    cfg = _cfg()
    m = SSWM(cfg)
    assert m.ssm.selection is m.predictor.selection
    assert m.ssm.selection is m.selection


def test_target_is_detached():
    cfg = _cfg()
    m = SSWM(cfg)
    o, a = _batch(cfg)
    _, z_tilde = m(o, a)
    assert z_tilde.requires_grad is False


def test_jepa_loss_grad_isolation():
    cfg = _cfg()
    m = SSWM(cfg)
    o, a = _batch(cfg)
    loss, _ = m.jepa_loss(o, a)
    loss.backward()
    # online path gets gradients
    assert any(p.grad is not None for p in m.context_encoder.trainable_parameters())
    assert any(p.grad is not None for p in m.predictor.out_proj.parameters())
    assert any(p.grad is not None for p in m.selection.parameters())
    # target encoder gets NONE
    assert all(p.grad is None for p in m.target_encoder.parameters())


def test_shared_selection_single_param_set():
    cfg = _cfg()
    m = SSWM(cfg)
    ids = [id(p) for p in m.trainable_parameters()]
    assert len(ids) == len(set(ids))  # no duplicate params despite sharing


def test_anchor_bounds_enforced():
    cfg = _cfg(seq_len=6, horizon_k=2)
    m = SSWM(cfg)
    o, a = _batch(cfg)
    with pytest.raises(ValueError):
        m(o, a, anchor=5)   # 5 + 2 >= 6
    with pytest.raises(ValueError):
        m(o, a, anchor=-1)


def test_default_anchor_is_valid():
    cfg = _cfg(seq_len=8, horizon_k=3)
    m = SSWM(cfg)
    o, a = _batch(cfg)
    z_hat, z_tilde = m(o, a)  # default anchor = T-1-k = 4
    assert z_hat.shape == (4, cfg.embed_dim)


def test_target_matches_encoder_at_init():
    # target encoder is a hard copy at init -> on the same future obs it equals the
    # online encoder's embedding (sanity that they share architecture/weights).
    cfg = _cfg()
    m = SSWM(cfg)
    m.context_encoder.eval()
    o, a = _batch(cfg)
    k, anchor = cfg.horizon_k, cfg.seq_len - 1 - cfg.horizon_k
    o_future = o[:, anchor + k].unsqueeze(1)
    with torch.no_grad():
        online = m.context_encoder(o_future)[:, 0]
        target = m.target_encoder(o_future)[:, 0]
    assert torch.allclose(online, target, atol=1e-5)


def test_full_training_step_reduces_loss():
    torch.manual_seed(0)
    cfg = _cfg()
    m = SSWM(cfg)
    m.context_encoder.train()
    o, a = _batch(cfg, b=8)
    opt = torch.optim.Adam(m.trainable_parameters(), lr=1e-2)
    first = last = None
    for step in range(40):
        loss, _ = m.jepa_loss(o, a)
        opt.zero_grad(); loss.backward(); opt.step()
        m.update_target()
        if step == 0: first = loss.item()
        last = loss.item()
    assert last < first


def test_no_collapse_after_training():
    torch.manual_seed(0)
    cfg = _cfg()
    m = SSWM(cfg)
    m.context_encoder.train()
    o, a = _batch(cfg, b=12)
    opt = torch.optim.Adam(m.trainable_parameters(), lr=1e-2)
    for _ in range(50):
        loss, _ = m.jepa_loss(o, a)
        opt.zero_grad(); loss.backward(); opt.step()
        m.update_target()
    _, metrics = m.jepa_loss(o, a)
    assert metrics["pred_std"] > 1e-3
    assert metrics["target_std"] > 1e-3
