from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.wireless_data import WirelessDataset, SionnaSpec


def _cfg(**kw) -> SSWMConfig:
    base = dict(n_subcarriers=32, n_antennas=8, seq_len=4, horizon_k=2, use_pretrained=False)
    base.update(kw)
    return SSWMConfig(**base)


def _ds(cfg, n=4):
    return WirelessDataset(cfg, n_samples=n, spec=SionnaSpec(scene="munich"), seed=0)


def test_sequence_shape():
    cfg = _cfg()
    x = _ds(cfg)[0]
    assert x.shape == (cfg.seq_len, 2, cfg.n_antennas, cfg.n_subcarriers)
    assert torch.isfinite(x).all()


def test_lwm_scale_magnitude():
    # LWM-style global scaling (x1e6) brings raw channels to O(1) while preserving
    # cross-antenna/subcarrier amplitude variation. Not bounded to 1, but finite and O(1)-O(100).
    cfg = _cfg()
    x = _ds(cfg)[1]
    mag = (x[:, 0] ** 2 + x[:, 1] ** 2).sqrt()
    assert torch.isfinite(x).all()
    assert 1e-3 < mag.mean().item() < 1e3


def test_repeatable_per_index():
    # Sionna's ray sampler has tiny Monte-Carlo noise; same index -> near-identical
    # relative to the channel scale (channels are O(1)-O(100) after the LWM x1e6 scaling).
    cfg = _cfg()
    ds = _ds(cfg)
    a, b = ds[2], ds[2]
    rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-9)
    assert rel < 1e-2


def test_batch_shape():
    cfg = _cfg()
    b = _ds(cfg, n=8).batch(3)
    assert b.shape == (3, cfg.seq_len, 2, cfg.n_antennas, cfg.n_subcarriers)


def test_temporal_correlation():
    cfg = _cfg(seq_len=6)
    x = _ds(cfg)[0]
    adjacent = (x[1:] - x[:-1]).pow(2).mean().item()
    distant = (x[3:] - x[:-3]).pow(2).mean().item()
    assert adjacent <= distant + 1e-6


def test_feeds_context_encoder():
    from implementation.context_encoder import ContextEncoder
    cfg = _cfg()
    o = _ds(cfg).batch(2)
    enc = ContextEncoder(cfg)
    x = enc(o)
    assert x.shape == (2, cfg.seq_len, cfg.embed_dim)
