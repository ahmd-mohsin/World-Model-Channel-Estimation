"""Scale-consistency contract: x_t, z_t, ẑ, z̃ all live at the same embedding magnitude."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM


def _cfg(**kw):
    base = dict(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, action_dim=4,
                embed_dim=64, state_dim=32, latent_dim=64, use_pretrained=False,
                normalize_embeddings=True, embed_scale=8.0)
    base.update(kw)
    return SSWMConfig(**base)


def _batch(cfg, b=4):
    return (torch.randn(b, cfg.seq_len, 2, cfg.n_antennas, cfg.n_subcarriers),
            torch.randn(b, cfg.seq_len, cfg.action_dim))


def test_all_embeddings_same_norm():
    cfg = _cfg()
    m = SSWM(cfg).eval()
    o, a = _batch(cfg)
    with torch.no_grad():
        x = m.context_encoder(o)[:, 0]            # x_t
        z = m.encode_sequence(o, a)[:, 0]         # z_t
        zt = m.target_encoder(o[:, :1])[:, 0]     # z̃
        zhat, ztilde = m(o, a)                    # ẑ, z̃ (final)
    target = cfg.embed_scale
    for name, emb in [("x_t", x), ("z_t", z), ("z~", zt), ("zhat", zhat), ("ztilde", ztilde)]:
        n = emb.norm(dim=-1)
        assert torch.allclose(n, torch.full_like(n, target), atol=1e-3), f"{name} norm {n.mean():.3f} != {target}"


def test_pred_and_target_std_match():
    # The scale drift we saw earlier (pred_std 2.4 vs target_std 5.4) must be gone.
    cfg = _cfg()
    m = SSWM(cfg).eval()
    o, a = _batch(cfg, b=16)
    with torch.no_grad():
        zhat, ztilde = m(o, a)
    r = zhat.std(0).mean() / ztilde.std(0).mean()
    assert 0.5 < r.item() < 2.0, f"pred/target std ratio {r.item():.2f} still mismatched"


def test_normalization_off_is_noop():
    cfg = _cfg(normalize_embeddings=False)
    m = SSWM(cfg).eval()
    o, a = _batch(cfg)
    with torch.no_grad():
        x = m.context_encoder(o)[:, 0]
    # without normalization, norms vary (not all equal to embed_scale)
    assert not torch.allclose(x.norm(dim=-1), torch.full((x.shape[0],), cfg.embed_scale), atol=1e-2)
