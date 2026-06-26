from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder


def _cfg(**kw) -> SSWMConfig:
    base = dict(
        n_subcarriers=32,
        n_antennas=32,
        seq_len=4,
        horizon_k=2,
        embed_dim=256,
        use_pretrained=False,
    )
    base.update(kw)
    return SSWMConfig(**base)


def _obs(cfg: SSWMConfig, b: int = 2) -> "torch.Tensor":
    return torch.randn(b, cfg.seq_len, cfg.obs_channels, cfg.n_subcarriers, cfg.n_antennas)


def test_output_shape():
    cfg = _cfg()
    enc = ContextEncoder(cfg)
    out = enc(_obs(cfg, b=2))
    assert out.shape == (2, cfg.seq_len, cfg.embed_dim)


def test_head_trainable_and_gradients_flow():
    cfg = _cfg()
    enc = ContextEncoder(cfg)
    out = enc(_obs(cfg, b=2))
    out.pow(2).mean().backward()
    assert all(p.requires_grad for p in enc.head.parameters())
    assert any(p.grad is not None for p in enc.head.parameters())


def test_variable_grid_size():
    cfg = _cfg(n_subcarriers=16, n_antennas=64)
    enc = ContextEncoder(cfg)
    assert enc(_obs(cfg, b=3)).shape == (3, cfg.seq_len, cfg.embed_dim)


@pytest.mark.parametrize("backbone", ["lwm", "ijepa", "stub"])
def test_backbone_is_pluggable(backbone):
    cfg = _cfg(backbone=backbone, use_pretrained=(backbone != "stub"))
    enc = ContextEncoder(cfg)
    out = enc(_obs(cfg, b=2))
    assert out.shape == (2, cfg.seq_len, cfg.embed_dim)


def test_lwm_pretrained_loads_and_freezes():
    cfg = _cfg(backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg)
    if not enc.backbone.is_pretrained:
        pytest.skip("LWM weights unavailable offline")
    for m in enc.backbone.frozen_modules():
        assert all(not p.requires_grad for p in m.parameters())
    trainable_ids = {id(p) for p in enc.trainable_parameters()}
    frozen_ids = {id(p) for m in enc.backbone.frozen_modules() for p in m.parameters()}
    assert trainable_ids.isdisjoint(frozen_ids)


def test_lwm_tokenization_shape():
    cfg = _cfg(backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg)
    if not enc.backbone.is_pretrained:
        pytest.skip("LWM weights unavailable offline")
    tokens = enc.backbone._tokenize(torch.randn(5, 2, 32, 32))
    n_patches = (32 // cfg.lwm_patch_rows) * (32 // cfg.lwm_patch_cols)
    assert tokens.shape == (5, n_patches + 1, cfg.lwm_element_length)


def test_lwm_gradients_reach_head_not_backbone():
    cfg = _cfg(backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg)
    if not enc.backbone.is_pretrained:
        pytest.skip("LWM weights unavailable offline")
    out = enc(_obs(cfg, b=2))
    out.pow(2).mean().backward()
    assert any(p.grad is not None for p in enc.head.parameters())
    assert all(p.grad is None for m in enc.backbone.frozen_modules() for p in m.parameters())


def test_train_mode_keeps_frozen_backbone_deterministic():
    cfg = _cfg(backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg)
    if not enc.backbone.is_pretrained:
        pytest.skip("LWM weights unavailable offline")
    enc.train()
    assert enc.head.training is True
    for m in enc.backbone.frozen_modules():
        assert all(not sub.training for sub in m.modules())
    o = _obs(cfg, b=2)
    assert torch.allclose(enc(o), enc(o))


def test_trainable_parameters_robust_to_deepcopy():
    import copy

    cfg = _cfg(backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg)
    clone = copy.deepcopy(enc)
    n_src = sum(p.numel() for p in enc.trainable_parameters())
    n_clone = sum(p.numel() for p in clone.trainable_parameters())
    assert n_src == n_clone
    assert n_clone < sum(p.numel() for p in clone.parameters())
