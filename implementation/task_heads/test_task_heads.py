from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT.parent))

from implementation.config import SSWMConfig
from implementation.task_heads import TaskHeads, add_noise, ls_estimate, mmse_estimate, nmse


def _cfg(**kw) -> SSWMConfig:
    base = dict(n_subcarriers=16, n_antennas=8, action_dim=4,
                embed_dim=64, latent_dim=64, use_pretrained=False, channel_head="mlp")
    base.update(kw)
    return SSWMConfig(**base)


def _obs(cfg, b):
    return torch.randn(b, cfg.obs_channels, cfg.n_antennas, cfg.n_subcarriers)


def test_head_output_shapes():
    cfg = _cfg()
    th = TaskHeads(cfg)
    z = torch.randn(4, cfg.latent_dim)
    out = th(z, _obs(cfg, 4))
    assert out["channel"].shape == (4, cfg.obs_channels * cfg.n_subcarriers * cfg.n_antennas)
    assert out["reward"].shape == (4, 1)
    assert out["policy"].shape == (4, cfg.action_dim)


def test_channel_grid_reshape():
    cfg = _cfg()
    th = TaskHeads(cfg)
    z = torch.randn(3, cfg.latent_dim)
    g = th.channel_grid(z, _obs(cfg, 3))
    assert g.shape == (3, cfg.obs_channels, cfg.n_antennas, cfg.n_subcarriers)


def test_channel_head_starts_at_ls():
    # With obs skip + zero-init residual, the channel head starts as the identity (= LS).
    cfg = _cfg()
    th = TaskHeads(cfg)
    z = torch.randn(2, cfg.latent_dim)
    obs = _obs(cfg, 2)
    out = th(z, obs)["channel"]
    assert torch.allclose(out, obs.reshape(2, -1), atol=1e-6)


def test_latent_only_mode():
    cfg = _cfg()
    th = TaskHeads(cfg, channel_use_obs=False)
    out = th(torch.randn(2, cfg.latent_dim))
    assert out["channel"].shape == (2, cfg.obs_channels * cfg.n_subcarriers * cfg.n_antennas)


def test_subset_of_heads():
    cfg = _cfg()
    th = TaskHeads(cfg, heads=("channel",))
    out = th(torch.randn(2, cfg.latent_dim), _obs(cfg, 2))
    assert set(out.keys()) == {"channel"}


def test_probe_does_not_touch_encoder():
    cfg = _cfg()
    th = TaskHeads(cfg)
    z = torch.randn(4, cfg.latent_dim)
    loss = th(z, _obs(cfg, 4))["channel"].pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in th.heads["channel"].parameters())


def test_channel_head_overfits():
    torch.manual_seed(0)
    cfg = _cfg()
    th = TaskHeads(cfg, heads=("channel",))
    z = torch.randn(16, cfg.latent_dim)
    obs = _obs(cfg, 16)
    target = torch.randn(16, cfg.obs_channels * cfg.n_subcarriers * cfg.n_antennas)
    opt = torch.optim.Adam(th.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        loss = (th(z, obs)["channel"] - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0: first = loss.item()
        last = loss.item()
    assert last < first * 0.2


# ---- classical baselines ----

def test_ls_equals_observation():
    Y = torch.randn(4, 2, 8, 16)
    assert torch.allclose(ls_estimate(Y), Y)


def test_mmse_beats_ls_at_low_snr():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    # correlated channels (low-rank) so MMSE has structure to exploit
    cfg = _cfg()
    base = torch.randn(200, 2, cfg.n_antennas, 2, generator=g)
    mix = torch.randn(2, cfg.n_subcarriers, generator=g)
    H = torch.einsum("bcat,ts->bcas", base, mix)  # (200,2,ant,sub) correlated across sub
    train, test = H[:150], H[150:]
    snr = 0.0
    Y = add_noise(test, snr, generator=g)
    ls = nmse(ls_estimate(Y), test)
    mmse = nmse(mmse_estimate(Y, train, snr), test)
    assert mmse < ls  # MMSE denoises; must beat raw LS at low SNR


def test_add_noise_respects_snr():
    torch.manual_seed(0)
    H = torch.randn(64, 2, 8, 16)
    hi = add_noise(H, 30.0)
    lo = add_noise(H, 0.0)
    assert (hi - H).pow(2).mean() < (lo - H).pow(2).mean()


# ---- U-Net channel head ----

def test_unet_head_shape_and_starts_at_ls():
    # Non-predictive U-Net: zero-init residual => starts exactly at LS (the observation).
    cfg = _cfg(channel_head="unet", predictive_estimation=False)
    th = TaskHeads(cfg, heads=("channel",))
    z = torch.randn(4, cfg.latent_dim)
    obs = _obs(cfg, 4)
    nv = torch.rand(4)
    out = th(z, obs, noise_var=nv)["channel"]
    assert out.shape == (4, cfg.obs_channels * cfg.n_antennas * cfg.n_subcarriers)
    assert torch.allclose(out, obs.reshape(4, -1), atol=1e-6)


def test_unet_head_predictive_fusion():
    # Predictive head: output depends on the prior latent (Kalman fusion path is wired).
    cfg = _cfg(channel_head="unet", predictive_estimation=True)
    th = TaskHeads(cfg, heads=("channel",))
    with torch.no_grad():
        for p in th.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    z = torch.randn(3, cfg.latent_dim); obs = _obs(cfg, 3); nv = torch.rand(3)
    o1 = th(z, obs, noise_var=nv, prior_latent=torch.zeros(3, cfg.latent_dim))["channel"]
    o2 = th(z, obs, noise_var=nv, prior_latent=torch.randn(3, cfg.latent_dim))["channel"]
    assert not torch.allclose(o1, o2)   # the world-model prior actually influences the estimate


def test_unet_head_uses_noise_and_latent():
    cfg = _cfg(channel_head="unet")
    th = TaskHeads(cfg, heads=("channel",))
    # perturb weights so it's no longer the zero-init identity
    with torch.no_grad():
        for p in th.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    z = torch.randn(2, cfg.latent_dim); obs = _obs(cfg, 2)
    o1 = th(z, obs, noise_var=torch.zeros(2))["channel"]
    o2 = th(z, obs, noise_var=torch.ones(2))["channel"]
    assert not torch.allclose(o1, o2)             # depends on noise var
    o3 = th(torch.randn(2, cfg.latent_dim), obs, noise_var=torch.zeros(2))["channel"]
    assert not torch.allclose(o1, o3)             # depends on latent


def test_unet_head_overfits():
    torch.manual_seed(0)
    cfg = _cfg(channel_head="unet")
    th = TaskHeads(cfg, heads=("channel",))
    z = torch.randn(8, cfg.latent_dim); obs = _obs(cfg, 8); nv = torch.rand(8)
    target = torch.randn(8, cfg.obs_channels * cfg.n_antennas * cfg.n_subcarriers)
    opt = torch.optim.Adam(th.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        loss = (th(z, obs, noise_var=nv)["channel"] - target).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s == 0: first = loss.item()
        last = loss.item()
    assert last < first * 0.3
