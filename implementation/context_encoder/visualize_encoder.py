from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def make_channel(n_ant, n_sub, n_paths, delay_spread, aod_center, snr_db, seed):
    g = torch.Generator().manual_seed(seed)
    ant = torch.arange(n_ant).view(-1, 1)
    sub = torch.arange(n_sub).view(1, -1)
    H = torch.zeros(n_ant, n_sub, dtype=torch.cfloat)
    for _ in range(n_paths):
        gain = torch.randn(1, generator=g) + 1j * torch.randn(1, generator=g)
        aod = aod_center + 0.1 * torch.randn(1, generator=g)
        delay = delay_spread * torch.rand(1, generator=g)
        steer = torch.exp(2j * np.pi * ant * aod / n_ant)
        freq = torch.exp(-2j * np.pi * sub * delay / n_sub)
        H = H + gain * steer * freq
    power = H.abs().pow(2).mean()
    noise_std = (power / (10 ** (snr_db / 10))).sqrt()
    H = H + noise_std * (torch.randn(n_ant, n_sub, generator=g) + 1j * torch.randn(n_ant, n_sub, generator=g)) / np.sqrt(2)
    H = H / H.abs().max()
    return torch.stack([H.real, H.imag], dim=0)


def encode(enc, cfg, channels):
    o = channels.unsqueeze(1).expand(-1, cfg.seq_len, -1, -1, -1).contiguous()
    with torch.no_grad():
        x = enc(o)[:, 0]
    return x.numpy()


def encode_lwm_features(enc, channels):
    with torch.no_grad():
        tokens = enc.backbone(channels)
    return tokens[:, 1:].mean(1).numpy()


def fig_channels(cfg):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    configs = [
        ("1 path, narrow", dict(n_paths=1, delay_spread=5, aod_center=0.2, snr_db=30)),
        ("3 paths", dict(n_paths=3, delay_spread=15, aod_center=0.2, snr_db=30)),
        ("10 paths, rich", dict(n_paths=10, delay_spread=30, aod_center=0.5, snr_db=30)),
        ("10 paths, low SNR", dict(n_paths=10, delay_spread=30, aod_center=0.5, snr_db=0)),
    ]
    for j, (name, kw) in enumerate(configs):
        H = make_channel(cfg.n_antennas, cfg.n_subcarriers, seed=j, **kw)
        for i, part in enumerate(["real", "imag"]):
            ax = axes[i, j]
            im = ax.imshow(H[i], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{name}\n({part})" if i == 0 else f"({part})", fontsize=9)
            ax.set_xlabel("subcarrier") if i == 1 else None
            ax.set_ylabel("antenna") if j == 0 else None
    fig.suptitle("Input channels o_t  (real/imag, antenna x subcarrier)", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02)
    p = OUT / "01_input_channels.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pca_by_param(enc, cfg, feature_fn, tag, title):
    rows = []
    labels_paths, labels_aod, labels_snr = [], [], []
    seed = 0
    for n_paths in [1, 2, 4, 8, 16]:
        for aod in [0.1, 0.4, 0.7]:
            for snr in [30, 10]:
                for rep in range(6):
                    seed += 1
                    H = make_channel(cfg.n_antennas, cfg.n_subcarriers, n_paths, 25, aod, snr, seed)
                    rows.append(H)
                    labels_paths.append(n_paths)
                    labels_aod.append(aod)
                    labels_snr.append(snr)
    channels = torch.stack(rows)
    feats = feature_fn(enc, channels) if feature_fn is encode_lwm_features else feature_fn(enc, cfg, channels)
    pca = PCA(n_components=2)
    z = pca.fit_transform(feats)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc0 = axes[0].scatter(z[:, 0], z[:, 1], c=np.log2(labels_paths), cmap="viridis", s=30)
    axes[0].set_title("colored by #paths (log2)")
    fig.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(z[:, 0], z[:, 1], c=labels_aod, cmap="plasma", s=30)
    axes[1].set_title("colored by angle-of-departure")
    fig.colorbar(sc1, ax=axes[1])
    sc2 = axes[2].scatter(z[:, 0], z[:, 1], c=labels_snr, cmap="coolwarm", s=30)
    axes[2].set_title("colored by SNR (dB)")
    fig.colorbar(sc2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    fig.suptitle(title, fontsize=13)
    p = OUT / f"02_pca_{tag}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p, feats, np.array(labels_paths), np.array(labels_aod), np.array(labels_snr)


def make_channel_fixed(n_ant, n_sub, gains, aods, delays, snr_db, seed):
    ant = torch.arange(n_ant).view(-1, 1)
    sub = torch.arange(n_sub).view(1, -1)
    H = torch.zeros(n_ant, n_sub, dtype=torch.cfloat)
    for gain, aod, delay in zip(gains, aods, delays):
        steer = torch.exp(2j * np.pi * ant * aod / n_ant)
        freq = torch.exp(-2j * np.pi * sub * delay / n_sub)
        H = H + gain * steer * freq
    if snr_db is not None:
        g = torch.Generator().manual_seed(seed)
        power = H.abs().pow(2).mean()
        noise_std = (power / (10 ** (snr_db / 10))).sqrt()
        H = H + noise_std * (torch.randn(n_ant, n_sub, generator=g) + 1j * torch.randn(n_ant, n_sub, generator=g)) / np.sqrt(2)
    H = H / H.abs().max()
    return torch.stack([H.real, H.imag], dim=0)


def fig_smoothness(enc, cfg):
    aods = np.linspace(0.0, 1.0, 60)
    gain = torch.tensor([1.0 + 0.0j])
    delay = torch.tensor([4.0])
    feats = []
    for aod in aods:
        H = make_channel_fixed(cfg.n_antennas, cfg.n_subcarriers, gain, [float(aod)], delay, None, seed=0)
        feats.append(encode_lwm_features(enc, H.unsqueeze(0))[0])
    feats = np.array(feats)
    dist = np.linalg.norm(feats - feats[0], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(aods, dist, "-o", ms=3)
    axes[0].set_xlabel("angle-of-departure (single clean path)")
    axes[0].set_ylabel("LWM feature distance from AoD=0")
    axes[0].set_title("Feature trajectory vs AoD\n(gains/delay held fixed; only AoD swept)")
    axes[0].grid(alpha=0.3)

    delays = np.linspace(0.0, 30.0, 60)
    feats2 = []
    for d in delays:
        H = make_channel_fixed(cfg.n_antennas, cfg.n_subcarriers, gain, [0.25], torch.tensor([float(d)]), None, seed=0)
        feats2.append(encode_lwm_features(enc, H.unsqueeze(0))[0])
    feats2 = np.array(feats2)
    dist2 = np.linalg.norm(feats2 - feats2[0], axis=1)
    axes[1].plot(delays, dist2, "-o", ms=3, color="#c44e52")
    axes[1].set_xlabel("path delay (single clean path)")
    axes[1].set_ylabel("LWM feature distance from delay=0")
    axes[1].set_title("Feature trajectory vs delay\n(gains/AoD held fixed; only delay swept)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Do frozen LWM features move continuously when ONE physical parameter changes?", fontsize=12)
    p = OUT / "03_parameter_sweep.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_separability(feats, labels_paths, labels_snr):
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import cross_val_score

    X = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    rich = (labels_paths >= 8).astype(int)
    acc = cross_val_score(LogisticRegression(max_iter=1000), X, rich, cv=5).mean()
    r2 = cross_val_score(LinearRegression(), X, np.log2(labels_paths), cv=5, scoring="r2").mean()
    snr_acc = cross_val_score(LogisticRegression(max_iter=1000), X, (labels_snr == 30).astype(int), cv=5).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ["rich-vs-sparse\n(#paths>=8)", "#paths regression\n(log2, R^2)", "SNR\n(30 vs 10 dB)"]
    vals = [acc, max(r2, 0), snr_acc]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.axhline(0.5, ls="--", c="gray", label="chance (classification)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("5-fold CV score")
    ax.set_title("Linear probe on FROZEN LWM features recovers physical channel properties")
    ax.legend()
    p = OUT / "04_linear_probe.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p, acc, r2, snr_acc


def main():
    cfg = SSWMConfig(n_subcarriers=32, n_antennas=32, seq_len=4, horizon_k=2,
                     embed_dim=256, backbone="lwm", use_pretrained=True)
    enc = ContextEncoder(cfg).eval()
    print(f"backbone pretrained: {enc.backbone.is_pretrained}")

    p1 = fig_channels(cfg)
    print("saved", p1)

    p2, feats, lp, la, ls = fig_pca_by_param(enc, cfg, encode_lwm_features, "lwm_features",
                                             "PCA of FROZEN LWM channel embeddings  (PC1 tracks SNR; #paths/AoD entangled in higher dims)")
    print("saved", p2)

    p2b, *_ = fig_pca_by_param(enc, cfg, encode, "full_encoder",
                               "PCA of full ContextEncoder output x_t (LWM + untrained projection head)")
    print("saved", p2b)

    p3 = fig_smoothness(enc, cfg)
    print("saved", p3)

    p4, acc, r2, snr_acc = fig_separability(feats, lp, ls)
    print("saved", p4)

    print("\n=== linear-probe summary (frozen LWM features) ===")
    print(f"rich-vs-sparse accuracy : {acc:.3f}")
    print(f"#paths log2 regression R2: {r2:.3f}")
    print(f"SNR 30-vs-10 accuracy   : {snr_acc:.3f}")
    print(f"\nAll figures in: {OUT}")


if __name__ == "__main__":
    main()
