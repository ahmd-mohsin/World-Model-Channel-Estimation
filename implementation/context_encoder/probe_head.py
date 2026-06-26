from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.wireless_data import SionnaChannelGenerator, SionnaSpec


def build_labeled_set(cfg, gen, n_per_class=20, seed=0):
    gen._lazy_init()
    rng = np.random.default_rng(seed)
    base = np.array(gen.spec.rx_start, dtype=np.float64)
    feats_in, labels = [], []
    centers = [base + np.array([0, 0, 0]),
               base + np.array([20, -15, 0]),
               base + np.array([-18, 12, 0]),
               base + np.array([10, 25, 0]),
               base + np.array([-22, -20, 0]),
               base + np.array([28, 8, 0])]
    for cls, c in enumerate(centers):
        for _ in range(n_per_class):
            pos = c + rng.normal(0, 1.5, 3)
            pos[2] = gen.spec.rx_height
            H = gen._channel_at(pos).detach().cpu() * gen.spec.channel_scale
            feats_in.append(torch.stack([H.real, H.imag], 0))
            labels.append(cls)
    o = torch.stack(feats_in, 0).unsqueeze(1)
    return o, np.array(labels)


def _encode(enc, o, dev, noisy=False):
    with torch.no_grad():
        oo = o.to(dev)
        if noisy:
            power = oo.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True)
            np_ = power / (10 ** (10.0 / 10))  # 10 dB SNR
            oo = oo + torch.randn_like(oo) * np_.sqrt()
        return enc(oo)[:, 0].cpu().numpy()


def probe_clean(enc, o, labels, dev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    x = _encode(enc, o, dev)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return cross_val_score(LogisticRegression(max_iter=2000), x, labels, cv=5).mean()


def probe_noise_robust(enc, o, labels, dev):
    # Train on clean embeddings, TEST on noisy embeddings -> measures noise-invariance,
    # which the VICReg objective explicitly optimizes. A random head should be far weaker here.
    from sklearn.linear_model import LogisticRegression
    xc = _encode(enc, o, dev, noisy=False)
    xn = _encode(enc, o, dev, noisy=True)
    mu, sd = xc.mean(0), xc.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000).fit((xc - mu) / sd, labels)
    return clf.score((xn - mu) / sd, labels)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=2, horizon_k=1,
                     embed_dim=256, backbone="lwm", use_pretrained=True)
    gen = SionnaChannelGenerator(cfg, SionnaSpec(scene="munich"))

    print("Building labeled Sionna location set (6 clusters x 30)...")
    o, labels = build_labeled_set(cfg, gen, n_per_class=30)
    print(f"  samples: {o.shape[0]}, classes: {len(set(labels))}")

    enc_random = ContextEncoder(cfg).to(dev).eval()
    ckpt = Path(__file__).resolve().parent / "checkpoints" / "head_vicreg.pt"
    enc_trained = ContextEncoder(cfg).to(dev).eval()
    enc_trained.head.load_state_dict(torch.load(ckpt, map_location=dev)["head"])

    chance = 1.0 / len(set(labels))
    print("\n=== Linear-probe accuracy (post-head x_t) ===")
    print(f"chance: {chance:.3f}")
    rc, tc = probe_clean(enc_random, o, labels, dev), probe_clean(enc_trained, o, labels, dev)
    print(f"[clean]        random {rc:.3f} | trained {tc:.3f} | Δ {tc-rc:+.3f}")
    rn, tn = probe_noise_robust(enc_random, o, labels, dev), probe_noise_robust(enc_trained, o, labels, dev)
    print(f"[noise-robust] random {rn:.3f} | trained {tn:.3f} | Δ {tn-rn:+.3f}")
    print(f"\nverdict (noise-robust): {'TRAINED HEAD HELPS' if tn > rn + 0.02 else 'no clear improvement'}")


if __name__ == "__main__":
    main()
