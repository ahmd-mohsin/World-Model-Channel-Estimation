"""Channel-estimation headline: SSWM channel head vs classical LS / MMSE across SNRs.

Task: observe a NOISY channel, recover the clean channel. The SSWM encodes the noisy
observation into z, a probe channel head maps z -> channel estimate. Compared to LS
(= noisy observation) and linear-MMSE (Wiener filter from training covariance).

Run on the box (needs Sionna shards in --data_dir):
    python scripts/eval-channel-estimation.py --data_dir data/act
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.task_heads import TaskHeads, add_noise, ls_estimate, mmse_estimate, nmse
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--probe_steps", type=int, default=3000)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                     embed_dim=256, action_dim=4, state_dim=64, latent_dim=256,
                     backbone="lwm", use_pretrained=True)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.1, seed=0)
    # use the LAST frame of each sequence as the channel to estimate (standardized)
    train_o, train_a = ds.all("train", device=dev)
    test_o, test_a = ds.all("test", device=dev)
    t = cfg.seq_len - 1
    H_train_clean = train_o[:, t]            # (Ntr, 2, ant, sub) standardized clean channel
    H_test_clean = test_o[:, t]
    print(f"train {H_train_clean.shape[0]} | test {H_test_clean.shape[0]}", flush=True)

    m = SSWM(cfg).to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(0)
    g = torch.Generator(device=dev).manual_seed(0)
    print(f"\n{'SNR':>5} | {'LS':>8} {'MMSE':>8} {'SSWM':>8} | winner", flush=True)
    print("-" * 48, flush=True)
    results = {}
    for snr in args.snrs:
        # noisy observations
        Ytr = add_noise(H_train_clean, snr, generator=g)
        Yte = add_noise(H_test_clean, snr, generator=g)

        # train a fresh channel-head probe on the FROZEN SSWM latent of the noisy obs
        head = TaskHeads(cfg, in_dim=cfg.latent_dim, heads=("channel",)).to(dev)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        with torch.no_grad():
            # latent of noisy training obs (replace last frame with noisy, encode sequence)
            seq_tr = train_o.clone(); seq_tr[:, t] = Ytr
            Ztr = m.encode_sequence(seq_tr, train_a)[:, t]
            seq_te = test_o.clone(); seq_te[:, t] = Yte
            Zte = m.encode_sequence(seq_te, test_a)[:, t]
        tgt_tr = H_train_clean.reshape(H_train_clean.shape[0], -1)
        for step in range(args.probe_steps):
            idx = torch.from_numpy(rng.choice(Ztr.shape[0], 256, replace=False)).to(dev)
            loss = F.mse_loss(head(Ztr[idx], Ytr[idx])["channel"], tgt_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            sswm_est = head(Zte, Yte)["channel"].reshape(H_test_clean.shape)
        ls = nmse(ls_estimate(Yte), H_test_clean)
        mm = nmse(mmse_estimate(Yte, H_train_clean, snr), H_test_clean)
        sw = nmse(sswm_est, H_test_clean)
        results[snr] = (ls, mm, sw)
        win = min([("LS", ls), ("MMSE", mm), ("SSWM", sw)], key=lambda x: x[1])[0]
        print(f"{snr:5.0f} | {ls:8.4f} {mm:8.4f} {sw:8.4f} | {win}", flush=True)

    print("\nNMSE (lower is better). SSWM = world-model channel head (probe on frozen latent).", flush=True)


if __name__ == "__main__":
    main()
