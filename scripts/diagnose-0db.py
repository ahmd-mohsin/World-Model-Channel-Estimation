"""Diagnose WHY 0 dB channel estimation is hard and what NMSE ceiling is achievable.

Works on the raw channel-estimation sub-problem (no world model), on real Sionna data, so we
separate causes cleanly:
  A. LS (= noisy obs)                         -> how much noise there is
  B. MMSE, TRUE covariance (oracle)           -> theoretical linear ceiling
  C. MMSE, per-scene covariance               -> realistic MMSE
  D. Global-mean channel (rank-0 prior)       -> how predictable the channel is a priori
  E. Learned MLP head trained TO CONVERGENCE  -> is our head undertrained or hitting a wall?
  F. Learned conv U-Net trained TO CONVERGENCE
Run on box:  python scripts/diagnose-0db.py --data_dir data/act60k
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.task_heads.baselines import add_noise, mmse_estimate, nmse
from implementation.task_heads.unet_head import UNetChannelHead

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(data_dir):
    xs, sc = [], []
    for f in sorted(glob.glob(f"{data_dir}/shard_*.pt")):
        d = torch.load(f, map_location="cpu")
        xs.append(d["data"][:, -1])                 # last frame (B,2,ant,sub)
        sc += [d.get("scene", "?")] * d["data"].shape[0]
    return torch.cat(xs, 0), sc


def train_head(net, Ytr, Htr, nvtr, Yte, steps, use_grid, latent_dim, cfg_shape):
    net = net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    B = Htr.shape[0]; tgt = Htr.reshape(B, -1)
    for s in range(steps):
        idx = torch.from_numpy(rng.choice(B, 512, replace=False)).to(dev)
        if use_grid:
            est = net(Ytr[idx], torch.zeros(idx.numel(), latent_dim, device=dev), nvtr[idx])
            est = est.reshape(idx.numel(), -1)
        else:
            yf = Ytr[idx].reshape(idx.numel(), -1)
            est = yf + net(yf)
        loss = F.mse_loss(est, tgt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if use_grid:
            e = net(Yte, torch.zeros(Yte.shape[0], latent_dim, device=dev),
                    (Hte_pw / (10 ** (SNR / 10)))).reshape(Yte.shape[0], -1)
        else:
            yf = Yte.reshape(Yte.shape[0], -1); e = yf + net(yf)
    return e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=40000); args = ap.parse_args()
    global SNR, Hte_pw
    SNR = 0.0

    H, scenes = load(args.data_dir)
    H = H.to(dev)
    n = H.shape[0]; ntr = int(n * 0.9)
    Htr, Hte = H[:ntr], H[ntr:]
    g = torch.Generator(device=dev).manual_seed(0)
    Ytr = add_noise(Htr, SNR, generator=g); Yte = add_noise(Hte, SNR, generator=g)
    Hte_pw = Hte.pow(2).mean(dim=(1, 2, 3))
    nvtr = (Htr.pow(2).mean(dim=(1, 2, 3)) / (10 ** (SNR / 10)))
    n_ant, n_sub = H.shape[2], H.shape[3]
    print(f"n={n} ant={n_ant} sub={n_sub} @ {SNR} dB\n")

    print("=== reference points (NMSE @ 0 dB, lower better) ===")
    print(f"A. LS (noisy obs)            : {nmse(Yte, Hte):.4f}")
    print(f"B. MMSE true-cov (ORACLE)    : {nmse(mmse_estimate(Yte, Hte, SNR), Hte):.4f}")
    print(f"C. MMSE train-cov (realistic): {nmse(mmse_estimate(Yte, Htr, SNR), Hte):.4f}")
    gm = Htr.mean(0, keepdim=True).expand_as(Hte)
    print(f"D. global-mean channel       : {nmse(gm, Hte):.4f}")

    # E. MLP head to convergence
    mlp = nn.Sequential(nn.Linear(2*n_ant*n_sub, 1024), nn.GELU(),
                        nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 2*n_ant*n_sub))
    nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
    e = train_head(mlp, Ytr, Htr, nvtr, Yte, args.steps, False, 0, None)
    print(f"E. MLP head ({args.steps} steps)   : {nmse(e, Hte.reshape(Hte.shape[0],-1)):.4f}")

    # F. U-Net head to convergence
    un = UNetChannelHead(8, n_ant, n_sub, base=48)
    e = train_head(un, Ytr, Htr, nvtr, Yte, args.steps, True, 8, None)
    print(f"F. U-Net head ({args.steps} steps) : {nmse(e, Hte.reshape(Hte.shape[0],-1)):.4f}")


if __name__ == "__main__":
    main()
