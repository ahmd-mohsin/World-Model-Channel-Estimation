from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

dev = "cuda" if torch.cuda.is_available() else "cpu"


class RawPredictor(nn.Module):
    """Predict o_{t+k} from a short channel HISTORY + velocity action, as a RESIDUAL on o_t.

    Works directly in channel space (flattened real/imag), zero-init output so it starts at
    persistence and must learn the motion-driven correction.
    """

    def __init__(self, chan_dim, action_dim, hist=3, hidden=1024):
        super().__init__()
        self.hist = hist
        self.enc = nn.Sequential(
            nn.Linear(chan_dim * hist + action_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
        )
        self.out = nn.Linear(hidden, chan_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, hist_frames, action, present):
        x = torch.cat([hist_frames.flatten(1), action], dim=-1)
        return present + self.out(self.enc(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data_dir}/shard_*.pt"))
    datas = [torch.load(f, map_location="cpu") for f in files]
    data = torch.cat([d["data"] for d in datas], 0)        # (N,T,2,ant,sub)
    action = torch.cat([d["action"] for d in datas], 0)    # (N,T,act)
    N, T = data.shape[:2]
    chan_dim = data[0, 0].numel()
    hist, k = 3, 3
    anchor = T - 1 - k
    print(f"N={N} T={T} chan_dim={chan_dim} scenes={[d.get('scene') for d in datas]}", flush=True)

    # per-channel standardization (over train) so MSE isn't dominated by big-scale scenes
    flat = data.reshape(N, T, chan_dim)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_test = max(256, N // 10)
    te, tr = perm[:n_test], perm[n_test:]
    mu = flat[tr].mean((0, 1)); sd = flat[tr].std((0, 1)) + 1e-6
    flat = (flat - mu) / sd
    act = (action - action.mean((0, 1))) / (action.std((0, 1)) + 1e-6)
    flat, act = flat.to(dev), act.to(dev)
    tr, te = tr.to(dev), te.to(dev)

    model = RawPredictor(chan_dim, act.shape[-1], hist=hist).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(0)

    def batch(idx):
        h = flat[idx][:, anchor - hist + 1:anchor + 1]       # (B,hist,chan)
        pres = flat[idx][:, anchor]
        fut = flat[idx][:, anchor + k]
        a = act[idx][:, anchor]
        return h, a, pres, fut

    t0 = time.time()
    for step in range(args.steps):
        idx = tr[torch.from_numpy(rng.choice(len(tr), args.bs, replace=False)).to(dev)]
        h, a, pres, fut = batch(idx)
        pred = model(h, a, pres)
        loss = F.mse_loss(pred, fut)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                pers = F.mse_loss(pres, fut).item()
            print(f"step {step:5d} | train_mse {loss.item():.4f} | persistence_mse {pers:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    model.eval()
    with torch.no_grad():
        h, a, pres, fut = batch(te)
        pred = model(h, a, pres)
        def nmse(p): return (F.mse_loss(p, fut) / fut.pow(2).mean()).item()
        print("\n==== HELD-OUT (raw channel space, standardized) ====", flush=True)
        print(f"predictor    NMSE: {nmse(pred):.4f}")
        print(f"persistence  NMSE: {nmse(pres):.4f}")
        print(f"linear-extrap NMSE: {nmse(pres + (flat[te][:,anchor]-flat[te][:,anchor-1])):.4f}")
        gain = nmse(pres) / max(nmse(pred), 1e-9)
        print(f"\npredictor beats persistence by {gain:.2f}x  "
              f"({'WIN' if gain>1.02 else 'LOSS'})", flush=True)


if __name__ == "__main__":
    main()
