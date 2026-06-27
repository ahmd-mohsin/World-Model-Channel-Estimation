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
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                     embed_dim=256, action_dim=4, state_dim=64, latent_dim=256,
                     backbone="lwm", use_pretrained=True, residual_prediction=True)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.1, seed=0)
    print(f"train {len(ds.train_idx)} | test {len(ds.test_idx)} | scenes {ds.scenes}", flush=True)
    print(f"chan_sd {ds.chan_sd.flatten().tolist()} (standardized inputs)", flush=True)

    m = SSWM(cfg).to(dev)
    m.context_encoder.train()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(0)

    t0 = time.time()
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        z_hat, z_tilde = m(o, a)
        loss = F.smooth_l1_loss(z_hat, z_tilde)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.update_target()
        if step % 1000 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f} | pred_std {z_hat.std(0).mean().item():.3f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    m.eval()
    with torch.no_grad():
        o, a = ds.all("test", device=dev)
        anchor = cfg.seq_len - 1 - cfg.horizon_k
        zh, zt = m(o, a)
        z_present = m.target_encoder(o[:, anchor].unsqueeze(1))[:, 0]
        z_mean = zt.mean(0, keepdim=True).expand_as(zt)

        def nmse(p): return (F.mse_loss(p, zt) / zt.pow(2).mean()).item()
        print("\n==== HELD-OUT (embedding space, velocity actions + standardized channels) ====", flush=True)
        print(f"predictor    NMSE: {nmse(zh):.4f}")
        print(f"persistence  NMSE: {nmse(z_present):.4f}")
        print(f"batch-mean   NMSE: {nmse(z_mean):.4f}")
        gain = nmse(z_present) / max(nmse(zh), 1e-9)
        print(f"\npredictor vs persistence: {gain:.2f}x  ({'WIN' if gain>1.02 else 'LOSS'})", flush=True)

    out = Path("implementation/checkpoints"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "config": cfg.__dict__}, out / "sswm_scaled.pt")
    print(f"saved -> {out/'sswm_scaled.pt'}", flush=True)


if __name__ == "__main__":
    main()
