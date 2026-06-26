from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_actions(o, action_dim):
    b, t = o.shape[:2]
    power = o.pow(2).mean(dim=(2, 3, 4))
    dp = torch.zeros_like(power); dp[:, 1:] = power[:, 1:] - power[:, :-1]
    base = torch.stack([power, dp], dim=-1)
    pad = torch.zeros(b, t, action_dim - 2, device=o.device)
    return torch.cat([base, pad], dim=-1)


def vicreg_var_cov(z, var_coeff=10.0, cov_coeff=1.0):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var = torch.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    cov = (cov.fill_diagonal_(0.0) ** 2).sum() / d
    return var_coeff * var + cov_coeff * cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data_dir}/shard_*.pt"))
    datas = [torch.load(f, map_location="cpu") for f in files]
    pool = torch.cat([d["data"] for d in datas], 0)
    scenes = [d.get("scene", "?") for d in datas]
    print(f"pool {tuple(pool.shape)} from {len(files)} shards, scenes={scenes}", flush=True)

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(pool.shape[0], generator=g)
    n_test = max(128, pool.shape[0] // 10)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train, test = pool[train_idx].to(dev), pool[test_idx].to(dev)
    print(f"train {train.shape[0]} | test {test.shape[0]}", flush=True)

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                     embed_dim=256, action_dim=8, state_dim=64, latent_dim=256,
                     backbone="lwm", use_pretrained=True)
    m = SSWM(cfg).to(dev)
    m.context_encoder.train()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(0)

    t0 = time.time()
    for step in range(args.steps):
        idx = rng.choice(train.shape[0], size=args.bs, replace=False)
        o = train[idx]; a = make_actions(o, cfg.action_dim)
        z_hat, z_tilde = m(o, a)
        # Residual prediction makes persistence the prior; the target's natural multi-scene
        # scale is real signal (not collapse), so we do NOT clamp it with VICReg here.
        loss = F.smooth_l1_loss(z_hat, z_tilde)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        m.update_target()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f} "
                  f"| pred_std {z_hat.std(0).mean().item():.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)

    # ---- held-out evaluation: predictor vs trivial baselines ----
    m.eval()
    with torch.no_grad():
        o = test; a = make_actions(o, cfg.action_dim)
        anchor = cfg.seq_len - 1 - cfg.horizon_k
        z = m.encode_sequence(o, a)
        z_hat = m.predictor(z[:, anchor], a[:, anchor:anchor + cfg.horizon_k])
        z_tilde = m.target_encoder(o[:, anchor + cfg.horizon_k].unsqueeze(1))[:, 0]
        z_present = m.target_encoder(o[:, anchor].unsqueeze(1))[:, 0]
        z_mean = z_tilde.mean(0, keepdim=True).expand_as(z_tilde)

        def nmse(p): return (F.mse_loss(p, z_tilde) / z_tilde.pow(2).mean()).item()
        print("\n==== HELD-OUT TEST (unseen sequences) ====", flush=True)
        print(f"predictor   NMSE: {nmse(z_hat):.4f}")
        print(f"persistence NMSE: {nmse(z_present):.4f}")
        print(f"batch-mean  NMSE: {nmse(z_mean):.4f}")
        print(f"pred_std {z_hat.std(0).mean().item():.4f} | target_std {z_tilde.std(0).mean().item():.4f}")
        gain = nmse(z_present) / max(nmse(z_hat), 1e-9)
        print(f"\npredictor beats persistence by {gain:.2f}x")

    out = Path("implementation/checkpoints"); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "config": cfg.__dict__}, out / "sswm_large.pt")
    print(f"saved -> {out/'sswm_large.pt'}", flush=True)


if __name__ == "__main__":
    main()
