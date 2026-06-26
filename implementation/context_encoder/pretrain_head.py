from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.wireless_data import WirelessDataset, SionnaSpec


def vicreg_loss(za, zb, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0):
    inv = F.mse_loss(za, zb)

    def var_cov(z):
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        var_term = torch.relu(1.0 - std).mean()
        n, d = z.shape
        cov = (z.T @ z) / (n - 1)
        cov_term = (cov.fill_diagonal_(0.0) ** 2).sum() / d
        return var_term, cov_term

    va, ca = var_cov(za)
    vb, cb = var_cov(zb)
    loss = sim_coeff * inv + var_coeff * 0.5 * (va + vb) + cov_coeff * 0.5 * (ca + cb)
    return loss, {"inv": inv.item(), "var": 0.5 * (va + vb).item(), "cov": 0.5 * (ca + cb).item()}


def add_awgn(o, snr_db_low=5.0, snr_db_high=25.0):
    b = o.shape[0]
    snr = torch.empty(b, device=o.device).uniform_(snr_db_low, snr_db_high)
    power = o.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True)
    noise_p = power / (10 ** (snr.view(-1, 1, 1, 1, 1) / 10))
    return o + torch.randn_like(o) * noise_p.sqrt()


def load_pool(data_dir, dev):
    files = sorted(Path(data_dir).glob("shard_*.pt"))
    if not files:
        return None, None
    datas, poss = [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        datas.append(d["data"])
        poss.append(d["pos"])
    data = torch.cat(datas, 0).to(dev)
    pos = torch.cat(poss, 0).to(dev)
    print(f"loaded pool from {len(files)} shards: {tuple(data.shape)}")
    return data, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=2,
                     embed_dim=256, backbone="lwm", use_pretrained=True)

    pool, _ = (load_pool(args.data_dir, dev) if args.data_dir else (None, None))
    if pool is None:
        print("no shards found; generating a small pool inline...")
        ds = WirelessDataset(cfg, n_samples=192, spec=SionnaSpec(scene="munich"), seed=0)
        pool = ds.batch(192).to(dev)
    n_seq = pool.shape[0]

    enc = ContextEncoder(cfg).to(dev).train()
    print(f"training head only: {sum(p.numel() for p in enc.trainable_parameters()):,} params (LWM frozen)")
    opt = torch.optim.AdamW(enc.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(0)

    history = []
    for step in range(args.steps):
        idx = rng.choice(n_seq, size=min(args.bs, n_seq), replace=False)
        o = pool[idx]
        # Positive pair = same channel, two independent noise augmentations.
        # Invariance to noise == the denoising prior behind channel estimation.
        oa, ob = add_awgn(o), add_awgn(o)
        za = enc(oa).reshape(-1, cfg.embed_dim)
        zb = enc(ob).reshape(-1, cfg.embed_dim)
        loss, parts = vicreg_loss(za, zb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append({"step": step, "loss": loss.item(), **parts})
        if step % 250 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | loss {loss.item():7.3f} | inv {parts['inv']:.4f} "
                  f"var {parts['var']:.4f} cov {parts['cov']:.4f}", flush=True)

    out_dir = Path(__file__).resolve().parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "head_vicreg.pt"
    torch.save({"head": enc.head.state_dict(), "config": cfg.__dict__, "history": history}, ckpt)
    print(f"\nsaved head checkpoint -> {ckpt}")
    _plot(history, out_dir / "06_pretrain_loss.png")


def _plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(steps, [h["loss"] for h in history])
    ax[0].set_title("VICReg total loss"); ax[0].set_xlabel("step"); ax[0].grid(alpha=0.3)
    for k in ("inv", "var", "cov"):
        ax[1].plot(steps, [h[k] for h in history], label=k)
    ax[1].set_title("VICReg components"); ax[1].set_xlabel("step"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("ContextEncoder head SSL pretraining (VICReg, noise-invariance on Sionna channels)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved loss curve -> {path}")


if __name__ == "__main__":
    main()
