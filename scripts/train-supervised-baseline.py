"""Same-capacity SUPERVISED U-Net baseline: no world model, no JEPA, no SSM.

Trains the identical UNetChannelHead directly on (noisy y, clean H) pairs to isolate what the
world-model apparatus actually buys for channel estimation. Uses the SAME conv U-Net, SAME SNR
randomization, SAME data/split as SSWM; the only difference is there is no latent z (a zero vector
is fed to the FiLM conditioning), no encoder, no predictor.

    python scripts/train-supervised-baseline.py --data_dir data/act60k [--holdout_scene S]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.task_heads.unet_head import UNetChannelHead
from implementation.task_heads.baselines import add_noise, ls_estimate, mmse_estimate, nmse
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--holdout_scene", default=None)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="supervised")
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, action_dim=4,
                     embed_dim=256, latent_dim=256, use_pretrained=False, channel_head="unet")
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=0, holdout_scene=args.holdout_scene)
    t = cfg.seq_len - 1
    Htr = ds.all("train", device=dev)[0][:, t]     # standardized clean channels (train)
    Hte = ds.all("test", device=dev)[0][:, t]
    print(f"train {Htr.shape[0]} | test {Hte.shape[0]} | holdout={args.holdout_scene}", flush=True)

    # SAME conv U-Net as SSWM's channel head; latent dim d fed as zeros (no world model).
    net = UNetChannelHead(cfg.latent_dim, cfg.n_antennas, cfg.n_subcarriers,
                          base=cfg.unet_base_ch).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"supervised U-Net params: {npar:,}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(0); g = torch.Generator(device=dev).manual_seed(0)
    zdim = cfg.latent_dim

    B = Htr.shape[0]
    for step in range(args.steps):
        idx = torch.from_numpy(rng.choice(B, args.bs, replace=False)).to(dev)
        H = Htr[idx]
        snr = torch.empty(H.shape[0], 1, 1, 1, device=dev).uniform_(0, 20)   # per-sample SNR
        nv = H.pow(2).mean(dim=(1, 2, 3), keepdim=True) / (10 ** (snr / 10))
        Y = H + torch.randn(H.shape, generator=g, device=dev) * nv.sqrt()
        z0 = torch.zeros(H.shape[0], zdim, device=dev)                       # NO world model
        est = net(Y, z0, nv.reshape(-1))
        loss = F.mse_loss(est, H)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"step {step:5d} | mse {loss.item():.4f}", flush=True)

    # eval vs LS / MMSE across SNR
    net.eval(); sweep = {}
    print("\nSNR      LS     MMSE    SUP", flush=True)
    with torch.no_grad():
        for snr in [0, 5, 10, 15, 20]:
            Yte = add_noise(Hte, snr, generator=g)
            nv = Hte.pow(2).mean(dim=(1, 2, 3)) / (10 ** (snr / 10))
            z0 = torch.zeros(Hte.shape[0], zdim, device=dev)
            est = net(Yte, z0, nv)
            ls = nmse(ls_estimate(Yte), Hte)
            mm = nmse(mmse_estimate(Yte, Htr, snr), Hte)
            sp = nmse(est, Hte)
            sweep[snr] = {"ls": ls, "mmse": mm, "sup": sp}
            print(f"{snr:3d}  {ls:7.4f} {mm:7.4f} {sp:7.4f}", flush=True)
    Path("dashboard").mkdir(exist_ok=True)
    Path(f"dashboard/eval_{args.tag}.json").write_text(json.dumps(sweep))
    print(f"saved dashboard/eval_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
