"""Train the learned deep baseline (ReEsNet) on sparse-pilot channel estimation.

Apples-to-apples competitor to Beam-WM under sparse pilots: a single-snapshot residual CNN that sees
only the masked noisy observation (+ mask + noise level), NO temporal history / world model / oracle
covariance. Same data, SNR randomization, comb stride, and test split as train-beam-sparse.py, so
results/sparse/eval_deep_s{stride}.json is directly comparable to eval_sp_s{stride}_full.json.

    torchrun --nproc_per_node=1 scripts/train-deep-baseline.py --data_dir data/stress6 \
        --stride 4 --tag deep_s4 [--holdout_scene S]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.wireless_data import ShardDataset
from implementation.task_heads import (ReEsNet, add_noise, comb_mask, pilot_interp, masked_mmse, nmse)

OUT = Path("implementation/checkpoints"); SPARSE = Path("results/sparse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--holdout_scene", default=None)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="deep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ant", type=int, default=8)
    ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    main_rank = rank == 0

    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, use_pretrained=False)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=args.seed, holdout_scene=args.holdout_scene)
    t = cfg.seq_len - 1
    Htr = ds.all("train", device=dev)[0][:, t]         # standardized clean channels (current frame)
    Hte = ds.all("test", device=dev)[0][:, t]
    if main_rank:
        print(f"train {Htr.shape[0]} test {Hte.shape[0]} | stride={args.stride} holdout={args.holdout_scene}",
              flush=True)

    net = ReEsNet(cfg.n_antennas, cfg.n_subcarriers, base=64, n_blocks=8).to(dev)
    if main_rank:
        print(f"ReEsNet params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    ddp = DDP(net, device_ids=[local])
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank + 100 * args.seed)
    g = torch.Generator(device=dev).manual_seed(rank + 100 * args.seed)
    mask = comb_mask(cfg.n_subcarriers, args.stride, dev)
    B = Htr.shape[0]

    for step in range(args.steps):
        idx = torch.from_numpy(rng.choice(B, args.bs, replace=False)).to(dev)
        H = Htr[idx]
        snr = torch.empty(H.shape[0], 1, 1, 1, device=dev).uniform_(-5, 30)     # match sparse range
        nv = H.pow(2).mean(dim=(1, 2, 3), keepdim=True) / (10 ** (snr / 10))
        Y = (H + torch.randn(H.shape, generator=g, device=dev) * nv.sqrt()) * mask
        est = ddp(Y, mask, nv.reshape(-1))
        loss = F.mse_loss(est, H)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if main_rank and (step % 2000 == 0 or step == args.steps - 1):
            print(f"step {step:5d} | mse {loss.item():.4f}", flush=True)

    dist.barrier()
    if main_rank:
        net.eval()
        Htr_cov = Htr[:3000]
        gg = torch.Generator(device=dev).manual_seed(0)
        sweep = {}
        print(f"\nSNR   interp   mMMSE     deep   (stride={args.stride})", flush=True)
        with torch.no_grad():
            for snr in [-5, 0, 5, 10, 15, 20, 30]:
                Yfull = add_noise(Hte, snr, generator=gg); Ymask = Yfull * mask
                nv = Hte.pow(2).mean(dim=(1, 2, 3)) / (10 ** (snr / 10))
                deep = net(Ymask, mask, nv)
                itp = pilot_interp(Ymask, mask); mm = masked_mmse(Ymask, mask, Htr_cov, snr)
                r = {"interp": nmse(itp, Hte), "mmse": nmse(mm, Hte), "deep": nmse(deep, Hte)}
                sweep[snr] = r
                print(f"{snr:3d}  {r['interp']:7.4f} {r['mmse']:7.4f} {r['deep']:7.4f}", flush=True)
        SPARSE.mkdir(parents=True, exist_ok=True)
        (SPARSE / f"eval_{args.tag}.json").write_text(json.dumps(sweep))
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "stride": args.stride}, OUT / f"{args.tag}.pt")
        print(f"saved -> results/sparse/eval_{args.tag}.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
