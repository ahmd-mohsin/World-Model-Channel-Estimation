"""Train the conditional diffusion channel estimator and evaluate NMSE vs SNR (full-grid), directly
comparable to the world model's eval_*.json (LS/MMSE/BEAM).

    torchrun --nproc_per_node=1 scripts/train-diffusion-estimator.py --data_dir data/mimo \
        --n_ant 32 --n_sub 32 --tag diff
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.task_heads.diffusion_estimator import DiffusionEstimator
from implementation.task_heads.baselines import add_noise, mmse_estimate
from implementation.wireless_data import ShardDataset

PRED = Path("results/diffusion")


def nmse(p, t): return (F.mse_loss(p, t) / t.pow(2).mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--tag", default="diff")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sample_steps", type=int, default=20)
    ap.add_argument("--n_ant", type=int, default=8); ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args(); torch.manual_seed(args.seed)

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    dev = f"cuda:{local}"; main_rank = rank == 0
    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, horizon_k=3,
                     action_dim=4, embed_dim=128, state_dim=64, latent_dim=128, use_pretrained=False)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=args.seed)
    m = DiffusionEstimator(cfg).to(dev)
    if main_rank: print(f"diffusion params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    ddp = DDP(m, device_ids=[local])
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank + 100 * args.seed)
    ng = torch.Generator(device=dev).manual_seed(rank + 100 * args.seed)
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        tot, met = ddp(o, a, noise_gen=ng)
        opt.zero_grad(); tot.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if main_rank and (step % 2000 == 0 or step == args.steps - 1):
            print(f"step {step:5d} | eps_loss {met['pred']:.4f}", flush=True)

    dist.barrier()
    if main_rank:
        m.eval()
        o, _ = ds.all("test", device=dev); Hte = o[:1000, -1]
        Htr = ds.all("train", device=dev)[0][:3000, -1]
        g = torch.Generator(device=dev).manual_seed(0)
        sweep = {}
        print("\nSNR      LS     MMSE    DIFF", flush=True)
        for snr in [-5, 0, 5, 10, 15, 20, 30]:
            Y = add_noise(Hte, snr, generator=g)
            Hd = m.estimate(Hte, snr, steps=args.sample_steps, noise_gen=g)
            ls = nmse(Y, Hte); mm = nmse(mmse_estimate(Y, Htr, snr), Hte); df = nmse(Hd, Hte)
            sweep[snr] = {"ls": ls, "mmse": mm, "diff": df}
            print(f"{snr:3d}  {ls:7.4f} {mm:7.4f} {df:7.4f}", flush=True)
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"eval_{args.tag}.json").write_text(json.dumps(sweep))
        print(f"wrote results/diffusion/eval_{args.tag}.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
