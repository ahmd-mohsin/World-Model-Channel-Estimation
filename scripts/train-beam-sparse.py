"""Train + eval the beamspace world model under SPARSE (comb) pilots.

Sparse pilots are where a temporal world-model prior should structurally win: unobserved subcarriers
have NO observation, so the estimate must come from the prior. We compare, at a fixed pilot stride:
  - full        : world model with prior (mask-aware fusion)
  - noprior     : same, prior removed (--ablate prior) — must interpolate from pilots alone
  - interp      : classical linear pilot interpolation
  - masked-MMSE : Wiener interpolation from pilots using train frequency covariance

    torchrun --nproc_per_node=1 scripts/train-beam-sparse.py --data_dir data/stress6 \
        --stride 4 --tag sp_full [--ablate prior] [--holdout_scene S]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel
from implementation.wireless_data import ShardDataset
from implementation.task_heads.baselines import (add_noise, comb_mask, pilot_interp, masked_mmse, nmse)

OUT = Path("implementation/checkpoints"); SPARSE = Path("results/sparse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--holdout_scene", default=None)
    ap.add_argument("--stride", type=int, default=4)          # 1-in-stride comb pilots
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="sp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="full", choices=list(BeamWorldModel.ABLATIONS))
    ap.add_argument("--deep_fusion", action="store_true",
                    help="use ReEsNet-capacity fusion head fed the WM prior")
    ap.add_argument("--finetune_steps", type=int, default=0,
                    help="after joint training, freeze all but the fusion head and train it on "
                         "estimation-only loss (kills the multitask dilution of the estimator)")
    ap.add_argument("--n_ant", type=int, default=8)
    ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    main_rank = rank == 0

    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, horizon_k=3,
                     action_dim=4, embed_dim=128, state_dim=64, latent_dim=128,
                     use_pretrained=False, unet_base_ch=48)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=args.seed, holdout_scene=args.holdout_scene)
    if main_rank:
        print(f"world={dist.get_world_size()} | train {len(ds.train_idx)} test {len(ds.test_idx)} "
              f"| stride={args.stride} ablate={args.ablate} holdout={args.holdout_scene}", flush=True)
    m = BeamWorldModel(cfg, ablate=args.ablate, sparse_stride=args.stride,
                       deep_fusion=args.deep_fusion).to(dev)
    if main_rank:
        print(f"params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank + 100 * args.seed)
    ng = torch.Generator(device=dev).manual_seed(rank + 100 * args.seed)

    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        total, met = ddp(o, a, noise_gen=ng)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if main_rank and (step % 2000 == 0 or step == args.steps - 1):
            print(f"step {step:5d} | total {met['total']:.4f} | chan_nmse {met['chan_nmse']:.4f}", flush=True)

    # ---- estimation-only fine-tune: freeze encoder/SSM/predictor, train only the fusion head ----
    if args.finetune_steps > 0:
        dist.barrier()
        n_tr, n_fr = m.freeze_for_finetune()
        if main_rank:
            print(f"[finetune] head params {n_tr:,} trainable, {n_fr:,} frozen (predictor intact)",
                  flush=True)
        # fresh DDP + optimizer over the head params only
        ft = DDP(m, device_ids=[local], find_unused_parameters=True)
        ft_lr = args.lr * 0.5
        ft_opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                   lr=ft_lr, weight_decay=1e-4)
        ft_sched = torch.optim.lr_scheduler.OneCycleLR(ft_opt, max_lr=ft_lr,
                                                       total_steps=args.finetune_steps, pct_start=0.1)
        for step in range(args.finetune_steps):
            o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
            total, met = ft(o, a, noise_gen=ng, chan_only=True)
            ft_opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
            ft_opt.step(); ft_sched.step()
            if main_rank and (step % 2000 == 0 or step == args.finetune_steps - 1):
                print(f"[ft] step {step:5d} | chan_nmse {met['chan_nmse']:.4f}", flush=True)

    dist.barrier()
    if main_rank:
        m.eval()
        o, a = ds.all("test", device=dev); o, a = o[:2000], a[:2000]
        Htr = ds.all("train", device=dev)[0][:3000, -1]
        Hte = o[:, -1]
        g = torch.Generator(device=dev).manual_seed(0)
        mask = comb_mask(cfg.n_subcarriers, args.stride, dev)
        sweep = {}
        print(f"\nSNR   interp   mMMSE    MODEL   (stride={args.stride})", flush=True)
        for snr in [-5, 0, 5, 10, 15, 20, 30]:
            Hhat, _, Ymask, _ = m.estimate_channel_sparse(o, a, snr_db=float(snr), stride=args.stride, noise_gen=g)
            # baselines share the SAME masked noisy obs
            Yfull = add_noise(Hte, snr, generator=g); Ymask_b = Yfull * mask
            itp = pilot_interp(Ymask_b, mask); mm = masked_mmse(Ymask_b, mask, Htr, snr)
            r = {"interp": nmse(itp, Hte), "mmse": nmse(mm, Hte), "model": nmse(Hhat, Hte)}
            sweep[snr] = r
            print(f"{snr:3d}  {r['interp']:7.4f} {r['mmse']:7.4f} {r['model']:7.4f}", flush=True)
        SPARSE.mkdir(parents=True, exist_ok=True)
        (SPARSE / f"eval_{args.tag}.json").write_text(json.dumps(sweep))
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__, "stride": args.stride},
                   OUT / f"beam_{args.tag}.pt")
        print(f"saved -> {OUT/f'beam_{args.tag}.pt'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
