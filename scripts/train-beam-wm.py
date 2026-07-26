"""Train the beamspace world model (DDP, 8 GPUs). Predict future beam channel + estimate current.

torchrun --nproc_per_node=8 scripts/train-beam-wm.py --data_dir data/act60k [--holdout_scene S]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel
from implementation.wireless_data import ShardDataset
from implementation.wireless_data.beamspace import to_beamspace, from_beamspace
from implementation.task_heads.baselines import add_noise, mmse_estimate, nmse

OUT = Path("implementation/checkpoints"); DASH = Path("dashboard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--holdout_scene", default=None)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="beam")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    main_rank = rank == 0

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, action_dim=4,
                     embed_dim=128, state_dim=64, latent_dim=128, use_pretrained=False, unet_base_ch=48)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=0, holdout_scene=args.holdout_scene)
    if main_rank:
        print(f"world={dist.get_world_size()} | train {len(ds.train_idx)} test {len(ds.test_idx)} "
              f"| holdout={args.holdout_scene} | scenes {ds.scenes}", flush=True)
    m = BeamWorldModel(cfg).to(dev)
    if main_rank:
        print(f"params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank); ng = torch.Generator(device=dev).manual_seed(rank)

    hist = []
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        total, met = ddp(o, a, noise_gen=ng)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if main_rank and (step % 1000 == 0 or step == args.steps - 1):
            met["step"] = step; hist.append(met)
            DASH.mkdir(exist_ok=True); (DASH / f"metrics_{args.tag}.json").write_text(json.dumps(hist))
            print(f"step {step:5d} | total {met['total']:.4f} | pred {met['pred']:.4f} "
                  f"| chan_nmse {met['chan_nmse']:.4f}", flush=True)

    dist.barrier()
    if main_rank:
        m.eval()
        o, a = ds.all("test", device=dev); o, a = o[:2000], a[:2000]
        Htr = ds.all("train", device=dev)[0][:, -1]
        Hte = o[:, -1]
        g = torch.Generator(device=dev).manual_seed(0)
        sweep = {}
        print("\nSNR      LS     MMSE    BEAM", flush=True)
        for snr in [0, 5, 10, 15, 20]:
            Hhat, _ = m.estimate_channel(o, a, snr_db=float(snr), noise_gen=g)
            Yte = add_noise(Hte, snr, generator=g)
            ls = nmse(Yte, Hte); mm = nmse(mmse_estimate(Yte, Htr, snr), Hte); bm = nmse(Hhat, Hte)
            sweep[snr] = {"ls": ls, "mmse": mm, "beam": bm}
            print(f"{snr:3d}  {ls:7.4f} {mm:7.4f} {bm:7.4f}", flush=True)
        (DASH / f"eval_{args.tag}.json").write_text(json.dumps(sweep))
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__}, OUT / f"beam_{args.tag}.pt")
        print(f"saved -> {OUT/f'beam_{args.tag}.pt'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
