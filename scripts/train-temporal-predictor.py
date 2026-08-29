"""Train a learned temporal predictor baseline (GRU/LSTM/Transformer) and evaluate prediction NMSE
vs horizon in CHANNEL space, so it is directly comparable to the world model's pred_*.json.

    torchrun --nproc_per_node=1 scripts/train-temporal-predictor.py --data_dir data/mimo \
        --n_ant 32 --n_sub 32 --backbone gru --tag tp_gru
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.task_heads.temporal_baseline import TemporalPredictor
from implementation.task_heads.ssm_predictor import DeepSSMPredictor
from implementation.wireless_data import ShardDataset
from implementation.wireless_data.beamspace import to_beamspace, from_beamspace

OUT = Path("implementation/checkpoints"); PRED = Path("results/temporal")


def nmse(p, t): return (F.mse_loss(p, t) / t.pow(2).mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone", default="gru", choices=["gru", "lstm", "transformer", "deepssm"])
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15000); ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tag", default="tp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ant", type=int, default=8); ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args(); torch.manual_seed(args.seed)

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
    dev = f"cuda:{local}"; main_rank = rank == 0
    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, horizon_k=3,
                     action_dim=4, embed_dim=128, state_dim=64, latent_dim=128,
                     use_pretrained=False, unet_base_ch=48)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=args.seed)
    if args.backbone == "deepssm":
        m = DeepSSMPredictor(cfg, n_layers=args.n_layers).to(dev)
    else:
        m = TemporalPredictor(cfg, backbone=args.backbone).to(dev)
    if main_rank: print(f"{args.backbone} params: {sum(p.numel() for p in m.parameters()):,}", flush=True)
    ddp = DDP(m, device_ids=[local])
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank + 100 * args.seed)
    ng = torch.Generator(device=dev).manual_seed(rank)
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        tot, met = ddp(o, a)
        opt.zero_grad(); tot.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if main_rank and (step % 2000 == 0 or step == args.steps - 1):
            print(f"step {step:5d} | pred {met['pred']:.4f}", flush=True)

    dist.barrier()
    if main_rank:
        m.eval()
        o, a = ds.all("test", device=dev); o, a = o[:2000], a[:2000]
        beam = to_beamspace(o); T = o.shape[1]
        pred = {}
        print("\n  k   persist    Temporal", flush=True)
        with torch.no_grad():
            for k in range(1, min(6, T - 1) + 1):
                anchor = T - 1 - k
                bp = m.predict(beam[:, :anchor + 1], a[:, :anchor + 1], a[:, anchor:anchor + k])
                Hp = from_beamspace(bp); Htrue = o[:, anchor + k]; Hpers = o[:, anchor]
                pred[k] = {"persist": nmse(Hpers, Htrue), "temporal": nmse(Hp, Htrue)}
                print(f"{k:3d}  {pred[k]['persist']:8.4f}  {pred[k]['temporal']:8.4f}", flush=True)
        PRED.mkdir(parents=True, exist_ok=True)
        (PRED / f"pred_{args.tag}.json").write_text(json.dumps(pred))
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__, "backbone": args.backbone},
                   OUT / f"{args.tag}.pt")
        print(f"saved -> results/temporal/pred_{args.tag}.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
