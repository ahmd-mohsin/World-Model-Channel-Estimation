"""End-to-end DDP training of the full SSWM with per-module + total losses, JSON-logged.

torchrun --nproc_per_node=8 scripts/train-e2e.py --data_dir data/act100k --steps 30000

Trains the whole network jointly: world-model JEPA + VICReg anti-collapse + channel-estimation
task. LoRA-unfrozen LWM. Rank 0 logs every component loss to metrics.json (for the dashboard),
runs held-out eval, and a channel-estimation NMSE-vs-LS/MMSE sweep across SNRs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.task_heads import add_noise, ls_estimate, mmse_estimate, nmse
from implementation.wireless_data import ShardDataset

OUT = Path("implementation/checkpoints")
DASH = Path("dashboard")


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--full_finetune", action="store_true",
                    help="fully unfreeze LWM (train all backbone weights, smaller LR) vs LoRA-only")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                     embed_dim=256, action_dim=4, state_dim=64, latent_dim=256,
                     backbone="lwm", use_pretrained=True, residual_prediction=True,
                     freeze_backbone=not args.full_finetune,
                     lora=not args.full_finetune, lora_rank=8, lora_alpha=16)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=0)
    mode = "FULL FINE-TUNE (LWM unfrozen)" if args.full_finetune else "LoRA (LWM frozen)"
    log(f"world={world} | train {len(ds.train_idx)} | test {len(ds.test_idx)} | {mode} | scenes {ds.scenes}")

    m = SSWM(cfg).to(dev)
    log(f"trainable params/GPU: {sum(p.numel() for p in m.all_trainable_parameters()):,}")
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)
    params = list(m.all_trainable_parameters())
    groups = m.param_groups(args.lr, args.backbone_lr_mult) if args.full_finetune else params
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    max_lrs = [g["lr"] for g in opt.param_groups]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lrs, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank)
    ng = torch.Generator(device=dev).manual_seed(rank)

    if is_main():
        DASH.mkdir(exist_ok=True)
        OUT.mkdir(parents=True, exist_ok=True)
    history = []
    m.context_encoder.train()
    t0 = time.time()
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        # Route through the DDP wrapper (loss=True) so gradient all-reduce hooks fire.
        total, met = ddp(o, a, loss=True, snr_db=args.snr, noise_gen=ng)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        m.update_target()
        if step % 200 == 0 or step == args.steps - 1:
            t = {k: torch.tensor(v, device=dev) for k, v in met.items()}
            for v in t.values():
                dist.all_reduce(v, op=dist.ReduceOp.AVG)
            met = {k: v.item() for k, v in t.items()}
            if is_main():
                met["step"] = step
                met["elapsed"] = time.time() - t0
                met["lr"] = sched.get_last_lr()[0]
                history.append(met)
                (DASH / "metrics.json").write_text(json.dumps(history))
                if step % 1000 == 0 or step == args.steps - 1:
                    log(f"step {step:5d} | total {met['total']:.4f} | jepa {met['jepa']:.4f} "
                        f"| vic {met['vic']:.4f} | chan {met['chan']:.4f} "
                        f"| chan_nmse {met['chan_nmse']:.4f} | {met['elapsed']:.0f}s")

    dist.barrier()
    if is_main():
        m.eval()
        evaluate(m, ds, cfg, dev)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__, "history": history},
                   OUT / "sswm_e2e.pt")
        log(f"saved -> {OUT/'sswm_e2e.pt'}")
    dist.destroy_process_group()


@torch.no_grad()
def evaluate(m, ds, cfg, dev):
    o, a = ds.all("test", device=dev)
    o = o[:2000]; a = a[:2000]
    t = cfg.seq_len - 1 - cfg.horizon_k
    # world-model predictor vs persistence
    zh, zt, zp = [], [], []
    for i in range(0, o.shape[0], 256):
        ob, ab = o[i:i+256], a[i:i+256]
        h, tt = m(ob, ab)
        zh.append(h); zt.append(tt)
        zp.append(m.target_encoder(ob[:, t].unsqueeze(1))[:, 0])
    zh, zt, zp = torch.cat(zh), torch.cat(zt), torch.cat(zp)
    pred_nmse = (F.mse_loss(zh, zt) / zt.pow(2).mean()).item()
    pers_nmse = (F.mse_loss(zp, zt) / zt.pow(2).mean()).item()
    log(f"\n[world model] predictor NMSE {pred_nmse:.4f} vs persistence {pers_nmse:.4f} "
        f"({pers_nmse/max(pred_nmse,1e-9):.2f}x)")

    # channel estimation vs LS/MMSE across SNRs
    tch = cfg.seq_len - 1
    H_tr = ds.all("train", device=dev)[0][:, tch]
    H_te = o[:, tch]
    g = torch.Generator(device=dev).manual_seed(0)
    sweep = {}
    log("\n[channel est] NMSE vs SNR:")
    log(f"{'SNR':>5} {'LS':>8} {'MMSE':>8} {'SSWM':>8}")
    for snr in [0, 5, 10, 15, 20]:
        Yte = add_noise(H_te, snr, generator=g)
        seq = o.clone(); seq[:, tch] = Yte
        zobs = []
        for i in range(0, seq.shape[0], 256):
            zobs.append(m.encode_sequence(seq[i:i+256], a[i:i+256])[:, tch])
        zobs = torch.cat(zobs)
        est = m.task_heads(zobs, Yte)["channel"].reshape(H_te.shape)
        ls = nmse(ls_estimate(Yte), H_te)
        mm = nmse(mmse_estimate(Yte, H_tr, snr), H_te)
        sw = nmse(est, H_te)
        sweep[snr] = {"ls": ls, "mmse": mm, "sswm": sw}
        log(f"{snr:5.0f} {ls:8.4f} {mm:8.4f} {sw:8.4f}")
    (DASH / "eval.json").write_text(json.dumps(
        {"pred_nmse": pred_nmse, "pers_nmse": pers_nmse, "channel_sweep": sweep}))


if __name__ == "__main__":
    main()
