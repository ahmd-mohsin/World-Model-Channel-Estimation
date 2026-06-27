"""Large-scale DDP training of the full SSWM across all 8 A100s.

Launch:  torchrun --nproc_per_node=8 scripts/train-ddp.py --data_dir data/act60k --steps 40000

LoRA-unfreezes LWM (trainable low-rank adapters), velocity actions + per-channel standardized
inputs (ShardDataset). Rank 0 logs, evaluates on held-out, and saves.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.wireless_data import ShardDataset


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=128)       # per-GPU
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                     embed_dim=256, action_dim=4, state_dim=64, latent_dim=256,
                     backbone="lwm", use_pretrained=True, residual_prediction=True,
                     lora=True, lora_rank=8, lora_alpha=16)

    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=0)
    log(f"world={world} | train {len(ds.train_idx)} | test {len(ds.test_idx)} | scenes {ds.scenes}")

    m = SSWM(cfg).to(dev)
    n_train = sum(p.numel() for p in m.trainable_parameters())
    log(f"trainable params/GPU: {n_train:,} (incl. LoRA)")
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)

    trainable = [p for p in m.trainable_parameters()]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.05)

    import numpy as np
    rng = np.random.default_rng(1000 + rank)
    m.context_encoder.train()
    t0 = time.time()
    for step in range(args.steps):
        o, a = ds.batch(args.bs, "train", rng=rng, device=dev)
        z_hat, z_tilde = ddp(o, a)
        # Scale-robust target: predict the future embedding's direction + magnitude. The MSE alone
        # is near-zero in LWM's noise-invariant space (tiny residual) and gives no gradient, so we
        # add a cosine term that forces the prediction to match the future direction, not just sit
        # at the persistence prior.
        # MSE-dominant (so held-out NMSE -- the reported metric -- is what we optimize) with a
        # small cosine term for directional stability. Earlier a unit-weight cosine term made the
        # magnitude unconstrained and NMSE uninformative.
        mse = F.mse_loss(z_hat, z_tilde)
        cos = (1.0 - F.cosine_similarity(z_hat, z_tilde, dim=-1)).mean()
        loss = mse + 0.05 * cos
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        m.update_target()
        if step % 1000 == 0 or step == args.steps - 1:
            lt = torch.tensor(loss.item(), device=dev)
            dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            log(f"step {step:5d} | loss {lt.item():.4f} | pred_std {z_hat.std(0).mean().item():.3f} "
                f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")

    dist.barrier()
    if is_main():
        m.eval()
        with torch.no_grad():
            o, a = ds.all("test", device=dev)
            # chunk to avoid OOM on the full test set
            zh, zt, zp = [], [], []
            anchor = cfg.seq_len - 1 - cfg.horizon_k
            for i in range(0, o.shape[0], 256):
                ob, ab = o[i:i+256], a[i:i+256]
                h, t = m(ob, ab)
                zh.append(h); zt.append(t)
                zp.append(m.target_encoder(ob[:, anchor].unsqueeze(1))[:, 0])
            zh, zt, zp = torch.cat(zh), torch.cat(zt), torch.cat(zp)
            zm = zt.mean(0, keepdim=True).expand_as(zt)
            def nmse(p): return (F.mse_loss(p, zt) / zt.pow(2).mean()).item()
            def cosm(p): return F.cosine_similarity(p, zt, dim=-1).mean().item()
            log("\n==== HELD-OUT (DDP large-scale, LoRA, velocity+standardized) ====")
            log(f"NMSE  predictor {nmse(zh):.4f} | persistence {nmse(zp):.4f} | batch-mean {nmse(zm):.4f}")
            log(f"COS   predictor {cosm(zh):.4f} | persistence {cosm(zp):.4f}")
            g = nmse(zp) / max(nmse(zh), 1e-9)
            log(f"predictor vs persistence (NMSE): {g:.3f}x ({'WIN' if g>1.02 else 'LOSS'}) | "
                f"cos {'WIN' if cosm(zh)>cosm(zp)+0.002 else 'LOSS'}")
        out = Path("implementation/checkpoints"); out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__}, out / "sswm_ddp.pt")
        log(f"saved -> {out/'sswm_ddp.pt'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
