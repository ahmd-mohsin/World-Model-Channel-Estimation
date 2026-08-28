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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="full", choices=list(BeamWorldModel.ABLATIONS))
    ap.add_argument("--n_ant", type=int, default=8)
    ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    dist.init_process_group("nccl")
    rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    main_rank = rank == 0

    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, horizon_k=3, action_dim=4,
                     embed_dim=128, state_dim=64, latent_dim=128, use_pretrained=False, unet_base_ch=48)
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=args.seed, holdout_scene=args.holdout_scene)
    if main_rank:
        print(f"world={dist.get_world_size()} | train {len(ds.train_idx)} test {len(ds.test_idx)} "
              f"| holdout={args.holdout_scene} | scenes {ds.scenes}", flush=True)
    m = BeamWorldModel(cfg, ablate=args.ablate).to(dev)
    if main_rank:
        print(f"params: {sum(p.numel() for p in m.parameters()):,} | ablate={args.ablate}", flush=True)
    ddp = DDP(m, device_ids=[local], find_unused_parameters=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    rng = np.random.default_rng(1000 + rank + 100 * args.seed)
    ng = torch.Generator(device=dev).manual_seed(rank + 100 * args.seed)

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
        # route outputs by tag: ab_* -> results/ablation, else results/stress
        STRESS = Path("results/ablation" if args.tag.startswith("ab_") else "results/stress")
        m.eval()
        o, a = ds.all("test", device=dev); o, a = o[:2000], a[:2000]
        Htr = ds.all("train", device=dev)[0][:, -1]
        Hte = o[:, -1]
        g = torch.Generator(device=dev).manual_seed(0)

        # ---- ESTIMATION: NMSE vs SNR (wide range) vs LS / MMSE ----
        sweep = {}
        print("\nSNR      LS     MMSE    BEAM", flush=True)
        for snr in [-5, 0, 5, 10, 15, 20, 30]:
            Hhat, _ = m.estimate_channel(o, a, snr_db=float(snr), noise_gen=g)
            Yte = add_noise(Hte, snr, generator=g)
            ls = nmse(Yte, Hte); mm = nmse(mmse_estimate(Yte, Htr, snr), Hte); bm = nmse(Hhat, Hte)
            sweep[snr] = {"ls": ls, "mmse": mm, "beam": bm}
            print(f"{snr:3d}  {ls:7.4f} {mm:7.4f} {bm:7.4f}", flush=True)
        STRESS.mkdir(parents=True, exist_ok=True)
        (DASH / f"eval_{args.tag}.json").write_text(json.dumps(sweep))
        (STRESS / f"eval_{args.tag}.json").write_text(json.dumps(sweep))

        # ---- PREDICTION: channel-space NMSE vs horizon k vs persistence / AR(1) ----
        # AR(1) fit in channel space on TRAIN: H_{t+1} ~ W H_t (complex, per-element ridge is
        # overkill; use a single global complex linear map on the vectorized channel).
        T = o.shape[1]
        otr_full = ds.all("train", device=dev)[0][:3000]
        def cvec(x):                       # (B,2,A,S) -> (B, A*S) complex
            return (x[:, 0] + 1j * x[:, 1]).reshape(x.shape[0], -1)
        Xtr = torch.cat([cvec(otr_full[:, t]) for t in range(T - 1)], 0)
        Ytr = torch.cat([cvec(otr_full[:, t + 1]) for t in range(T - 1)], 0)
        lam = 1e-2 * Xtr.shape[0]
        d = Xtr.shape[1]
        W = torch.linalg.solve(Xtr.conj().T @ Xtr + lam * torch.eye(d, dtype=Xtr.dtype, device=dev),
                               Xtr.conj().T @ Ytr).T          # (d,d) complex
        def uncvec(xc):                    # (B,d) complex -> (B,2,A,S)
            xr = xc.reshape(-1, cfg.n_antennas, cfg.n_subcarriers)
            return torch.stack([xr.real, xr.imag], 1)
        _, z, h = m.encode(m._to(o), a)        # working domain honors the ablation
        pred = {}
        print("\n  k   persist    AR(1)     BEAM", flush=True)
        for k in range(1, min(6, T - 1) + 1):
            anchor = T - 1 - k
            b_pred = m.predict_beam(z[:, anchor], h[:, anchor], a[:, anchor:anchor + k])
            H_pred = m._from(b_pred)
            H_true = o[:, anchor + k]
            H_pers = o[:, anchor]                       # persistence: last seen frame
            h_ar = cvec(o[:, anchor])
            for _ in range(k):
                h_ar = h_ar @ W.T
            H_ar = uncvec(h_ar)
            pr = nmse(H_pers, H_true); ar = nmse(H_ar, H_true); bm = nmse(H_pred, H_true)
            pred[k] = {"persist": pr, "ar1": ar, "beam": bm}
            print(f"{k:3d}  {pr:8.4f} {ar:8.4f} {bm:8.4f}", flush=True)
        (DASH / f"pred_{args.tag}.json").write_text(json.dumps(pred))
        (STRESS / f"pred_{args.tag}.json").write_text(json.dumps(pred))

        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": m.state_dict(), "config": cfg.__dict__}, OUT / f"beam_{args.tag}.pt")
        print(f"saved -> {OUT/f'beam_{args.tag}.pt'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
