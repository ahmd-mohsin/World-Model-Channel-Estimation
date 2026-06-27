from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--step", type=float, default=0.05)   # metres/step (~0.6 lambda @3.5GHz)
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, use_pretrained=False)
    import drjit as dr
    import sionna.rt as rt

    scene = rt.load_scene(getattr(rt.scene, args.scene))
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=cfg.n_antennas, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    bbox = scene.mi_scene.bbox(); mn, mx = np.array(bbox.min), np.array(bbox.max)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    span_x, span_y = (mx[0] - mn[0]) * 0.3, (mx[1] - mn[1]) * 0.3
    tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
    scene.add(rt.Transmitter(name="tx", position=[float(cx), float(cy), tx_z]))
    scene.add(rt.Receiver(name="rx", position=[float(cx), float(cy), 1.5]))
    rx = scene.get("rx")
    solver = rt.PathSolver()
    n = cfg.n_subcarriers
    freqs = (np.arange(n) - n // 2) * 30e3
    fdr = dr.cuda.ad.Float(freqs.astype(np.float32))
    rng = np.random.default_rng(9000 + args.shard)

    def chan(pos):
        rx.position = np.asarray(pos, dtype=np.float32)
        H = solver(scene, max_depth=3).cfr(frequencies=fdr, normalize=False, out_type="torch")
        H = H.reshape(-1, n)[: cfg.n_antennas]
        if H.shape[0] < cfg.n_antennas:
            H = torch.cat([H, torch.zeros(cfg.n_antennas - H.shape[0], n, dtype=H.dtype, device=H.device)], 0)
        return (torch.stack([H.real, H.imag], 0) * 1e6).detach().cpu()

    seqs, acts, kept, tries = [], [], 0, 0
    t0 = time.time()
    while kept < args.n and tries < args.n * 6:
        tries += 1
        start = np.array([cx + rng.uniform(-span_x, span_x), cy + rng.uniform(-span_y, span_y), 1.5])
        theta = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.5, 1.5)                      # per-sequence speed multiplier
        vx, vy = np.cos(theta) * speed, np.sin(theta) * speed
        d = np.array([vx, vy, 0.0]) * args.step
        frames = [chan(start + d * t) for t in range(cfg.seq_len)]
        seq = torch.stack(frames, 0)
        if seq.abs().max().item() < 1e-6:
            continue
        # Action per step = the (constant) velocity vector that drives channel evolution,
        # plus its magnitude. This is the physical control the predictor conditions on.
        a = torch.tensor([vx, vy, speed, theta / np.pi], dtype=torch.float32)
        acts.append(a.unsqueeze(0).repeat(cfg.seq_len, 1))
        seqs.append(seq); kept += 1
        if kept % 100 == 0:
            print(f"[{args.scene} s{args.shard}] {kept}/{args.n} ({tries} tries, {time.time()-t0:.0f}s)", flush=True)

    data = torch.stack(seqs, 0)
    action = torch.stack(acts, 0)        # (N, T, 4)
    torch.save({"data": data, "action": action, "scene": args.scene, "step": args.step}, args.out)
    print(f"[{args.scene} s{args.shard}] saved data {tuple(data.shape)} action {tuple(action.shape)} "
          f"-> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
