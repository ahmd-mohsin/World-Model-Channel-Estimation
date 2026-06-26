from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig


def build_scene(scene_name, cfg):
    import sionna.rt as rt
    scene = rt.load_scene(getattr(rt.scene, scene_name))
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=cfg.n_antennas, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    bbox = scene.mi_scene.bbox()
    mn, mx = np.array(bbox.min), np.array(bbox.max)
    return scene, rt, mn, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, use_pretrained=False)
    import drjit as dr
    scene, rt, mn, mx = build_scene(args.scene, cfg)
    solver = rt.PathSolver()
    n = cfg.n_subcarriers
    freqs = (np.arange(n) - n // 2) * 30e3
    fdr = dr.cuda.ad.Float(freqs.astype(np.float32))
    rng = np.random.default_rng(7000 + args.shard)

    # TX fixed high near scene center; RX sampled across the central 60% footprint at street level.
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    span_x, span_y = (mx[0] - mn[0]) * 0.3, (mx[1] - mn[1]) * 0.3
    tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
    scene.add(rt.Transmitter(name="tx", position=[float(cx), float(cy), tx_z]))
    scene.add(rt.Receiver(name="rx", position=[float(cx), float(cy), 1.5]))
    rx = scene.get("rx")

    def chan(pos):
        rx.position = np.asarray(pos, dtype=np.float32)
        H = solver(scene, max_depth=3).cfr(frequencies=fdr, normalize=False, out_type="torch")
        H = H.reshape(-1, n)[: cfg.n_antennas]
        if H.shape[0] < cfg.n_antennas:
            H = torch.cat([H, torch.zeros(cfg.n_antennas - H.shape[0], n, dtype=H.dtype, device=H.device)], 0)
        return (torch.stack([H.real, H.imag], 0) * 1e6).detach().cpu()

    seqs, positions, kept, tries = [], [], 0, 0
    t0 = time.time()
    while kept < args.n and tries < args.n * 5:
        tries += 1
        start = np.array([cx + rng.uniform(-span_x, span_x),
                          cy + rng.uniform(-span_y, span_y), 1.5])
        theta = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(theta), np.sin(theta), 0.0])
        frames = [chan(start + d * 0.015 * t) for t in range(cfg.seq_len)]
        seq = torch.stack(frames, 0)
        if seq.abs().max().item() < 1e-6:   # no paths reached -> skip dead location
            continue
        seqs.append(seq); positions.append(start[:2]); kept += 1
        if kept % 50 == 0:
            print(f"[{args.scene} shard {args.shard}] {kept}/{args.n} ({tries} tries, {time.time()-t0:.0f}s)", flush=True)

    data = torch.stack(seqs, 0)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)
    torch.save({"data": data, "pos": pos, "scene": args.scene}, args.out)
    print(f"[{args.scene} shard {args.shard}] saved {tuple(data.shape)} -> {args.out} "
          f"({tries} tries, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
