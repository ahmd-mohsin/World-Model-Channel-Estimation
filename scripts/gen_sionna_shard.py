from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.wireless_data import SionnaChannelGenerator, SionnaSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--scene", type=str, default="munich")
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=2,
                     embed_dim=256, backbone="lwm", use_pretrained=False)
    gen = SionnaChannelGenerator(cfg, SionnaSpec(scene=args.scene))
    gen._lazy_init()

    rng = np.random.default_rng(1000 + args.shard)
    seqs, positions = [], []
    for i in range(args.n):
        start = np.array([rng.uniform(20, 80), rng.uniform(40, 120), 1.5])
        theta = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        frames = []
        for t in range(cfg.seq_len):
            pos = start + direction * gen.spec.step_size_m * t
            pos[2] = gen.spec.rx_height
            H = gen._channel_at(pos).detach().cpu() * gen.spec.channel_scale
            frames.append(torch.stack([H.real, H.imag], 0))
        seqs.append(torch.stack(frames, 0))
        positions.append(start[:2])
        if (i + 1) % 20 == 0:
            print(f"[shard {args.shard}] {i+1}/{args.n}", flush=True)

    data = torch.stack(seqs, 0)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)
    torch.save({"data": data, "pos": pos}, args.out)
    print(f"[shard {args.shard}] saved {tuple(data.shape)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
