"""Generate TRUE-MIMO velocity-action Sionna sequences: multi-antenna base station AND multi-antenna
user equipment, so the per-subcarrier channel is a full N_tx x N_rx matrix (not MISO/SISO).

The spatial matrix is folded into a single "spatial" axis of size N_tx*N_rx so the rest of the
pipeline (orthonormal 2D beamspace over spatial x subcarrier, the world model, ReEsNet) is unchanged.
Beamspace stays energy-preserving and exactly invertible.

    python scripts/gen_sionna_mimo.py --shard 0 --n 8000 --scene munich --n_tx 8 --n_rx 4 \
        --n_sub 32 --step 0.15 --speed_lo 0.3 --speed_hi 2.0 --out data/mimo/shard_0.pt

Stored: data (N,T,2,N_tx*N_rx,N_sub), action (N,T,4), scene, step, n_tx, n_rx.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--speed_lo", type=float, default=0.3)
    ap.add_argument("--speed_hi", type=float, default=2.0)
    ap.add_argument("--n_tx", type=int, default=8)      # BS antennas
    ap.add_argument("--n_rx", type=int, default=4)      # UE antennas (>1 => true MIMO)
    ap.add_argument("--n_sub", type=int, default=32)
    args = ap.parse_args()

    import drjit as dr
    import sionna.rt as rt

    scene = rt.load_scene(getattr(rt.scene, args.scene))
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=args.n_tx, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=args.n_rx, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    bbox = scene.mi_scene.bbox(); mn, mx = np.array(bbox.min), np.array(bbox.max)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    span_x, span_y = (mx[0] - mn[0]) * 0.3, (mx[1] - mn[1]) * 0.3
    tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
    scene.add(rt.Transmitter(name="tx", position=[float(cx), float(cy), tx_z]))
    scene.add(rt.Receiver(name="rx", position=[float(cx), float(cy), 1.5]))
    rx = scene.get("rx")
    solver = rt.PathSolver()
    n = args.n_sub
    freqs = (np.arange(n) - n // 2) * 30e3
    fdr = dr.cuda.ad.Float(freqs.astype(np.float32))
    rng = np.random.default_rng(9000 + args.shard)
    S = args.n_tx * args.n_rx

    def chan(pos):
        rx.position = np.asarray(pos, dtype=np.float32)
        # cfr returns the full MIMO CFR; flatten TX,RX antenna dims -> (n_tx*n_rx, n_sub)
        H = solver(scene, max_depth=3).cfr(frequencies=fdr, normalize=False, out_type="torch")
        H = H.reshape(-1, n)                             # (?, n_sub) collapse all antenna dims
        if H.shape[0] < S:
            H = torch.cat([H, torch.zeros(S - H.shape[0], n, dtype=H.dtype, device=H.device)], 0)
        H = H[:S]
        return (torch.stack([H.real, H.imag], 0) * 1e6).detach().cpu()   # (2,S,n_sub)

    seqs, acts, kept, tries = [], [], 0, 0
    t0 = time.time()
    while kept < args.n and tries < args.n * 6:
        tries += 1
        start = np.array([cx + rng.uniform(-span_x, span_x), cy + rng.uniform(-span_y, span_y), 1.5])
        theta = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(args.speed_lo, args.speed_hi)
        vx, vy = np.cos(theta) * speed, np.sin(theta) * speed
        d = np.array([vx, vy, 0.0]) * args.step
        frames = [chan(start + d * t) for t in range(args.seq_len)]
        seq = torch.stack(frames, 0)
        if seq.abs().max().item() < 1e-6:
            continue
        a = torch.tensor([vx, vy, speed, theta / np.pi], dtype=torch.float32)
        acts.append(a.unsqueeze(0).repeat(args.seq_len, 1))
        seqs.append(seq); kept += 1
        if kept % 100 == 0:
            print(f"[{args.scene} s{args.shard}] {kept}/{args.n} ({tries} tries, {time.time()-t0:.0f}s)", flush=True)

    data = torch.stack(seqs, 0)                 # (N,T,2,S,n_sub)
    action = torch.stack(acts, 0)               # (N,T,4)
    torch.save({"data": data, "action": action, "scene": args.scene, "step": args.step,
                "n_tx": args.n_tx, "n_rx": args.n_rx}, args.out)
    print(f"[{args.scene} s{args.shard}] saved data {tuple(data.shape)} "
          f"(n_tx={args.n_tx} n_rx={args.n_rx}) -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
