"""Generate a LONG continuous receiver walk through a real Sionna scene, plus the scene's building
mesh, for the real-time dashboard. Runs ON THE BOX (needs Sionna).

Unlike the T=8 sub-wavelength training sequences, this walks the RX in a straight line across a large
fraction of the scene for hundreds of steps, so the channel genuinely evolves and the animation shows
predictions tracking a moving channel over a long run.

    python scripts/gen-long-walk.py --scene etoile --steps 400 --step 0.6 --out /tmp/walk_etoile.pt

Output .pt: {mesh:{V,F,bbox,tx}, walk:{data (L,2,A,S), pos (L,3)}, scene, step}
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="etoile")
    ap.add_argument("--steps", type=int, default=400)   # walk length L
    ap.add_argument("--step", type=float, default=0.6)  # metres/step (larger -> more evolution)
    ap.add_argument("--n_ant", type=int, default=8)
    ap.add_argument("--n_sub", type=int, default=32)
    ap.add_argument("--max_faces", type=int, default=45000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import drjit as dr
    import sionna.rt as rt

    scene = rt.load_scene(getattr(rt.scene, args.scene))
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=args.n_ant, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                    horizontal_spacing=0.5, pattern="iso", polarization="V")
    bbox = scene.mi_scene.bbox(); mn, mx = np.array(bbox.min), np.array(bbox.max)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
    scene.add(rt.Transmitter(name="tx", position=[float(cx), float(cy), tx_z]))
    scene.add(rt.Receiver(name="rx", position=[float(cx), float(cy), 1.5]))
    rx = scene.get("rx")
    solver = rt.PathSolver()
    n = args.n_sub
    freqs = (np.arange(n) - n // 2) * 30e3
    fdr = dr.cuda.ad.Float(freqs.astype(np.float32))

    # ---- building mesh ----
    verts, faces, off = [], [], 0
    for o in scene.objects.values():
        m = o.mi_mesh
        vb = np.array(m.vertex_positions_buffer()).reshape(-1, 3)
        fb = np.array(m.faces_buffer()).reshape(-1, 3).astype(np.int64)
        verts.append(vb); faces.append(fb + off); off += m.vertex_count()
    V = np.concatenate(verts, 0); F = np.concatenate(faces, 0)
    rng = np.random.default_rng(0)
    if F.shape[0] > args.max_faces:
        F = F[rng.choice(F.shape[0], args.max_faces, replace=False)]

    # ---- long straight walk through the CENTRAL footprint (where the generator sampled RX and
    # coverage exists). Centre the walk on the scene centre and span +-25% so it stays in-coverage. ----
    theta = rng.uniform(0, 2 * np.pi)
    dirv = np.array([np.cos(theta), np.sin(theta), 0.0])
    span = min(mx[0] - mn[0], mx[1] - mn[1])
    half = 0.25 * span                                      # walk extent = 50% of the smaller span
    walk_len = args.steps * args.step
    if walk_len > 2 * half:                                 # shrink step so the walk fits the footprint
        args.step = (2 * half) / args.steps
    start = np.array([cx, cy, 1.5]) - dirv * (args.step * args.steps / 2)  # centred on scene centre

    def chan(pos):
        rx.position = np.asarray(pos, dtype=np.float32)
        H = solver(scene, max_depth=3).cfr(frequencies=fdr, normalize=False, out_type="torch")
        H = H.reshape(-1, n)[: args.n_ant]
        if H.shape[0] < args.n_ant:
            H = torch.cat([H, torch.zeros(args.n_ant - H.shape[0], n, dtype=H.dtype, device=H.device)], 0)
        return (torch.stack([H.real, H.imag], 0) * 1e6).detach().cpu()

    data, pos = [], []
    t = 0
    for s in range(args.steps):
        p = start + dirv * (s * args.step)
        H = chan(p)
        if H.abs().max().item() < 1e-6:   # dead spot: hold previous channel so the walk stays continuous
            if data:
                H = data[-1].clone()
            else:
                continue
        data.append(H); pos.append(p.astype(np.float32))
        if s % 50 == 0:
            print(f"[{args.scene}] step {s}/{args.steps} kept={len(data)}", flush=True)
    if not data:
        raise SystemExit("no coverage along walk; try a different scene/heading")
    data = torch.stack(data, 0)                 # (L,2,A,S)
    pos = np.stack(pos, 0)                       # (L,3)

    torch.save({
        "mesh": {"V": np.round(V, 2).astype(np.float32), "F": F.astype(np.int32),
                 "bbox_min": mn, "bbox_max": mx, "tx": np.array([cx, cy, tx_z], np.float32)},
        "walk": {"data": data, "pos": pos},
        "scene": args.scene, "step": args.step, "n_ant": args.n_ant, "n_sub": args.n_sub,
    }, args.out)
    print(f"saved {args.out}: L={data.shape[0]} steps, mesh V={V.shape[0]} F={F.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
