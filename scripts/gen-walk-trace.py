"""Turn a long continuous walk (from gen-long-walk.py) into a dashboard trace: at every position
along the walk, run the beam world model's ESTIMATION (sparse noisy pilots -> estimate) and, using a
sliding history window, its PREDICTION of the next channel. Emits the building MESH + a long
per-step timeline the dashboard animates continuously.

    python scripts/gen-walk-trace.py --ckpt implementation/checkpoints/beam_dft_s4.pt \
        --walk data/demo/walk_etoile.pt --stride 4 --out dashboard/walk_trace.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel

def nmse(a, b): return float(np.mean((a - b) ** 2) / (np.mean(b ** 2) + 1e-12))
def mag(H):     return np.abs(H[0] + 1j * H[1]).mean(axis=0)          # (S,)
def grid(H):    return np.abs(H[0] + 1j * H[1])                       # (A,S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--snr_db", type=float, default=10.0)
    ap.add_argument("--win", type=int, default=8, help="history window T fed to the model")
    ap.add_argument("--pred_k", type=int, default=3, help="prediction horizon to display")
    ap.add_argument("--out", default="dashboard/walk_trace.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = BeamWorldModel(cfg, sparse_stride=args.stride, deep_fusion=True)
    m.load_state_dict(ck["model"], strict=False); m.eval()

    blob = torch.load(args.walk, map_location="cpu", weights_only=False)
    W = blob["walk"]["data"]                     # (L,2,A,S)
    pos = np.asarray(blob["walk"]["pos"])         # (L,3)
    mesh = blob["mesh"]; scene = blob["scene"]
    L, _, na, ns = W.shape
    T = args.win

    # standardize channels on the walk (per-plane); the model was trained on standardized data.
    mu = W.mean(dim=(0, 2, 3), keepdim=True); sd = W.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    Wn = (W - mu) / sd
    # action per step from position deltas (vx,vy,speed,theta/pi), standardized roughly like training
    d = np.diff(pos, axis=0, prepend=pos[:1])
    speed = np.linalg.norm(d[:, :2], axis=1) + 1e-9
    theta = np.arctan2(d[:, 1], d[:, 0])
    act = np.stack([d[:, 0], d[:, 1], speed, theta / np.pi], axis=1).astype(np.float32)
    act = (act - act.mean(0)) / (act.std(0) + 1e-6)
    A = torch.from_numpy(act)

    g = torch.Generator().manual_seed(0)
    steps = []
    with torch.no_grad():
        for t in range(T, L):
            o = Wn[t - T + 1:t + 1].unsqueeze(0)         # (1,T,2,A,S) history ending at t
            a = A[t - T + 1:t + 1].unsqueeze(0)
            # ESTIMATION of the current channel from sparse noisy pilots
            Hhat, Hcl, Ymask, _ = m.estimate_channel_sparse(o, a, snr_db=args.snr_db,
                                                            stride=args.stride, noise_gen=g)
            Hh = Hhat[0].numpy(); Hc = Hcl[0].numpy(); Y = Ymask[0].numpy()
            # PREDICTION: from anchor (one before end), roll pred_k ahead; compare to the true future
            beam = m._to(o); _, z, h = m.encode(beam, a)
            anchor = T - 1 - args.pred_k
            pred_mag = pers_mag = None; pnm = qnm = None
            if anchor >= 0:
                bp = m.predict_beam(z[:, anchor], h[:, anchor], a[:, anchor:anchor + args.pred_k])
                Hp = m._from(bp)[0].numpy(); Htru = o[0, anchor + args.pred_k].numpy()
                Hpe = o[0, anchor].numpy()
                pred_mag = mag(Hp).round(4).tolist(); pers_mag = mag(Hpe).round(4).tolist()
                pnm = round(nmse(Hp, Htru), 4); qnm = round(nmse(Hpe, Htru), 4)
            steps.append({
                "t": t, "pos": [round(float(x), 2) for x in pos[t]],
                "truth_mag": mag(Hc).round(4).tolist(),
                "obs_mag": mag(Y).round(4).tolist(),
                "est_mag": mag(Hh).round(4).tolist(),
                "truth_grid": grid(Hc).round(3).tolist(),
                "est_grid": grid(Hh).round(3).tolist(),
                "est_nmse": round(nmse(Hh, Hc), 5),
                "pred_mag": pred_mag, "pred_nmse": pnm, "persist_nmse": qnm,
            })

    # decimate mesh vertices/faces already capped upstream; ship flat arrays
    V = np.asarray(mesh["V"]).astype(np.float32)
    F = np.asarray(mesh["F"]).astype(np.int32)
    out = {
        "scene": scene, "stride": args.stride, "snr_db": args.snr_db,
        "n_ant": int(na), "n_sub": int(ns), "pred_k": args.pred_k,
        "mesh": {"V": np.round(V, 2).flatten().tolist(), "F": F.flatten().tolist(),
                 "bbox_min": np.asarray(mesh["bbox_min"]).tolist(),
                 "bbox_max": np.asarray(mesh["bbox_max"]).tolist(),
                 "tx": np.asarray(mesh["tx"]).tolist()},
        "path": [[round(float(x), 2) for x in p] for p in pos.tolist()],
        "steps": steps,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    sz = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {len(steps)} steps, mesh {V.shape[0]}v/{F.shape[0]}f, {sz:.1f} MB")
    est = [s["est_nmse"] for s in steps]
    pn = [s["pred_nmse"] for s in steps if s["pred_nmse"] is not None]
    print(f"  est NMSE mean {np.mean(est):.4f}  |  pred NMSE mean {np.mean(pn):.4f}")


if __name__ == "__main__":
    main()
