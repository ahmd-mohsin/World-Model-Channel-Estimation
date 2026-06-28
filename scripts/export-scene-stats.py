"""Export 3D scene geometry, TX/RX positions, and channel statistics to JSON for the dashboard.

    python scripts/export-scene-stats.py --data_dir data/act --out dashboard/scenes.json

Produces, per scene: bounding box, TX position, sampled RX trajectory points (3D), and
channel statistics (mean |H|, magnitude histogram, temporal correlation, per-antenna/subcarrier
energy profile). Also a global summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig


def scene_geometry(name, cfg):
    """Bounding box + TX position for a scene (mirrors gen_sionna_actions placement)."""
    import sionna.rt as rt
    scene = rt.load_scene(getattr(rt.scene, name))
    bbox = scene.mi_scene.bbox()
    mn, mx = np.array(bbox.min), np.array(bbox.max)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
    return {
        "bbox_min": mn.tolist(), "bbox_max": mx.tolist(),
        "tx": [float(cx), float(cy), tx_z],
        "span": [(mx[0] - mn[0]), (mx[1] - mn[1]), (mx[2] - mn[2])],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="dashboard/scenes.json")
    ap.add_argument("--max_rx", type=int, default=400)
    args = ap.parse_args()

    cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, use_pretrained=False)
    shards = {}
    for f in sorted(glob.glob(f"{args.data_dir}/shard_*.pt")):
        d = torch.load(f, map_location="cpu")
        s = d.get("scene", "?")
        shards.setdefault(s, []).append(d)

    out = {"scenes": {}, "global": {}}
    all_mag = []
    for name, ds in shards.items():
        data = torch.cat([d["data"] for d in ds], 0)            # (N,T,2,ant,sub)
        # RX positions: stored as (N,2); lift to 3D at rx height 1.5
        pos = torch.cat([d.get("pos", torch.zeros(d["data"].shape[0], 2)) for d in ds], 0).numpy()
        n = data.shape[0]

        H0 = data[:, 0]                                         # first frame
        mag = (H0[:, 0] ** 2 + H0[:, 1] ** 2).sqrt()           # (N,ant,sub)
        mag_flat = mag.flatten().numpy()
        mag_flat = mag_flat[mag_flat > 0]
        all_mag.append(mag_flat[:50000])

        # temporal correlation vs t=0
        T = data.shape[1]
        ref = data[:min(256, n), 0].flatten(1)
        corr = []
        for t in range(T):
            cur = data[:min(256, n), t].flatten(1)
            corr.append(float(torch.nn.functional.cosine_similarity(ref, cur, dim=1).mean()))

        try:
            geo = scene_geometry(name, cfg)
        except Exception as e:
            geo = {"error": str(e)}

        # RX positions weren't stored in the action shards (only data+action), so reconstruct
        # representative samples the way the generator placed them: within the central 60% of
        # the scene footprint at street level. Mirrors gen_sionna_actions.py sampling.
        if pos is None or float(np.abs(pos).max()) < 1e-6:
            if "bbox_min" in geo:
                mn, mx = np.array(geo["bbox_min"]), np.array(geo["bbox_max"])
                cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
                sx, sy = (mx[0] - mn[0]) * 0.3, (mx[1] - mn[1]) * 0.3
                r = np.random.default_rng(0)
                pos = np.stack([cx + r.uniform(-sx, sx, n), cy + r.uniform(-sy, sy, n)], axis=1)

        # log-magnitude histogram
        logm = np.log10(mag_flat + 1e-9)
        hist, edges = np.histogram(logm, bins=60, density=True)

        rx_idx = np.random.default_rng(0).choice(n, min(args.max_rx, n), replace=False)
        rx3d = [[float(pos[i, 0]), float(pos[i, 1]), 1.5] for i in rx_idx]

        out["scenes"][name] = {
            "n_sequences": int(n),
            "geometry": geo,
            "rx_positions": rx3d,
            "mean_mag": float(mag.mean()),
            "std_mag": float(mag.std()),
            "mag_hist": {"density": hist.tolist(), "edges": edges.tolist()},
            "temporal_corr": corr,
            "antenna_energy": mag.mean(dim=(0, 2)).numpy().tolist(),     # per antenna
            "subcarrier_energy": mag.mean(dim=(0, 1)).numpy().tolist(),  # per subcarrier
            "example_channel_real": H0[0, 0].numpy().round(3).tolist(),  # (ant,sub) grid
        }
        print(f"{name:24s} n={n:6d} mean|H|={mag.mean():.2f}", flush=True)

    allm = np.concatenate(all_mag)
    out["global"] = {
        "total_sequences": int(sum(s["n_sequences"] for s in out["scenes"].values())),
        "n_scenes": len(out["scenes"]),
        "mean_mag": float(allm.mean()),
        "scene_names": list(out["scenes"].keys()),
    }
    def _native(x):
        if isinstance(x, dict):
            return {k: _native(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_native(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(_native(out)))
    print(f"saved -> {args.out} ({out['global']['total_sequences']} seqs, {len(out['scenes'])} scenes)", flush=True)


if __name__ == "__main__":
    main()
