"""Export real Sionna scene GEOMETRY (building/street triangle meshes) to JSON for the dashboard.

    python scripts/export-scene-mesh.py --out dashboard/meshes.json

For each scene, pulls every object's Mitsuba mesh (vertex positions + face indices), plus the
TX position and sampled RX positions (placed as the generator does). The dashboard renders these
as actual 3D environments, not abstract boxes. Large meshes are face-capped to keep JSON light.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCENES = ["simple_street_canyon", "etoile", "florence", "munich", "san_francisco"]
MAX_FACES = 60000  # per scene cap (decimate by face subsampling if exceeded)


def get_mesh(scene):
    import drjit as dr
    verts_all, faces_all, offset = [], [], 0
    for o in scene.objects.values():
        m = o.mi_mesh
        nv, nf = m.vertex_count(), m.face_count()
        vb = np.array(m.vertex_positions_buffer()).reshape(-1, 3)
        fb = np.array(m.faces_buffer()).reshape(-1, 3).astype(np.int64)
        verts_all.append(vb)
        faces_all.append(fb + offset)
        offset += nv
    V = np.concatenate(verts_all, 0)
    F = np.concatenate(faces_all, 0)
    return V, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dashboard/meshes.json")
    ap.add_argument("--rx", type=int, default=300)
    args = ap.parse_args()

    import sionna.rt as rt

    out = {"scenes": {}}
    rng = np.random.default_rng(0)
    for name in SCENES:
        try:
            scene = rt.load_scene(getattr(rt.scene, name))
        except Exception as e:
            print(f"{name}: load failed {e}", flush=True); continue
        V, F = get_mesh(scene)
        bbox = scene.mi_scene.bbox()
        mn, mx = np.array(bbox.min), np.array(bbox.max)
        cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
        tx_z = float(mn[2] + 0.7 * (mx[2] - mn[2]))
        sx, sy = (mx[0] - mn[0]) * 0.3, (mx[1] - mn[1]) * 0.3
        rxp = np.stack([cx + rng.uniform(-sx, sx, args.rx),
                        cy + rng.uniform(-sy, sy, args.rx),
                        np.full(args.rx, 1.5)], axis=1)

        if F.shape[0] > MAX_FACES:
            keep = rng.choice(F.shape[0], MAX_FACES, replace=False)
            F = F[keep]

        # round to cut JSON size
        out["scenes"][name] = {
            "vertices": np.round(V, 2).astype(np.float32).flatten().tolist(),
            "faces": F.astype(np.int32).flatten().tolist(),
            "bbox_min": mn.tolist(), "bbox_max": mx.tolist(),
            "tx": [float(cx), float(cy), tx_z],
            "rx": np.round(rxp, 2).tolist(),
            "n_faces": int(F.shape[0]), "n_verts": int(V.shape[0]),
        }
        print(f"{name:22s} verts={V.shape[0]} faces={F.shape[0]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    sz = Path(args.out).stat().st_size / 1e6
    print(f"saved -> {args.out} ({sz:.1f} MB, {len(out['scenes'])} scenes)", flush=True)


if __name__ == "__main__":
    main()
