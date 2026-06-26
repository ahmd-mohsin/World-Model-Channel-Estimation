from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)


def load_by_scene(data_dir):
    by_scene = {}
    pos_by_scene = {}
    for f in sorted(glob.glob(f"{data_dir}/shard_*.pt")):
        d = torch.load(f, map_location="cpu")
        s = d.get("scene", "?")
        by_scene.setdefault(s, []).append(d["data"])
        pos_by_scene.setdefault(s, []).append(d["pos"])
    return ({s: torch.cat(v, 0) for s, v in by_scene.items()},
            {s: torch.cat(v, 0) for s, v in pos_by_scene.items()})


def fig_rx_coverage(pos_by_scene):
    n = len(pos_by_scene)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (scene, pos) in zip(axes, pos_by_scene.items()):
        p = pos.numpy()
        ax.scatter(p[:, 0], p[:, 1], s=6, alpha=0.5)
        ax.set_title(f"{scene}\n({len(p)} RX positions)", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
    fig.suptitle("Receiver coverage per Sionna scene (sampled trajectory start points)", fontsize=12)
    p = OUT / "scene_coverage.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def fig_magnitude_dist(by_scene):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for scene, data in by_scene.items():
        mag = (data[:, 0, 0] ** 2 + data[:, 0, 1] ** 2).sqrt().flatten().numpy()
        mag = mag[mag > 0]
        axes[0].hist(np.log10(mag + 1e-9), bins=80, alpha=0.5, label=scene, density=True)
    axes[0].set_title("Channel magnitude distribution (log10 |H|) per scene")
    axes[0].set_xlabel("log10 |H|"); axes[0].set_ylabel("density"); axes[0].legend(); axes[0].grid(alpha=0.3)

    means = {s: (d[:, 0, 0] ** 2 + d[:, 0, 1] ** 2).sqrt().mean().item() for s, d in by_scene.items()}
    axes[1].bar(list(means.keys()), list(means.values()), color="#4c72b0")
    axes[1].set_title("Mean |H| per scene  (scale diversity the model must handle)")
    axes[1].set_ylabel("mean |H|"); axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Per-scene channel distributions", fontsize=12)
    p = OUT / "scene_distributions.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def fig_example_channels(by_scene):
    scenes = list(by_scene.keys())
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.2 * len(scenes), 3.4))
    if len(scenes) == 1:
        axes = [axes]
    for ax, scene in zip(axes, scenes):
        H = by_scene[scene][0, 0, 0]  # real part, first seq, t=0
        im = ax.imshow(H.numpy(), aspect="auto", cmap="RdBu_r")
        ax.set_title(scene, fontsize=10); ax.set_xlabel("subcarrier")
        if ax is axes[0]:
            ax.set_ylabel("antenna")
    fig.suptitle("Example channel (Re{H}) per scene", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02)
    p = OUT / "scene_examples.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def fig_temporal_corr(by_scene):
    fig, ax = plt.subplots(figsize=(8, 5))
    for scene, data in by_scene.items():
        d = data[:64]  # (N,T,2,ant,sub)
        T = d.shape[1]
        ref = d[:, 0]
        corr = []
        for t in range(T):
            a = ref.flatten(1); b = d[:, t].flatten(1)
            c = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
            corr.append(c)
        ax.plot(range(T), corr, "-o", ms=3, label=scene)
    ax.set_title("Temporal correlation of channel sequences (cosine vs t=0)")
    ax.set_xlabel("timestep"); ax.set_ylabel("mean cosine similarity"); ax.legend(); ax.grid(alpha=0.3)
    p = OUT / "scene_temporal_corr.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/large"
    by_scene, pos_by_scene = load_by_scene(data_dir)
    print("scenes:", {s: tuple(v.shape) for s, v in by_scene.items()})
    for fn in (fig_rx_coverage(pos_by_scene), fig_magnitude_dist(by_scene),
               fig_example_channels(by_scene), fig_temporal_corr(by_scene)):
        print("saved", fn)


if __name__ == "__main__":
    main()
