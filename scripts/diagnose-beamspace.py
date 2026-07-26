"""GO/NO-GO for the beamspace direction. Answers, on real Sionna channels:
  1. Round-trip: to_beamspace -> from_beamspace exact?
  2. Sparsity: how few angle-delay taps hold 90/99% of the energy?
  3. Faithfulness: does a top-K sparse beamspace approximation reconstruct H far better than the
     lossy LWM latent (which was 0.31 NMSE)? This is the whole premise.

    python scripts/diagnose-beamspace.py --data_dir data/act60k
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.wireless_data.beamspace import to_beamspace, from_beamspace, sparsity

dev = "cuda" if torch.cuda.is_available() else "cpu"


def nmse(a, b): return ((a - b).pow(2).mean() / b.pow(2).mean()).item()


def topk_reconstruct(H, k_frac):
    """Keep only the top-k beamspace taps by magnitude, invert -> reconstruction NMSE."""
    B = to_beamspace(H)
    mag2 = B[:, 0] ** 2 + B[:, 1] ** 2                      # (N,ant,sub)
    n = mag2.shape[-1] * mag2.shape[-2]
    k = max(1, int(k_frac * n))
    flat = mag2.flatten(1)
    thr = torch.topk(flat, k, dim=1).values[:, -1:]        # per-sample k-th largest
    mask = (flat >= thr).reshape(mag2.shape).unsqueeze(1).float()   # (N,1,ant,sub)
    Bk = B * mask
    return nmse(from_beamspace(Bk), H)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.data_dir}/shard_*.pt"))
    H = torch.cat([torch.load(f, map_location="cpu")["data"][:, -1] for f in files], 0)[:4000].to(dev)
    print(f"channels: {tuple(H.shape)}\n")

    # 1. round-trip
    rt = nmse(from_beamspace(to_beamspace(H)), H)
    print(f"1. round-trip NMSE (should be ~0):            {rt:.2e}")

    # 2. sparsity
    s90 = sparsity(H, 0.90); s99 = sparsity(H, 0.99)
    print(f"2. taps for 90% energy: {s90*100:.1f}% of grid | 99%: {s99*100:.1f}%")

    # 3. faithfulness vs lossy latent (0.31)
    print("\n3. top-K beamspace reconstruction NMSE (vs LWM-latent baseline 0.31):")
    for kf in [0.02, 0.05, 0.10, 0.25, 0.50]:
        print(f"   keep {kf*100:4.0f}% of taps -> NMSE {topk_reconstruct(H, kf):.4f}")


if __name__ == "__main__":
    main()
