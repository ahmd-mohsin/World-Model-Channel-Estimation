"""Measure the world-model predictor's NMSE vs prediction HORIZON k, against baselines.

A world model's value is the horizon curve: a good predictor should beat persistence by an
increasing margin as k grows (persistence decays; learned dynamics extrapolate). We evaluate the
trained SSWM predictor rolled k=1..K steps and compare to:
  - persistence:  z~_t             (echo present embedding)
  - AR(1):        fit z~_{t+1} ~ A z~_t on train, roll k steps

Runs on the box against a trained checkpoint.
    python scripts/eval-predictor-horizon.py --data_dir data/act60k --ckpt <path>
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", default="implementation/checkpoints/sswm_e2e_snrrange.pt")
    ap.add_argument("--Kmax", type=int, default=6)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = SSWM(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.05, seed=0)
    o, a = ds.all("test", device=dev)
    o, a = o[:2000], a[:2000]
    T = cfg.seq_len

    # Encode all frames' target embeddings once (z~_t for every t), for baselines.
    def emb(frames):  # (B,2,ant,sub) -> (B,d)
        outs = []
        for i in range(0, frames.shape[0], 256):
            outs.append(m.target_encoder(frames[i:i+256].unsqueeze(1))[:, 0])
        return torch.cat(outs)
    Z = torch.stack([emb(o[:, t]) for t in range(T)], dim=1)   # (B,T,d)

    # AR(1) on the embedding sequence: fit z_{t+1} ≈ W z_t (ridge) on train split.
    otr, atr = ds.all("train", device=dev); otr = otr[:4000]
    Ztr = torch.stack([emb(otr[:, t]) for t in range(T)], dim=1)
    X = Ztr[:, :-1].reshape(-1, cfg.embed_dim); Y = Ztr[:, 1:].reshape(-1, cfg.embed_dim)
    lam = 1e-2 * X.shape[0]
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(cfg.embed_dim, device=dev), X.T @ Y).T  # (d,d)

    def nmse(p, tgt): return (F.mse_loss(p, tgt) / tgt.pow(2).mean()).item()

    print(f"ckpt={Path(args.ckpt).name}  test={o.shape[0]}  T={T}\n")
    print(f"{'k':>3} {'persist':>9} {'AR(1)':>9} {'SSWM':>9}   winner")
    print("-" * 46)
    curve = {}
    for k in range(1, min(args.Kmax, T - 1) + 1):
        anchor = T - 1 - k
        z_t = m.encode_sequence(o, a)[:, anchor]
        z_hat = m.predictor(z_t, a[:, anchor:anchor + k])          # SSWM prediction
        if getattr(cfg, "residual_prediction", False):
            z_hat = z_hat + Z[:, anchor]
        from implementation.config import scale_embedding
        z_hat = scale_embedding(z_hat, cfg)
        tgt = Z[:, anchor + k]
        # baselines
        z_pers = Z[:, anchor]
        z_ar = Z[:, anchor]
        for _ in range(k):
            z_ar = z_ar @ W.T
        p, ar, sw = nmse(z_pers, tgt), nmse(z_ar, tgt), nmse(z_hat, tgt)
        best = min([("persist", p), ("AR1", ar), ("SSWM", sw)], key=lambda x: x[1])[0]
        curve[k] = {"persist": p, "ar1": ar, "sswm": sw}
        print(f"{k:>3} {p:>9.4f} {ar:>9.4f} {sw:>9.4f}   {best}")
    Path("dashboard").mkdir(exist_ok=True)
    Path("dashboard/horizon.json").write_text(json.dumps(curve))
    print("\nsaved dashboard/horizon.json")


if __name__ == "__main__":
    main()
