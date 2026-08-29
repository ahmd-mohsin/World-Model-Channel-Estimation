"""Downstream metric (#2): predict-then-precode SPECTRAL EFFICIENCY under CSI aging.

Feedback/processing delay means the precoder designed at time t is applied at t+k, on a channel that
has moved. We compare the CSI used to design the SVD precoder (evaluated on the TRUE future H_{t+k}):
  - perfect    : genie H_{t+k}                     (upper bound, no aging)
  - persist    : stale H_t (no prediction)         (what you get without a predictor)
  - WM         : Beam-WM selective-SSM prediction of H_{t+k}
  - Transformer: learned temporal-predictor baseline prediction of H_{t+k}   (fair strong baseline)

The question this answers: does a slightly-worse-NMSE predictor still recover the throughput that CSI
aging destroys, and is the WM competitive with the specialist Transformer on the metric that matters?

    python scripts/throughput-aging.py --wm_ckpt implementation/checkpoints/beam_indist_s0.pt \
        --tf_ckpt implementation/checkpoints/tp_tf_slow.pt --data data/mimo \
        --n_tx 8 --n_rx 4 --out results/throughput_aging_slow.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel
from implementation.task_heads.temporal_baseline import TemporalPredictor
from implementation.task_heads.ssm_predictor import DeepSSMPredictor
from implementation.wireless_data import ShardDataset
from implementation.wireless_data.beamspace import from_beamspace


def mismatched_rate(H_true, H_est, snr_lin, n_tx, n_rx):
    """H_*: (B,2,n_tx*n_rx,n_sub). Mean sum-rate (bits/s/Hz): SVD precoder from H_est on true H."""
    def to_mimo(H):
        c = (H[:, 0] + 1j * H[:, 1])                          # (B,S,F)
        B, S, F = c.shape
        return c.reshape(B, n_tx, n_rx, F).permute(0, 3, 2, 1)  # (B,F,nr,nt)
    Ht, He = to_mimo(H_true), to_mimo(H_est)
    r = min(n_tx, n_rx); p = snr_lin / r
    Ue, Se, Vhe = torch.linalg.svd(He)
    V = Vhe.conj().transpose(-1, -2)[..., :, :r]
    G = Ue[..., :, :r]
    E = G.conj().transpose(-1, -2) @ Ht @ V
    gain = E.abs() ** 2
    diag = torch.diagonal(gain, dim1=-2, dim2=-1)
    interf = gain.sum(-1) - diag
    sinr = (p * diag) / (p * interf + 1.0)
    return torch.log2(1 + sinr).sum(-1).mean().item()


def _load_compatible(m, sd):
    """Load only params whose shape matches — we need the predictor path (encoder/SSM/predictor/
    decoder); estimation heads may differ across persist-aug versions and are unused here."""
    cur = m.state_dict()
    ok = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    m.load_state_dict(ok, strict=False)
    return len(ok), len(cur)


def load_wm(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = BeamWorldModel(cfg).to(dev)
    n, tot = _load_compatible(m, ck["model"]); m.eval()
    print(f"WM: loaded {n}/{tot} params (predictor path)", flush=True)
    return m, cfg


def load_tf(ckpt, cfg, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m = TemporalPredictor(cfg, backbone=ck.get("backbone", "transformer")).to(dev)
    m.load_state_dict(ck["model"], strict=False); m.eval()
    return m


def load_ds(ckpt, cfg, dev, n_layers=3):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m = DeepSSMPredictor(cfg, n_layers=n_layers).to(dev)
    m.load_state_dict(ck["model"], strict=False); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_ckpt", required=True); ap.add_argument("--tf_ckpt", required=True)
    ap.add_argument("--ds_ckpt", default=None, help="deep-SSM L=3 predictor (gap-closed)")
    ap.add_argument("--ds_layers", type=int, default=3)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_ant", type=int, default=32); ap.add_argument("--n_sub", type=int, default=32)
    ap.add_argument("--n_tx", type=int, default=8); ap.add_argument("--n_rx", type=int, default=4)
    ap.add_argument("--snrs", default="0,10,20"); ap.add_argument("--out", default="results/throughput_aging.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    wm, cfg = load_wm(args.wm_ckpt, dev)
    tf = load_tf(args.tf_ckpt, cfg, dev)
    dsm = load_ds(args.ds_ckpt, cfg, dev, args.ds_layers) if args.ds_ckpt else None
    ds = ShardDataset(args.data, cfg, test_frac=0.05, seed=0)
    o, a = ds.all("test", device=dev); o, a = o[:1500], a[:1500]
    beam = wm._to(o); T = o.shape[1]
    snrs = [int(s) for s in args.snrs.split(",")]

    out = {}
    with torch.no_grad():
        _, z, h = wm.encode(beam, a)
        for k in range(1, min(6, T - 1) + 1):
            anchor = T - 1 - k
            H_true = o[:, anchor + k]
            H_persist = o[:, anchor]
            H_wm = wm._from(wm.predict_beam(z[:, anchor], h[:, anchor], a[:, anchor:anchor + k]))
            H_tf = from_beamspace(tf.predict(beam[:, :anchor + 1], a[:, :anchor + 1],
                                             a[:, anchor:anchor + k]))
            H_ds = (from_beamspace(dsm.predict(beam[:, :anchor + 1], a[:, :anchor + 1],
                                               a[:, anchor:anchor + k])) if dsm else None)
            out[k] = {}
            for snr in snrs:
                sl = 10 ** (snr / 10)
                rec = {"perfect": mismatched_rate(H_true, H_true, sl, args.n_tx, args.n_rx),
                       "persist": mismatched_rate(H_true, H_persist, sl, args.n_tx, args.n_rx),
                       "wm": mismatched_rate(H_true, H_wm, sl, args.n_tx, args.n_rx),
                       "tf": mismatched_rate(H_true, H_tf, sl, args.n_tx, args.n_rx)}
                if H_ds is not None:
                    rec["deepssm"] = mismatched_rate(H_true, H_ds, sl, args.n_tx, args.n_rx)
                out[k][snr] = rec

    print(f"{'k':>2} {'SNR':>4} {'perfect':>8} {'persist':>8} {'WM':>8} {'dSSM':>8} {'TF':>8}  "
          f"{'dSSM rec%':>9} {'TF rec%':>9}")
    for k in out:
        for snr in snrs:
            d = out[k][snr]
            denom = max(d["perfect"] - d["persist"], 1e-6)   # aging-loss recovery
            dr = 100 * (d.get("deepssm", d["persist"]) - d["persist"]) / denom
            tr = 100 * (d["tf"] - d["persist"]) / denom
            print(f"{k:>2} {snr:>4} {d['perfect']:>8.3f} {d['persist']:>8.3f} {d['wm']:>8.3f} "
                  f"{d.get('deepssm', float('nan')):>8.3f} {d['tf']:>8.3f}  {dr:>8.1f}% {tr:>8.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
