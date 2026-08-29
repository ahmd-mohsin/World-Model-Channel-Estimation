"""Downstream metric (#1): achievable spectral efficiency (bits/s/Hz) with MIMO SVD precoding designed
on the ESTIMATED channel and evaluated on the TRUE channel — the communication payoff of lower NMSE.

Per subcarrier the folded 32-spatial vector unfolds to an N_t x N_r matrix; we treat it as the MIMO
channel H (N_r x N_t). We compare estimators {perfect-CSI (upper bound), World Model, MMSE, LS}:
  - design precoder V = right singular vectors of the ESTIMATE Hhat (top r=min(Nt,Nr) streams)
  - receive combiner G = left singular vectors of Hhat
  - effective channel on the TRUE H: E = G^H H V  (r x r); off-diagonal = inter-stream interference
  - equal-power sum rate = sum_k log2(1 + |E_kk|^2 * p / (p*sum_{j!=k}|E_kj|^2 + 1)),  p = SNR/r
Reports mean rate per SNR and % of perfect-CSI rate retained. Mismatched precoding penalizes CSI error
in the way that actually costs throughput. Run on the box (has checkpoint + data/mimo).

    python scripts/spectral-efficiency.py --ckpt implementation/checkpoints/beam_indist_s0.pt \
        --data data/mimo --n_ant 32 --n_sub 32 --n_tx 8 --n_rx 4 --out results/spectral_eff.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel
from implementation.wireless_data import ShardDataset
from implementation.task_heads.baselines import add_noise, mmse_estimate


def mismatched_rate(H_true, H_est, snr_lin, n_tx, n_rx):
    """H_*: (B, 2, n_tx*n_rx, n_sub) real/imag. Returns mean sum-rate (bits/s/Hz) over batch,subcarriers."""
    def to_mimo(H):
        c = (H[:, 0] + 1j * H[:, 1])                       # (B, S, F)
        B, S, F = c.shape
        c = c.reshape(B, n_tx, n_rx, F).permute(0, 3, 2, 1)  # (B,F, n_rx, n_tx) = H_rx_tx per subcarrier
        return c
    Ht = to_mimo(H_true); He = to_mimo(H_est)              # (B,F,nr,nt)
    r = min(n_tx, n_rx)
    p = snr_lin / r
    # SVD of estimate: He = U S Vh ; precoder V = Vh^H[:, :r], combiner G = U[:, :r]
    Ue, Se, Vhe = torch.linalg.svd(He)                     # Ue:(...,nr,nr) Vhe:(...,nt,nt)
    V = Vhe.conj().transpose(-1, -2)[..., :, :r]           # (...,nt,r)
    G = Ue[..., :, :r]                                     # (...,nr,r)
    E = G.conj().transpose(-1, -2) @ Ht @ V                # (...,r,r) effective true channel
    gain = (E.abs() ** 2)                                  # |E_kj|^2
    diag = torch.diagonal(gain, dim1=-2, dim2=-1)          # (...,r) signal
    interf = gain.sum(-1) - diag                           # (...,r) leakage
    sinr = (p * diag) / (p * interf + 1.0)
    rate = torch.log2(1 + sinr).sum(-1)                    # sum over streams -> (B,F)
    return rate.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_ant", type=int, default=32); ap.add_argument("--n_sub", type=int, default=32)
    ap.add_argument("--n_tx", type=int, default=8); ap.add_argument("--n_rx", type=int, default=4)
    ap.add_argument("--out", default="results/spectral_eff.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = BeamWorldModel(cfg).to(dev); m.load_state_dict(ck["model"], strict=False); m.eval()
    ds = ShardDataset(args.data, cfg, test_frac=0.05, seed=0)
    o, a = ds.all("test", device=dev); o, a = o[:1500], a[:1500]
    Htr = ds.all("train", device=dev)[0][:3000, -1]
    Hte = o[:, -1]
    g = torch.Generator(device=dev).manual_seed(0)

    out = {}
    print(f"{'SNR':>4}  {'perfect':>8} {'WorldModel':>10} {'MMSE':>8} {'LS':>8}   WM %opt")
    for snr in [-5, 0, 5, 10, 15, 20, 30]:
        snr_lin = 10 ** (snr / 10)
        with torch.no_grad():
            Hwm, _ = m.estimate_channel(o, a, snr_db=float(snr), noise_gen=g)
        Y = add_noise(Hte, snr, generator=g)
        Hmm = mmse_estimate(Y, Htr, snr)
        r_opt = mismatched_rate(Hte, Hte, snr_lin, args.n_tx, args.n_rx)   # perfect CSI upper bound
        r_wm  = mismatched_rate(Hte, Hwm,  snr_lin, args.n_tx, args.n_rx)
        r_mm  = mismatched_rate(Hte, Hmm,  snr_lin, args.n_tx, args.n_rx)
        r_ls  = mismatched_rate(Hte, Y,    snr_lin, args.n_tx, args.n_rx)
        out[snr] = {"perfect": r_opt, "wm": r_wm, "mmse": r_mm, "ls": r_ls,
                    "wm_pct_opt": 100 * r_wm / r_opt}
        print(f"{snr:>4}  {r_opt:>8.3f} {r_wm:>10.3f} {r_mm:>8.3f} {r_ls:>8.3f}   {100*r_wm/r_opt:>5.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
