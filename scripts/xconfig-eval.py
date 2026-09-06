"""Cross-config generalization: a full-grid WM estimator trained on 8x4 (spatial-agnostic decoder),
evaluated ZERO-SHOT on 16x8 (128 folded spatial). Reports NMSE vs SNR on 16x8 against LS/MMSE, plus
the model's in-config 8x4 NMSE for reference --- how much does moving to an unseen antenna geometry cost?

    python scripts/xconfig-eval.py --ckpt implementation/checkpoints/beam_est_agn.pt \
        --data8 data/mimo --data16 data/mimo16x8 --out results/xconfig.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.beam_wm import BeamWorldModel
from implementation.wireless_data import ShardDataset
from implementation.task_heads.baselines import add_noise, mmse_estimate


def nmse(p, t): return (F.mse_loss(p, t) / t.pow(2).mean()).item()


def eval_cfg(m, data, n_ant, dev, g):
    cfg = SSWMConfig(**{k: v for k, v in m.config.__dict__.items() if k in SSWMConfig.__dataclass_fields__})
    cfg.n_antennas = n_ant
    ds = ShardDataset(data, cfg, test_frac=0.05, seed=0)
    o, a = ds.all("test", device=dev); o, a = o[:1000], a[:1000]
    Htr = ds.all("train", device=dev)[0][:2000, -1]; Hte = o[:, -1]
    m.decoder.n_ant = n_ant                       # spatial-agnostic decoder -> new grid
    out = {}
    for snr in [-5, 0, 5, 10, 15, 20, 30]:
        with torch.no_grad():
            Hhat, _ = m.estimate_channel(o, a, snr_db=float(snr), noise_gen=g)
        Y = add_noise(Hte, snr, generator=g)
        out[snr] = {"ls": nmse(Y, Hte), "mmse": nmse(mmse_estimate(Y, Htr, snr), Hte), "beam": nmse(Hhat, Hte)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data8", required=True)
    ap.add_argument("--data16", required=True); ap.add_argument("--out", default="results/xconfig.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = BeamWorldModel(cfg, agnostic_decoder=True).to(dev); m.load_state_dict(ck["model"], strict=False); m.eval()
    g = torch.Generator(device=dev).manual_seed(0)
    res = {"in_config_8x4": eval_cfg(m, args.data8, 32, dev, g),
           "zeroshot_16x8": eval_cfg(m, args.data16, 128, dev, g)}
    print(f"{'SNR':>4}  {'8x4 beam':>9} {'16x8 beam':>10} {'16x8 MMSE':>10} {'16x8 LS':>9}")
    for s in [-5, 0, 5, 10, 15, 20, 30]:
        a8 = res["in_config_8x4"][s]; a16 = res["zeroshot_16x8"][s]
        print(f"{s:>4}  {a8['beam']:>9.4f} {a16['beam']:>10.4f} {a16['mmse']:>10.4f} {a16['ls']:>9.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
