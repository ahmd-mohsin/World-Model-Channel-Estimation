"""Cross-mobility generalization: load a trained predictor checkpoint and evaluate prediction NMSE on a
(possibly different-mobility) test set. Tests whether action-conditioned dynamics transfer across speed
regimes --- a core world-model claim.

    python scripts/xeval-predictor.py --ckpt implementation/checkpoints/ds3_slow.pt \
        --backbone deepssm --n_layers 3 --data data/mimo_fast --n_ant 32 --n_sub 32 --tag ds3_slow_on_fast
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.task_heads.temporal_baseline import TemporalPredictor
from implementation.task_heads.ssm_predictor import DeepSSMPredictor
from implementation.wireless_data import ShardDataset
from implementation.wireless_data.beamspace import to_beamspace, from_beamspace


def nmse(p, t): return (F.mse_loss(p, t) / t.pow(2).mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--backbone", required=True)
    ap.add_argument("--n_layers", type=int, default=3); ap.add_argument("--data", required=True)
    ap.add_argument("--n_ant", type=int, default=32); ap.add_argument("--n_sub", type=int, default=32)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SSWMConfig(n_subcarriers=args.n_sub, n_antennas=args.n_ant, seq_len=8, horizon_k=3,
                     action_dim=4, embed_dim=128, state_dim=64, latent_dim=128, use_pretrained=False,
                     unet_base_ch=48)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if args.backbone == "deepssm":
        m = DeepSSMPredictor(cfg, n_layers=args.n_layers).to(dev)
    else:
        m = TemporalPredictor(cfg, backbone=args.backbone).to(dev)
    m.load_state_dict(ck["model"], strict=False); m.eval()
    ds = ShardDataset(args.data, cfg, test_frac=0.05, seed=0)
    o, a = ds.all("test", device=dev); o, a = o[:2000], a[:2000]
    beam = to_beamspace(o); T = o.shape[1]
    pred = {}
    print(f"\n[{args.tag}]  k   persist   pred", flush=True)
    with torch.no_grad():
        for k in range(1, min(6, T - 1) + 1):
            anchor = T - 1 - k
            bp = m.predict(beam[:, :anchor + 1], a[:, :anchor + 1], a[:, anchor:anchor + k])
            Hp = from_beamspace(bp); Ht = o[:, anchor + k]; Hpers = o[:, anchor]
            pred[k] = {"persist": nmse(Hpers, Ht), "pred": nmse(Hp, Ht)}
            print(f"       {k:2d}  {pred[k]['persist']:7.4f}  {pred[k]['pred']:7.4f}", flush=True)
    Path("results/xmobility").mkdir(parents=True, exist_ok=True)
    Path(f"results/xmobility/pred_{args.tag}.json").write_text(json.dumps(pred))
    print(f"wrote results/xmobility/pred_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
