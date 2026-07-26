"""Label-efficiency sweep: does the SSL-pretrained representation help when LABELS ARE SCARCE?

The full-data control showed supervised ties SSWM — expected, because SSL's advantage is DATA
EFFICIENCY, not asymptotic accuracy. Here both models get only N labeled (y,H) pairs:
  - SSWM (SSL):   FREEZE the pretrained encoder+SSM; train a small channel head on N labels
                  (linear/conv probe on the frozen SSL latent + raw obs).
  - supervised:   train the SAME U-Net from scratch on the same N labels (no pretraining).
Hypothesis: SSL wins at small N, gap closes as N -> full.

    python scripts/label-efficiency.py --data_dir data/act60k --ckpt <sswm ckpt>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.task_heads.unet_head import UNetChannelHead
from implementation.task_heads.baselines import add_noise, nmse
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_noisy(H, g):
    snr = torch.empty(H.shape[0], 1, 1, 1, device=dev).uniform_(0, 20)
    nv = H.pow(2).mean(dim=(1, 2, 3), keepdim=True) / (10 ** (snr / 10))
    Y = H + torch.randn(H.shape, generator=g, device=dev) * nv.sqrt()
    return Y, nv.reshape(-1)


def eval_head(fn, Hte, g):
    """fn(Y, nv)->est. Report avg NMSE over an SNR sweep."""
    out = {}
    for snr in [0, 10, 20]:
        Yte = add_noise(Hte, snr, generator=g)
        nv = Hte.pow(2).mean(dim=(1, 2, 3)) / (10 ** (snr / 10))
        with torch.no_grad():
            out[snr] = nmse(fn(Yte, nv), Hte)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", default="implementation/checkpoints/sswm_e2e_multihorizon.pt")
    ap.add_argument("--Ns", type=int, nargs="+", default=[100, 300, 1000, 3000, 10000])
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
    m = SSWM(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)                          # FREEZE the SSL model
    ds = ShardDataset(args.data_dir, cfg, test_frac=0.1, seed=0)
    t = cfg.seq_len - 1
    Htr_all = ds.all("train", device=dev)[0]
    Hte = ds.all("test", device=dev)[0][:, t]
    atr_all = ds.all("train", device=dev)[1]
    g = torch.Generator(device=dev).manual_seed(0)
    rng = np.random.default_rng(0)

    def sswm_latent(Yframe, seq_o, seq_a):
        s = seq_o.clone(); s[:, t] = Yframe
        return m.encode_sequence(s, seq_a)[:, t]

    results = {}
    print(f"{'N':>6} | {'SSL@0':>7} {'Sup@0':>7} | {'SSL@10':>7} {'Sup@10':>7} | {'SSL@20':>7} {'Sup@20':>7}")
    print("-" * 72)
    for N in args.Ns:
        idx = torch.from_numpy(rng.choice(Htr_all.shape[0], N, replace=False)).to(dev)
        Otr, Atr = Htr_all[idx], atr_all[idx]
        Htr = Otr[:, t]

        # --- SSL: small conv head on FROZEN latent + obs (encoder frozen) ---
        ssl_head = UNetChannelHead(cfg.latent_dim, cfg.n_antennas, cfg.n_subcarriers, base=cfg.unet_base_ch).to(dev)
        opt = torch.optim.Adam(ssl_head.parameters(), lr=1e-3)
        for _ in range(args.steps):
            b = torch.from_numpy(rng.choice(N, min(128, N), replace=False)).to(dev)
            Y, nv = make_noisy(Htr[b], g)
            with torch.no_grad():
                z = sswm_latent(Y, Otr[b], Atr[b])
            est = ssl_head(Y, z, nv)
            loss = F.mse_loss(est, Htr[b]); opt.zero_grad(); loss.backward(); opt.step()

        # --- supervised: same U-Net from scratch, zero latent ---
        sup = UNetChannelHead(cfg.latent_dim, cfg.n_antennas, cfg.n_subcarriers, base=cfg.unet_base_ch).to(dev)
        opt = torch.optim.Adam(sup.parameters(), lr=1e-3)
        z0 = torch.zeros(min(128, N), cfg.latent_dim, device=dev)
        for _ in range(args.steps):
            b = torch.from_numpy(rng.choice(N, min(128, N), replace=False)).to(dev)
            Y, nv = make_noisy(Htr[b], g)
            est = sup(Y, z0[:Y.shape[0]], nv)
            loss = F.mse_loss(est, Htr[b]); opt.zero_grad(); loss.backward(); opt.step()

        # eval both (SSL needs a test sequence context; use test split's own frames)
        Ote, Ate = ds.all("test", device=dev)[0], ds.all("test", device=dev)[1]
        def ssl_fn(Y, nv):
            z = sswm_latent(Y, Ote, Ate); return ssl_head(Y, z, nv)
        def sup_fn(Y, nv):
            return sup(Y, torch.zeros(Y.shape[0], cfg.latent_dim, device=dev), nv)
        e_ssl = eval_head(ssl_fn, Hte, g); e_sup = eval_head(sup_fn, Hte, g)
        results[N] = {"ssl": e_ssl, "sup": e_sup}
        print(f"{N:>6} | {e_ssl[0]:>7.4f} {e_sup[0]:>7.4f} | {e_ssl[10]:>7.4f} {e_sup[10]:>7.4f} | {e_ssl[20]:>7.4f} {e_sup[20]:>7.4f}", flush=True)

    Path("dashboard").mkdir(exist_ok=True)
    Path("dashboard/label_efficiency.json").write_text(json.dumps(results))
    print("\nsaved dashboard/label_efficiency.json")


if __name__ == "__main__":
    main()
