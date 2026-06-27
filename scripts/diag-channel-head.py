"""Diagnose WHY the SSWM channel head loses: is it the probe, or the lossy LWM latent?

Compares channel reconstruction from three inputs at 20 dB (near clean):
  (a) raw noisy observation (512-d)         -> upper bound a learned head can do
  (b) frozen LWM latent z (256-d)           -> what SSWM channel head actually uses
  (c) raw clean channel (oracle identity)   -> sanity (should be ~0)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.task_heads import add_noise, nmse
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3, embed_dim=256,
                 action_dim=4, state_dim=64, latent_dim=256, backbone="lwm", use_pretrained=True)
ds = ShardDataset("data/act", cfg, test_frac=0.1, seed=0)
tr_o, tr_a = ds.all("train", device=dev); te_o, te_a = ds.all("test", device=dev)
t = cfg.seq_len - 1
Htr, Hte = tr_o[:, t], te_o[:, t]
obs_dim = Htr[0].numel()
snr = 20.0
g = torch.Generator(device=dev).manual_seed(0)
Ytr, Yte = add_noise(Htr, snr, generator=g), add_noise(Hte, snr, generator=g)

m = SSWM(cfg).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
with torch.no_grad():
    s = tr_o.clone(); s[:, t] = Ytr; Ztr = m.encode_sequence(s, tr_a)[:, t]
    s = te_o.clone(); s[:, t] = Yte; Zte = m.encode_sequence(s, te_a)[:, t]

def probe(Xtr, Xte, name):
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(), nn.Linear(512, obs_dim)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tgt = Htr.reshape(Htr.shape[0], -1); rng = np.random.default_rng(0)
    for _ in range(3000):
        idx = torch.from_numpy(rng.choice(Xtr.shape[0], 256, replace=False)).to(dev)
        loss = F.mse_loss(net(Xtr[idx]), tgt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        est = net(Xte).reshape(Hte.shape)
    print(f"{name:32s} NMSE: {nmse(est, Hte):.4f}", flush=True)

print("Channel reconstruction at 20 dB (lower better):", flush=True)
probe(Yte.reshape(Yte.shape[0], -1)[ds.train_idx[:0]] if False else Ytr.reshape(Ytr.shape[0], -1),
      Yte.reshape(Yte.shape[0], -1), "(a) from raw noisy obs (512-d)")
probe(Ztr, Zte, "(b) from frozen LWM latent (256-d)")
print(f"{'LS (raw obs as-is)':32s} NMSE: {nmse(Yte, Hte):.4f}", flush=True)
