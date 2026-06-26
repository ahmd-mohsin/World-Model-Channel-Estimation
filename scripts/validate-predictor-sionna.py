from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.wireless_data import WirelessDataset, SionnaSpec

dev = "cuda" if torch.cuda.is_available() else "cpu"


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def make_actions(o):
    # Action = normalized per-step displacement proxy: the change in mean channel power.
    # (Stand-in for a real Tx/Rx control; gives the model an informative, non-trivial action.)
    b, t = o.shape[:2]
    power = o.pow(2).mean(dim=(2, 3, 4))                  # (B,T)
    dp = torch.zeros_like(power)
    dp[:, 1:] = power[:, 1:] - power[:, :-1]
    base = torch.stack([power, dp], dim=-1)              # (B,T,2)
    pad = torch.zeros(b, t, cfg.action_dim - 2, device=o.device)
    return torch.cat([base, pad], dim=-1)


cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=8, horizon_k=3,
                 embed_dim=256, action_dim=8, state_dim=64, latent_dim=256,
                 backbone="lwm", use_pretrained=True)

banner("Generate real Sionna channel sequences (temporally correlated)")
ds = WirelessDataset(cfg, n_samples=256, spec=SionnaSpec(scene="munich"), seed=0)
pool = ds.batch(256).to(dev)
print(f"pool {tuple(pool.shape)} on {pool.device}")

m = SSWM(cfg).to(dev)
m.context_encoder.train()
opt = torch.optim.Adam(m.trainable_parameters(), lr=1e-3)
rng = np.random.default_rng(0)

banner("Train SSWM JEPA objective on real channels")
for step in range(600):
    idx = rng.choice(pool.shape[0], size=32, replace=False)
    o = pool[idx]
    a = make_actions(o)
    loss, metrics = m.jepa_loss(o, a)
    opt.zero_grad(); loss.backward(); opt.step()
    m.update_target()
    if step % 100 == 0 or step == 599:
        print(f"step {step:4d} | loss {metrics['loss']:.4f} | pred_std {metrics['pred_std']:.4f} "
              f"| target_std {metrics['target_std']:.4f}")

banner("Cross-check: does the predictor beat trivial baselines on REAL dynamics?")
m.eval()
with torch.no_grad():
    o = pool[:128]
    a = make_actions(o)
    anchor = cfg.seq_len - 1 - cfg.horizon_k
    z = m.encode_sequence(o, a)
    z_t = z[:, anchor]
    z_hat = m.predictor(z_t, a[:, anchor:anchor + cfg.horizon_k])
    z_tilde = m.target_encoder(o[:, anchor + cfg.horizon_k].unsqueeze(1))[:, 0]

    # Baselines in the SAME embed space:
    #  - persistence: predict the PRESENT target embedding (encode o_t)
    #  - mean: predict the batch mean future embedding
    z_present = m.target_encoder(o[:, anchor].unsqueeze(1))[:, 0]
    z_mean = z_tilde.mean(0, keepdim=True).expand_as(z_tilde)

    def nmse(p):
        return (F.mse_loss(p, z_tilde) / z_tilde.pow(2).mean()).item()

    print(f"predictor      NMSE: {nmse(z_hat):.4f}")
    print(f"persistence    NMSE: {nmse(z_present):.4f}   (predict present, ignore dynamics)")
    print(f"batch-mean     NMSE: {nmse(z_mean):.4f}   (predict mean, ignore input)")
    verdict = "PREDICTOR BEATS PERSISTENCE" if nmse(z_hat) < nmse(z_present) else "no gain over persistence"
    print(f"\nverdict: {verdict}")

banner("DONE")
