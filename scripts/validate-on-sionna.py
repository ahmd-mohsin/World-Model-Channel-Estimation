from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.target_encoder import TargetEncoder
from implementation.selection_net import SelectionNet
from implementation.wireless_data import WirelessDataset, SionnaSpec

dev = "cuda" if torch.cuda.is_available() else "cpu"


def banner(s):
    print("\n" + "=" * 68 + f"\n{s}\n" + "=" * 68)


cfg = SSWMConfig(n_subcarriers=32, n_antennas=8, seq_len=6, horizon_k=2,
                 embed_dim=256, action_dim=8, state_dim=64,
                 backbone="lwm", use_pretrained=True)

banner("Generate real Sionna RT channel sequences")
ds = WirelessDataset(cfg, n_samples=24, spec=SionnaSpec(scene="munich"), seed=0)
o = ds.batch(8).to(dev)
print(f"observations o_t (Sionna): {tuple(o.shape)} on {o.device}")
print(f"finite: {torch.isfinite(o).all().item()} | mean |H|: {o.abs().mean().item():.4f}")

banner("ContextEncoder on real channels")
enc = ContextEncoder(cfg).to(dev).eval()
with torch.no_grad():
    x = enc(o)
print(f"x_t: {tuple(x.shape)} | std across batch: {x.std(0).mean().item():.4f}")

banner("TargetEncoder (EMA) on real channels")
tgt = TargetEncoder(enc, cfg).to(dev)
with torch.no_grad():
    z = tgt(o)
print(f"z~: {tuple(z.shape)} | matches encoder at init: {torch.allclose(x, z, atol=1e-5)}")

banner("SelectionNet on real-channel-derived actions")
acts = torch.randn(8, cfg.seq_len, cfg.action_dim, device=dev)
sel = SelectionNet(cfg).to(dev).eval()
with torch.no_grad():
    A, B, C, dt = sel(acts)
print(f"A {tuple(A.shape)} all<0: {(A<0).all().item()} | dt all>0: {(dt>0).all().item()}")

banner("Feature quality: do LWM features separate distinct channels on REAL data?")
# Compare channels from genuinely DIFFERENT receiver locations (not sub-wavelength
# trajectory steps, which are near-identical by design).
from implementation.wireless_data import SionnaSpec as _Spec
gen = ds._gen


def _chan_at(pos):
    H = gen._channel_at(np.asarray(pos, dtype=np.float32)).detach().cpu() * gen.spec.channel_scale
    return torch.stack([H.real, H.imag], dim=0).unsqueeze(0).to(dev)


with torch.no_grad():
    A = _chan_at([45.0, 90.0, 1.5])
    B = _chan_at([60.0, 70.0, 1.5])
    fa = enc.backbone(A)[0, 1:].mean(0)
    fa2 = enc.backbone(A + 0.001 * torch.randn_like(A))[0, 1:].mean(0)
    fb = enc.backbone(B)[0, 1:].mean(0)
d_self = (fa - fa2).pow(2).mean().sqrt().item()
d_cross = (fa - fb).pow(2).mean().sqrt().item()
print(f"input A vs B (RMS): {(A - B).pow(2).mean().sqrt().item():.4f}")
print(f"RMS(A vs A+noise) : {d_self:.6f}")
print(f"RMS(loc A vs B)   : {d_cross:.6f}")
print(f"separation ratio  : {d_cross / max(d_self, 1e-9):.1f}x  (>1 => distinguishes real channels)")

banner("ALL SIONNA REAL-DATA CHECKS DONE")
