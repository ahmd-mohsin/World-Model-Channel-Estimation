from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.sswm import SSWM
from implementation.wireless_data import ShardDataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("implementation/checkpoints/sswm_ddp.pt", map_location="cpu")
cfg = SSWMConfig(**{k: v for k, v in ck["config"].items() if k in SSWMConfig.__dataclass_fields__})
m = SSWM(cfg).to(dev)
m.load_state_dict(ck["model"]); m.eval()
ds = ShardDataset("data/act60k", cfg, test_frac=0.05, seed=0)
anchor = cfg.seq_len - 1 - cfg.horizon_k

with torch.no_grad():
    o, a = ds.all("test", device=dev)
    zh, zt, zp = [], [], []
    for i in range(0, o.shape[0], 256):
        h, t = m(o[i:i+256], a[i:i+256])
        zh.append(h); zt.append(t)
        zp.append(m.target_encoder(o[i:i+256, anchor].unsqueeze(1))[:, 0])
    zh, zt, zp = torch.cat(zh), torch.cat(zt), torch.cat(zp)

def nmse(p): return (F.mse_loss(p, zt) / zt.pow(2).mean()).item()
def cos(p): return F.cosine_similarity(p, zt, dim=-1).mean().item()

print("=== metric: NMSE (lower better) ===")
print(f"predictor {nmse(zh):.4f} | persistence {nmse(zp):.4f}")
print("=== metric: cosine similarity to future (higher better) ===")
print(f"predictor {cos(zh):.4f} | persistence {cos(zp):.4f}")
print(f"\npred_std {zh.std(0).mean():.3f} | target_std {zt.std(0).mean():.3f} | persist_std {zp.std(0).mean():.3f}")
