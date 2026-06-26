from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder


def synth_channel(batch: int, cfg: SSWMConfig, n_paths: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ant = torch.arange(cfg.n_antennas).view(1, -1, 1, 1)
    sub = torch.arange(cfg.n_subcarriers).view(1, 1, -1, 1)
    gain = torch.randn(batch, 1, 1, n_paths, generator=g)
    aod = torch.rand(batch, 1, 1, n_paths, generator=g)
    delay = torch.rand(batch, 1, 1, n_paths, generator=g)
    phase = 2 * torch.pi * (ant * aod / cfg.n_antennas + sub * delay / cfg.n_subcarriers)
    real = (gain * torch.cos(phase)).sum(-1)
    imag = (gain * torch.sin(phase)).sum(-1)
    H = torch.stack([real, imag], dim=1)
    H = H / H.flatten(1).abs().amax(dim=1).view(-1, 1, 1, 1)
    return H.unsqueeze(1).expand(batch, cfg.seq_len, 2, cfg.n_antennas, cfg.n_subcarriers).contiguous()


def main() -> None:
    cfg = SSWMConfig(
        n_subcarriers=32,
        n_antennas=32,
        seq_len=8,
        horizon_k=4,
        embed_dim=256,
        backbone="lwm",
        use_pretrained=True,
    )
    enc = ContextEncoder(cfg).eval()

    print("=" * 70)
    print("SSWM ContextEncoder  ::  o_t  ->  x_t   (figure: 'context encoder')")
    print("=" * 70)
    print(f"backbone           : {cfg.backbone}  (wi-lab/lwm-v1.1, DeepMIMO-pretrained)")
    print(f"pretrained loaded  : {enc.backbone.is_pretrained}")
    print(f"backbone hidden    : {enc.backbone.hidden}")
    print(f"embed_dim (x_t)    : {cfg.embed_dim}")

    n_train = sum(p.numel() for p in enc.trainable_parameters())
    n_total = sum(p.numel() for p in enc.parameters())
    print(f"trainable params   : {n_train:,} / {n_total:,}  (rest = frozen LWM)")

    o = synth_channel(batch=2, cfg=cfg)
    print(f"\nobservations o_t   : {tuple(o.shape)}   (B, T, [real/imag], N_ant, N_sub)")

    with torch.no_grad():
        x = enc(o)
    print(f"embedding x_t      : {tuple(x.shape)}   (B, T, embed_dim)  --> feeds the SSM block")

    print("\n--- what x_t looks like (projection head is untrained random init) ---")
    print(f"dtype / device     : {x.dtype} / {x.device}")
    print(f"mean / std         : {x.mean().item():+.4f} / {x.std().item():.4f}")
    print(f"min / max          : {x.min().item():+.4f} / {x.max().item():+.4f}")
    print(f"x_t[0, 0, :8]      : {x[0, 0, :8].numpy().round(4)}")

    with torch.no_grad():
        a = synth_channel(batch=1, cfg=cfg, n_paths=3, seed=1)
        b = synth_channel(batch=1, cfg=cfg, n_paths=20, seed=7)
        emb_a = enc.backbone(a[:, 0])[0, 1:]
        emb_a_noisy = enc.backbone((a[:, 0] + 0.02 * torch.randn_like(a[:, 0])))[0, 1:]
        emb_b = enc.backbone(b[:, 0])[0, 1:]
    d_self = (emb_a - emb_a_noisy).pow(2).mean().sqrt().item()
    d_cross = (emb_a - emb_b).pow(2).mean().sqrt().item()
    print("\n--- sanity on PRETRAINED LWM channel embeddings (per-patch tokens) ---")
    print(f"RMS dist(A vs A+noise)  : {d_self:.4f}  (small: same channel maps close)")
    print(f"RMS dist(3-path vs 20-path): {d_cross:.4f}  (larger: distinct channels separate)")
    print(f"separation ratio        : {d_cross / max(d_self, 1e-9):.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
