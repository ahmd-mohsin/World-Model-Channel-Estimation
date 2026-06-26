from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from implementation.config import SSWMConfig
from implementation.context_encoder import ContextEncoder
from implementation.target_encoder import TargetEncoder

assert torch.cuda.is_available(), "CUDA not available"
DEV = "cuda"
torch.manual_seed(0)


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def cfg(**kw):
    base = dict(n_subcarriers=32, n_antennas=32, seq_len=8, horizon_k=4,
                embed_dim=256, backbone="lwm", use_pretrained=True, ema_momentum=0.99)
    base.update(kw)
    return SSWMConfig(**base)


def obs(c, b=16):
    return torch.randn(b, c.seq_len, 2, c.n_subcarriers, c.n_antennas, device=DEV)


banner("ENV")
print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0), "| count", torch.cuda.device_count())

banner("1. ContextEncoder forward on GPU")
c = cfg()
ctx = ContextEncoder(c).to(DEV)
print("backbone pretrained:", ctx.backbone.is_pretrained)
o = obs(c)
x = ctx(o)
assert x.shape == (16, c.seq_len, c.embed_dim) and x.is_cuda
print(f"o {tuple(o.shape)} -> x {tuple(x.shape)} on {x.device}")

banner("2. Backward through frozen LWM on GPU (grads reach head, not backbone)")
ctx.zero_grad()
ctx(o).pow(2).mean().backward()
head_grad = any(p.grad is not None for p in ctx.head.parameters())
bb_grad = any(p.grad is not None for m in ctx.backbone.frozen_modules() for p in m.parameters())
print("head has grad:", head_grad, "| frozen backbone has grad:", bb_grad)
assert head_grad and not bb_grad

banner("3. TargetEncoder on GPU: forward detached + EMA update")
tgt = TargetEncoder(ctx, c).to(DEV)
z = tgt(o)
assert z.shape == x.shape and z.is_cuda and not z.requires_grad
print(f"z~ {tuple(z.shape)} on {z.device} | requires_grad={z.requires_grad}")
before = next(iter(tgt.encoder.head.parameters())).clone()
with torch.no_grad():
    next(iter(ctx.head.parameters())).add_(1.0)
tgt.ema_update(ctx)
after = next(iter(tgt.encoder.head.parameters()))
print("EMA moved target head param:", not torch.allclose(before, after))
assert not torch.allclose(before, after)

banner("4. Full JEPA coordination loop on GPU (no collapse)")
c2 = cfg()
ctx2 = ContextEncoder(c2).to(DEV).train()
tgt2 = TargetEncoder(ctx2, c2).to(DEV)
proxy = torch.randn(16, c2.seq_len, c2.embed_dim, device=DEV)
o2 = obs(c2)
opt = torch.optim.Adam(ctx2.trainable_parameters(), lr=5e-3)
first = last = None
for step in range(60):
    loss = (ctx2(o2) - proxy).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    tgt2.ema_update(ctx2)
    if step == 0: first = loss.item()
    last = loss.item()
with torch.no_grad():
    std = ctx2(o2).reshape(-1, c2.embed_dim).std(0).mean().item()
print(f"loss {first:.4f} -> {last:.4f} (down: {last < first}) | embedding std {std:.4f} (>0: no collapse)")
assert last < first and std > 1e-3

banner("5. Multi-GPU device placement sanity (8x A100)")
for g in range(torch.cuda.device_count()):
    d = f"cuda:{g}"
    e = ContextEncoder(cfg(seq_len=2, horizon_k=1)).to(d)
    oo = torch.randn(2, 2, 2, 32, 32, device=d)
    xx = e(oo)
    assert str(xx.device) == d
    print(f"  {d}: OK {tuple(xx.shape)}")
    del e, oo, xx
    torch.cuda.empty_cache()

banner("ALL GPU CHECKS PASSED")
print(f"peak mem cuda:0 = {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
