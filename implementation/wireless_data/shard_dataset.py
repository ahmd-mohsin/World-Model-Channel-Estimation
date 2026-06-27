from __future__ import annotations

import glob

import torch

try:
    from ..config import SSWMConfig
except ImportError:
    from config import SSWMConfig


class ShardDataset:
    """Loads pre-generated Sionna shards (data + velocity action) for scaled training.

    Each shard (from scripts/gen_sionna_actions.py) holds:
      data   : (N, T, 2, n_ant, n_sub)   channels, already x1e6 scaled
      action : (N, T, action_dim)        per-step velocity action [vx, vy, speed, theta]

    Applies per-channel standardization computed on the TRAIN split so the ~15x cross-scene
    magnitude spread does not dominate (the single biggest correction the GPU runs revealed).
    Actions are standardized too. Exposes a deterministic train/test split.
    """

    def __init__(self, data_dir: str, config: SSWMConfig, test_frac: float = 0.1, seed: int = 0):
        files = sorted(glob.glob(f"{data_dir}/shard_*.pt"))
        if not files:
            raise FileNotFoundError(f"no shards in {data_dir}")
        shards = [torch.load(f, map_location="cpu") for f in files]
        self.data = torch.cat([s["data"] for s in shards], 0)        # (N,T,2,ant,sub)
        self.action = torch.cat([s["action"] for s in shards], 0).float()
        self.scenes = [s.get("scene", "?") for s in shards]
        self.config = config

        n = self.data.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_test = max(128, int(n * test_frac))
        self.test_idx = perm[:n_test]
        self.train_idx = perm[n_test:]

        # standardization stats from TRAIN only (per real/imag plane, broadcast over ant/sub)
        tr = self.data[self.train_idx]
        self.chan_mu = tr.mean(dim=(0, 1, 3, 4), keepdim=True)        # (1,1,2,1,1)
        self.chan_sd = tr.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-6
        tra = self.action[self.train_idx]
        self.act_mu = tra.mean(dim=(0, 1), keepdim=True)
        self.act_sd = tra.std(dim=(0, 1), keepdim=True) + 1e-6

    def _norm(self, o, a):
        return (o - self.chan_mu) / self.chan_sd, (a - self.act_mu) / self.act_sd

    def batch(self, batch_size: int, split: str = "train", rng=None, device="cpu"):
        idx_pool = self.train_idx if split == "train" else self.test_idx
        if rng is None:
            sel = idx_pool[torch.randperm(len(idx_pool))[:batch_size]]
        else:
            import numpy as np
            sel = idx_pool[torch.from_numpy(rng.choice(len(idx_pool), batch_size, replace=False))]
        o, a = self._norm(self.data[sel], self.action[sel])
        return o.to(device), a.to(device)

    def all(self, split: str = "test", device="cpu"):
        idx = self.test_idx if split == "test" else self.train_idx
        o, a = self._norm(self.data[idx], self.action[idx])
        return o.to(device), a.to(device)
