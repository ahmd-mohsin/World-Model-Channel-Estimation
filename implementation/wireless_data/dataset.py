from __future__ import annotations

import numpy as np
import torch

try:
    from ..config import SSWMConfig
    from .sionna_channels import SionnaChannelGenerator, SionnaSpec
except ImportError:
    from config import SSWMConfig
    from sionna_channels import SionnaChannelGenerator, SionnaSpec


class WirelessDataset(torch.utils.data.Dataset):
    """Channel-sequence dataset for the SSWM pipeline (Sionna RT only).

    Returns observation tensors shaped (seq_len, 2, n_antennas, n_subcarriers),
    produced by ray-tracing a moving receiver through a Sionna RT scene. There is
    no synthetic fallback by design: every module is exercised on real ray-traced
    wireless channels. Requires `sionna` / `sionna-rt` to be installed.
    """

    def __init__(self, config: SSWMConfig, n_samples: int = 256,
                 spec: SionnaSpec | None = None, seed: int = 0) -> None:
        self.config = config
        self.n_samples = n_samples
        self.seed = seed
        self._gen = SionnaChannelGenerator(config, spec)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed + idx)
        return self._gen.generate_sequence(rng=rng)

    def batch(self, batch_size: int) -> torch.Tensor:
        return torch.stack([self[i] for i in range(batch_size)], dim=0)
