from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from ..config import SSWMConfig
except ImportError:
    from config import SSWMConfig


@dataclass
class SionnaSpec:
    scene: str = "munich"
    frequency: float = 3.5e9
    subcarrier_spacing: float = 30e3
    max_depth: int = 3
    rx_height: float = 1.5
    # Per-step displacement along the trajectory, in metres. Kept sub-wavelength
    # (lambda @ 3.5 GHz ~= 8.6 cm) so consecutive channels stay temporally correlated
    # -- the smooth evolution a world model is meant to predict.
    step_size_m: float = 0.015
    tx_position: tuple = (8.5, 21.0, 27.0)
    rx_start: tuple = (45.0, 90.0, 1.5)
    seed: int = 0
    # LWM was trained on raw DeepMIMO channels scaled by 1e6 (deepmimo_data_cleaning:
    # `channel * 1e6`), which preserves the natural cross-antenna/subcarrier amplitude
    # variation LWM relies on. We mirror that: use Sionna's RAW (un-normalized) CFR and
    # apply a fixed global scale, rather than per-sample max-normalization (which destroys
    # absolute scale and was measured to make LWM features ~32x less discriminative).
    channel_scale: float = 1e6


class SionnaChannelGenerator:
    """Generates temporally-correlated MIMO-OFDM channel sequences via Sionna RT.

    A receiver is moved along a straight trajectory; at each step the ray-traced
    channel frequency response is computed. Output sequences are formatted as the
    pipeline's observation tensor: (T, 2, n_antennas, n_subcarriers).
    """

    def __init__(self, config: SSWMConfig, spec: SionnaSpec | None = None) -> None:
        self.config = config
        self.spec = spec or SionnaSpec()
        self._scene = None
        self._solver = None
        self._freqs_dr = None

    def _lazy_init(self):
        if self._scene is not None:
            return
        import drjit as dr
        import sionna.rt as rt

        spec, cfg = self.spec, self.config
        scene = rt.load_scene(getattr(rt.scene, spec.scene))
        scene.frequency = spec.frequency
        scene.tx_array = rt.PlanarArray(
            num_rows=1, num_cols=cfg.n_antennas, vertical_spacing=0.5,
            horizontal_spacing=0.5, pattern="iso", polarization="V",
        )
        scene.rx_array = rt.PlanarArray(
            num_rows=1, num_cols=1, vertical_spacing=0.5,
            horizontal_spacing=0.5, pattern="iso", polarization="V",
        )
        scene.add(rt.Transmitter(name="tx", position=list(spec.tx_position)))
        scene.add(rt.Receiver(name="rx", position=list(spec.rx_start)))

        self._rt = rt
        self._dr = dr
        self._scene = scene
        self._solver = rt.PathSolver()

        n = cfg.n_subcarriers
        freqs = (np.arange(n) - n // 2) * spec.subcarrier_spacing
        self._freqs_dr = dr.cuda.ad.Float(freqs.astype(np.float32))

    def _channel_at(self, position) -> torch.Tensor:
        rx = self._scene.get("rx")
        rx.position = np.asarray(position, dtype=np.float32)
        paths = self._solver(self._scene, max_depth=self.spec.max_depth)
        H = paths.cfr(frequencies=self._freqs_dr, normalize=False, out_type="torch")
        H = H.reshape(-1, self.config.n_subcarriers)
        H = H[: self.config.n_antennas]
        if H.shape[0] < self.config.n_antennas:
            pad = self.config.n_antennas - H.shape[0]
            H = torch.cat([H, torch.zeros(pad, self.config.n_subcarriers, dtype=H.dtype, device=H.device)], 0)
        return H

    def generate_sequence(self, direction=None, rng: np.random.Generator | None = None) -> torch.Tensor:
        self._lazy_init()
        cfg, spec = self.config, self.spec
        rng = rng or np.random.default_rng(spec.seed)
        if direction is None:
            theta = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        start = np.array(spec.rx_start, dtype=np.float64)

        frames = []
        for t in range(cfg.seq_len):
            pos = start + direction * spec.step_size_m * t
            pos[2] = spec.rx_height
            H = self._channel_at(pos).detach().cpu() * spec.channel_scale
            frames.append(torch.stack([H.real, H.imag], dim=0))
        return torch.stack(frames, dim=0)

    def generate_batch(self, batch: int, seed: int | None = None) -> torch.Tensor:
        rng = np.random.default_rng(self.spec.seed if seed is None else seed)
        seqs = [self.generate_sequence(rng=rng) for _ in range(batch)]
        return torch.stack(seqs, dim=0)
