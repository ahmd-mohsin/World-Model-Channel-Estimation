from __future__ import annotations

import torch


def to_beamspace(H: torch.Tensor) -> torch.Tensor:
    """Antenna-subcarrier channel -> angle-delay (beamspace) domain via 2D FFT.

    Input  H : (..., 2, N_ant, N_sub)  real/imag planes of the complex channel.
    Output B : (..., 2, N_ant, N_sub)  real/imag of the 2D-DFT (orthonormal), where propagation
    paths concentrate into a few (angle, delay) taps -> the representation is SPARSE.

    IFFT over antennas -> angle (beam) axis; FFT over subcarriers -> delay axis. Orthonormal norm
    so the transform is energy-preserving and exactly invertible (Parseval), enabling a faithful
    reconstruction objective.
    """
    Hc = torch.complex(H[..., 0, :, :], H[..., 1, :, :])          # (...,N_ant,N_sub)
    beam = torch.fft.ifft(Hc, dim=-2, norm="ortho")               # antenna -> angle
    beam = torch.fft.fft(beam, dim=-1, norm="ortho")              # subcarrier -> delay
    return torch.stack([beam.real, beam.imag], dim=-3)


def from_beamspace(B: torch.Tensor) -> torch.Tensor:
    """Inverse of to_beamspace: angle-delay -> antenna-subcarrier. Exact (orthonormal 2D DFT)."""
    Bc = torch.complex(B[..., 0, :, :], B[..., 1, :, :])
    H = torch.fft.ifft(Bc, dim=-1, norm="ortho")                  # delay -> subcarrier
    H = torch.fft.fft(H, dim=-2, norm="ortho")                    # angle -> antenna
    return torch.stack([H.real, H.imag], dim=-3)


def sparsity(H: torch.Tensor, frac: float = 0.9) -> float:
    """Fraction of taps needed to hold `frac` of the beamspace energy (lower = sparser)."""
    B = to_beamspace(H)
    mag2 = (B[..., 0, :, :] ** 2 + B[..., 1, :, :] ** 2).flatten(-2)   # (...,N_ant*N_sub)
    e = torch.sort(mag2, dim=-1, descending=True).values
    csum = e.cumsum(-1) / e.sum(-1, keepdim=True).clamp_min(1e-12)
    n_taps = (csum < frac).float().sum(-1) + 1
    return (n_taps / mag2.shape[-1]).mean().item()
