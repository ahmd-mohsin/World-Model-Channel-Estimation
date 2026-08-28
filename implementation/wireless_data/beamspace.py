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


# ------------------------------------------------------------------------------------------------
# MIMO separable beamspace (F1 fix): the spatial axis is a FOLDED N_tx*N_rx matrix. A plain 1-D DFT
# over the folded axis mixes departure and arrival angles and under-sparsifies. The separable
# transform does an INDEPENDENT angular DFT over the TX and RX antenna sub-axes (departure-angle and
# arrival-angle) plus the delay DFT over subcarriers -> a true 3-D angle-angle-delay representation
# that concentrates each path into far fewer taps. Still orthonormal (each 1-D DFT is unitary) so it
# is energy-preserving and exactly invertible. Reduces to the 2-D transform when n_rx==1.
# ------------------------------------------------------------------------------------------------
def to_beamspace_mimo(H: torch.Tensor, n_tx: int, n_rx: int) -> torch.Tensor:
    """Separable MIMO beamspace. H: (...,2,n_tx*n_rx,n_sub) -> (...,2,n_tx*n_rx,n_sub) beamspace.

    Unfolds the spatial axis into (n_tx, n_rx); IFFT over TX antennas (departure angle), IFFT over RX
    antennas (arrival angle), FFT over subcarriers (delay); refolds. Orthonormal (each DFT), exactly
    invertible via from_beamspace_mimo.
    """
    Hc = torch.complex(H[..., 0, :, :], H[..., 1, :, :])          # (..., n_tx*n_rx, n_sub)
    *lead, S, F = Hc.shape
    Hc = Hc.reshape(*lead, n_tx, n_rx, F)                          # unfold spatial -> (tx, rx)
    b = torch.fft.ifft(Hc, dim=-3, norm="ortho")                  # TX antenna -> departure angle
    b = torch.fft.ifft(b, dim=-2, norm="ortho")                   # RX antenna -> arrival angle
    b = torch.fft.fft(b, dim=-1, norm="ortho")                    # subcarrier -> delay
    b = b.reshape(*lead, S, F)                                    # refold
    return torch.stack([b.real, b.imag], dim=-3)


def from_beamspace_mimo(B: torch.Tensor, n_tx: int, n_rx: int) -> torch.Tensor:
    """Exact inverse of to_beamspace_mimo."""
    Bc = torch.complex(B[..., 0, :, :], B[..., 1, :, :])
    *lead, S, F = Bc.shape
    Bc = Bc.reshape(*lead, n_tx, n_rx, F)
    H = torch.fft.ifft(Bc, dim=-1, norm="ortho")                  # delay -> subcarrier
    H = torch.fft.fft(H, dim=-2, norm="ortho")                    # arrival angle -> RX antenna
    H = torch.fft.fft(H, dim=-3, norm="ortho")                    # departure angle -> TX antenna
    H = H.reshape(*lead, S, F)
    return torch.stack([H.real, H.imag], dim=-3)


def sparsity(H: torch.Tensor, frac: float = 0.9) -> float:
    """Fraction of taps needed to hold `frac` of the beamspace energy (lower = sparser)."""
    B = to_beamspace(H)
    mag2 = (B[..., 0, :, :] ** 2 + B[..., 1, :, :] ** 2).flatten(-2)   # (...,N_ant*N_sub)
    e = torch.sort(mag2, dim=-1, descending=True).values
    csum = e.cumsum(-1) / e.sum(-1, keepdim=True).clamp_min(1e-12)
    n_taps = (csum < frac).float().sum(-1) + 1
    return (n_taps / mag2.shape[-1]).mean().item()
