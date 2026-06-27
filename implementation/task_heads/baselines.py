from __future__ import annotations

import torch


def add_noise(H: torch.Tensor, snr_db: float, generator=None) -> torch.Tensor:
    """Add complex AWGN to a channel grid (B, 2, ant, sub) at the given SNR."""
    power = H.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    noise_p = power / (10 ** (snr_db / 10))
    noise = torch.randn(H.shape, generator=generator, device=H.device, dtype=H.dtype)
    return H + noise * noise_p.sqrt()


def ls_estimate(Y: torch.Tensor) -> torch.Tensor:
    """Least-squares channel estimate. With unit pilots, LS = the noisy observation itself."""
    return Y


def mmse_estimate(Y: torch.Tensor, H_clean_train: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Linear MMSE (Wiener) estimate using the spatial channel covariance from training data.

    Operates on the complex channel as a length-(ant*sub) vector per sample:
        Ĥ = R_hh (R_hh + σ² I)^{-1} Y
    R_hh is estimated from clean training channels; σ² from the SNR and signal power.
    """
    B = Y.shape[0]
    n_ant, n_sub = Y.shape[2], Y.shape[3]
    d = n_ant * n_sub

    def to_complex(x):
        return (x[:, 0] + 1j * x[:, 1]).reshape(x.shape[0], -1)

    def to_real(xc):
        xr = xc.reshape(-1, n_ant, n_sub)
        return torch.stack([xr.real, xr.imag], dim=1)

    Hc = to_complex(H_clean_train)                       # (N, d) complex
    R = (Hc.conj().t() @ Hc) / Hc.shape[0]               # (d, d) covariance
    sig_power = Hc.abs().pow(2).mean()
    noise_var = sig_power / (10 ** (snr_db / 10))
    I = torch.eye(d, dtype=R.dtype, device=R.device)
    W = R @ torch.linalg.inv(R + noise_var * I)          # Wiener filter (d, d)

    Yc = to_complex(Y)                                   # (B, d) complex
    Hhat = (W @ Yc.t()).t()                              # (B, d)
    return to_real(Hhat)


def nmse(est: torch.Tensor, target: torch.Tensor) -> float:
    return (torch.nn.functional.mse_loss(est, target) / target.pow(2).mean()).item()
