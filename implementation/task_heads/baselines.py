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


def comb_mask(n_sub: int, stride: int, device) -> torch.Tensor:
    """1-in-`stride` comb pilot mask over subcarriers: (1,1,1,n_sub), 1=pilot, 0=unobserved."""
    m = torch.zeros(n_sub, device=device)
    m[::stride] = 1.0
    return m.reshape(1, 1, 1, n_sub)


def pilot_interp(Y_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Linear interpolation across subcarriers from comb pilots (standard practical estimator).

    Y_masked : (B,2,A,S) noisy obs, unobserved subcarriers zeroed.
    mask     : (1,1,1,S) comb mask. Interpolates each (batch,plane,antenna) row over the subcarrier
    axis using the pilot positions; edges are held (clamp). Operates on real & imag planes.
    """
    b, c, a, s = Y_masked.shape
    idx = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]           # pilot subcarrier indices
    xp = idx.to(Y_masked.dtype)
    grid = torch.arange(s, device=Y_masked.device, dtype=Y_masked.dtype)
    # gather pilot values: (B,C,A,P)
    vals = Y_masked.index_select(-1, idx)
    # piecewise-linear interp per position along S (vectorized searchsorted)
    hi = torch.searchsorted(xp, grid.clamp(max=xp[-1]), right=False).clamp(1, len(xp) - 1)
    lo = hi - 1
    x0, x1 = xp[lo], xp[hi]
    w = ((grid - x0) / (x1 - x0).clamp(min=1e-6)).clamp(0, 1)         # (S,)
    v0 = vals.index_select(-1, lo)                                    # (B,C,A,S)
    v1 = vals.index_select(-1, hi)
    out = v0 + (v1 - v0) * w.reshape(1, 1, 1, s)
    return out


def masked_mmse(Y_masked: torch.Tensor, mask: torch.Tensor, H_clean_train: torch.Tensor,
                snr_db: float) -> torch.Tensor:
    """MMSE interpolation from comb pilots: estimate full channel from pilot observations using the
    frequency covariance learned on training channels.  Per antenna row h in C^S:
        ĥ = R_full,pilot (R_pilot,pilot + σ² I)^{-1} y_pilot
    R is the S×S subcarrier covariance (averaged over antennas) from clean training channels.
    """
    b, _, na, ns = Y_masked.shape
    idx = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    dev = Y_masked.device

    def cplx(x):  # (N,2,A,S) -> (N*A, S) complex
        return (x[:, 0] + 1j * x[:, 1]).reshape(-1, x.shape[-1])

    Htr = cplx(H_clean_train)                                        # (Ntr*A, S)
    R = (Htr.conj().t() @ Htr) / Htr.shape[0]                        # (S,S) freq covariance
    sig = Htr.abs().pow(2).mean()
    nvar = sig / (10 ** (snr_db / 10))
    Rpp = R[idx][:, idx]                                             # (P,P)
    Rfp = R[:, idx]                                                  # (S,P)
    W = Rfp @ torch.linalg.inv(Rpp + nvar * torch.eye(len(idx), dtype=R.dtype, device=dev))  # (S,P)
    Yc = (Y_masked[:, 0] + 1j * Y_masked[:, 1]).reshape(-1, ns)      # (B*A, S) complex
    yp = Yc.index_select(-1, idx)                                    # (B*A, P)
    Hhat = (W @ yp.t()).t().reshape(b, na, ns)                       # (B,A,S) complex
    return torch.stack([Hhat.real, Hhat.imag], dim=1)


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
