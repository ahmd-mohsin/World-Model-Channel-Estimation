"""Beamspace world model: predict the FUTURE CHANNEL in the sparse angle-delay basis, and estimate
the current channel by fusing that prediction with the noisy observation.

Design (JEPA/EMA removed; the world model predicts something FAITHFUL, not an invariant latent):

    o_t --2D FFT--> beam_t --enc--> x_t --selective SSM--> z_t
    (z_t, planned actions) --predictor--> future beam channel  b_hat_{t+k}   [reconstruction loss]
    estimate: b_hat_t (predicted-from-history prior) + noisy beam obs --Kalman gate--> beam_est_t
              --inv FFT--> H_hat_t                                          [channel loss]

Every module is load-bearing: encoder/SSM/predictor produce the prior (temporal info a single-shot
estimator cannot have); the gate fuses it with the observation. One sparse, faithful representation
serves both prediction and estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..config import SSWMConfig
    from ..selection_net import SelectionNet
    from ..selective_ssm import discretize
    from ..wireless_data.beamspace import to_beamspace, from_beamspace
except ImportError:
    from config import SSWMConfig
    from selection_net import SelectionNet
    from selective_ssm import discretize
    from wireless_data.beamspace import to_beamspace, from_beamspace


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class BeamEncoder(nn.Module):
    """Beamspace channel grid (B,2,A,S) -> latent x (B,d). Conv over sparse angle-delay taps."""

    def __init__(self, n_ant, n_sub, d, base=48):
        super().__init__()
        self.enc = nn.Sequential(_ConvBlock(2, base), nn.AvgPool2d(2), _ConvBlock(base, base * 2),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Linear(base * 2, d)

    def forward(self, beam):
        return self.proj(self.enc(beam))


class BeamDecoder(nn.Module):
    """latent (B,d) -> beamspace channel grid (B,2,A,S). Reconstructs sparse taps."""

    def __init__(self, n_ant, n_sub, d, base=48):
        super().__init__()
        self.n_ant, self.n_sub = n_ant, n_sub
        self.fc = nn.Linear(d, base * n_ant * n_sub)
        self.base = base
        self.net = nn.Sequential(_ConvBlock(base, base), nn.Conv2d(base, 2, 1))

    def forward(self, z):
        b = z.shape[0]
        h = self.fc(z).reshape(b, self.base, self.n_ant, self.n_sub)
        return self.net(h)


class BeamWorldModel(nn.Module):
    def __init__(self, config: SSWMConfig):
        super().__init__()
        self.config = config
        d, na, ns = config.latent_dim, config.n_antennas, config.n_subcarriers
        self.encoder = BeamEncoder(na, ns, d, base=config.unet_base_ch)
        self.selection = SelectionNet(config)
        # SSM input mixing on [x; a]
        self.in_proj = nn.Linear(d + config.action_dim, config.state_dim)
        self.D = nn.Parameter(torch.ones(config.state_dim))
        self.ssm_out = nn.Linear(config.state_dim, d)
        self.norm = nn.LayerNorm(config.state_dim)
        # predictor: rolls latent with planned actions, decodes to a FUTURE beam channel
        self.pred_in = nn.Linear(config.action_dim, config.state_dim)
        self.z_to_h = nn.Linear(d, config.state_dim)
        self.decoder = BeamDecoder(na, ns, d, base=config.unet_base_ch)
        self.pred_to_z = nn.Linear(config.state_dim, d)
        # estimation fusion: learned Kalman gate on [obs_beam, prior_beam, noise]
        self.gate = nn.Sequential(nn.Conv2d(5, config.unet_base_ch, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(config.unet_base_ch, 2, 3, padding=1), nn.Sigmoid())
        self.refine = nn.Sequential(_ConvBlock(5, config.unet_base_ch), nn.Conv2d(config.unet_base_ch, 2, 1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    # ---- encode a beam sequence to latents + SSM states ----
    def encode(self, beam_seq, a):
        b, t = beam_seq.shape[:2]
        x = self.encoder(beam_seq.reshape(b * t, *beam_seq.shape[2:])).reshape(b, t, -1)
        A, B, C, dt = self.selection(a)
        u = self.in_proj(torch.cat([x, a], dim=-1))
        dA, dB = discretize(A, B, dt)
        h = torch.zeros(b, self.config.state_dim, device=x.device, dtype=x.dtype)
        zs, hs = [], []
        for i in range(t):
            h = dA[:, i] * h + dB[:, i] * u[:, i]
            y = C[:, i] * h + self.D * u[:, i]
            zs.append(self.ssm_out(self.norm(y))); hs.append(h)
        return x, torch.stack(zs, 1), torch.stack(hs, 1)

    def predict_beam(self, z_t, h_t, planned_acts):
        """Roll k steps from (z_t, SSM state h_t) with planned actions -> future beam channel."""
        h = h_t if h_t is not None else self.z_to_h(z_t)
        for j in range(planned_acts.shape[1]):
            a_j = planned_acts[:, j]
            A, B, C, dt = self.selection(a_j)
            u = self.pred_in(a_j)
            dA, dB = discretize(A, B, dt)
            h = dA * h + dB * u
            y = C * h + self.D * u
        return self.decoder(self.pred_to_z(y))          # (B,2,A,S) predicted beam channel

    def estimate(self, obs_beam, prior_beam, noise_var):
        b, _, na, ns = obs_beam.shape
        nv = noise_var.reshape(b, 1, 1, 1).expand(b, 1, na, ns)
        g = self.gate(torch.cat([obs_beam, prior_beam, nv], 1))       # (B,2,A,S) in [0,1]
        base = g * obs_beam + (1 - g) * prior_beam                    # Kalman fusion
        return base + self.refine(torch.cat([base, prior_beam, nv], 1))

    def losses(self, o, a, snr_range=(0.0, 20.0), horizon_range=(1, 6), noise_gen=None):
        cfg = self.config
        import random
        T = o.shape[1]
        beam = to_beamspace(o)                  # (B,T,2,A,S) -> beamspace (transform acts on last 3 dims)
        k = random.randint(horizon_range[0], min(horizon_range[1], T - 2))
        anchor = T - 1 - k

        x, z, h = self.encode(beam, a)

        # --- prediction loss: predict future beam channel k ahead ---
        b_pred = self.predict_beam(z[:, anchor], h[:, anchor], a[:, anchor:anchor + k])
        b_future = beam[:, anchor + k]
        loss_pred = F.mse_loss(b_pred, b_future)

        # --- estimation loss: predict-from-history prior + noisy obs -> current channel ---
        H_clean = o[:, anchor]                                        # antenna-domain clean
        power = H_clean.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        lo, hi = snr_range
        snr = torch.empty(H_clean.shape[0], 1, 1, 1, device=o.device).uniform_(lo, hi)
        nvar = power / (10 ** (snr / 10))
        Y = H_clean + torch.randn(H_clean.shape, generator=noise_gen, device=o.device) * nvar.sqrt()
        obs_beam = to_beamspace(Y)
        # prior for current step: predict from p steps back using clean history
        p = min(2, anchor)
        if p >= 1:
            prior_beam = self.predict_beam(z[:, anchor - p], h[:, anchor - p], a[:, anchor - p:anchor])
        else:
            prior_beam = torch.zeros_like(obs_beam)
        est_beam = self.estimate(obs_beam, prior_beam.detach(), nvar.reshape(-1))
        H_hat = from_beamspace(est_beam)
        loss_chan = F.mse_loss(H_hat, H_clean)

        total = loss_pred + 5.0 * loss_chan
        with torch.no_grad():
            met = {"total": total.item(), "pred": loss_pred.item(), "chan": loss_chan.item(),
                   "chan_nmse": (F.mse_loss(H_hat, H_clean) / H_clean.pow(2).mean()).item()}
        return total, met

    @torch.no_grad()
    def estimate_channel(self, o_seq, a, snr_db, noise_gen=None):
        """Inference: estimate the current (last-frame) channel from a noisy obs + history prior."""
        beam = to_beamspace(o_seq)
        x, z, h = self.encode(beam, a)
        t = o_seq.shape[1] - 1
        H_clean = o_seq[:, t]
        power = H_clean.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        nvar = power / (10 ** (snr_db / 10))
        Y = H_clean + torch.randn(H_clean.shape, generator=noise_gen, device=o_seq.device) * nvar.sqrt()
        p = min(2, t)
        prior_beam = self.predict_beam(z[:, t - p], h[:, t - p], a[:, t - p:t]) if p >= 1 else torch.zeros_like(to_beamspace(Y))
        est_beam = self.estimate(to_beamspace(Y), prior_beam, nvar.reshape(-1))
        return from_beamspace(est_beam), H_clean

    def forward(self, o, a, **kw):
        return self.losses(o, a, **kw)
