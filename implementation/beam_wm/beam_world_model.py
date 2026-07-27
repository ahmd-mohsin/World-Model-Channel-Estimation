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
    # ablation modes — each removes exactly ONE mechanism, all else identical:
    #   "full"      : the complete model (default)
    #   "prior"     : estimation gate sees NO world-model prior (prior_beam := 0)
    #   "action"    : actions zeroed everywhere (SSM/predictor can't use motion)
    #   "ssm"       : prior := previous clean frame echoed (persistence; no learned dynamics)
    #   "beamspace" : operate in the raw antenna-subcarrier domain (skip the 2D-DFT)
    ABLATIONS = ("full", "prior", "action", "ssm", "beamspace")

    def __init__(self, config: SSWMConfig, ablate: str = "full", sparse_stride: int = 0,
                 deep_fusion: bool = False, fusion_blocks: int = 8):
        super().__init__()
        assert ablate in self.ABLATIONS, f"unknown ablation {ablate!r}"
        self.ablate = ablate
        self.sparse_stride = sparse_stride      # 0 = dense pilots; s>1 = 1-in-s comb pilots
        self.deep_fusion = deep_fusion          # ReEsNet-capacity sparse head fed the WM prior
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
        # SPARSE-PILOT fusion (H-domain, mask-aware): [Y_masked(2), prior_H(2), mask(1), nv(1)] = 6ch
        self.sparse_gate = nn.Sequential(nn.Conv2d(6, config.unet_base_ch, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(config.unet_base_ch, 2, 3, padding=1), nn.Sigmoid())
        self.sparse_refine = nn.Sequential(_ConvBlock(6, config.unet_base_ch),
                                           nn.Conv2d(config.unet_base_ch, 2, 1))
        nn.init.zeros_(self.sparse_refine[-1].weight); nn.init.zeros_(self.sparse_refine[-1].bias)
        if deep_fusion:
            # ReEsNet-capacity fusion head: SAME residual body as the standalone baseline, but fed
            # the WM prior as extra channels -> [Y_masked(2), prior_H(2), mask(1), nv(1)] = 6 in.
            # At equal capacity it sees strictly MORE than ReEsNet (obs + world-model prediction),
            # so if the prior has value the fused head should be >= ReEsNet everywhere.
            from ..task_heads.deep_baseline import _ResBlock
            fb = 64
            self.fuse_in = nn.Conv2d(6, fb, 3, padding=1)
            self.fuse_body = nn.Sequential(*[_ResBlock(fb) for _ in range(fusion_blocks)])
            self.fuse_mid = nn.Conv2d(fb, fb, 3, padding=1)
            self.fuse_out = nn.Conv2d(fb, 2, 3, padding=1)
            # SNR-ADAPTIVE ANCHOR (fixes the -5dB instability + 30dB overhead): a mask-aware gate
            # produces base = g*obs + (1-g)*prior (-> prior at low SNR, -> obs at high SNR), and the
            # residual body only CORRECTS it. fuse_out zero-init => training starts exactly at the
            # gated fusion, so the low-SNR fallback is deterministic (kills the seed variance).
            self.fuse_gate = nn.Sequential(nn.Conv2d(6, fb, 3, padding=1), nn.GELU(),
                                           nn.Conv2d(fb, 2, 3, padding=1), nn.Sigmoid())
            nn.init.zeros_(self.fuse_out.weight); nn.init.zeros_(self.fuse_out.bias)

    # ---- domain transform (identity under the "beamspace" ablation) ----
    def _to(self, x):
        return x if self.ablate == "beamspace" else to_beamspace(x)

    def _from(self, x):
        return x if self.ablate == "beamspace" else from_beamspace(x)

    def _act(self, a):
        """Zero actions under the 'action' ablation so motion cannot inform dynamics."""
        return torch.zeros_like(a) if self.ablate == "action" else a

    # ---- encode a beam sequence to latents + SSM states ----
    def encode(self, beam_seq, a):
        b, t = beam_seq.shape[:2]
        a = self._act(a)
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
        planned_acts = self._act(planned_acts)
        for j in range(planned_acts.shape[1]):
            a_j = planned_acts[:, j]
            A, B, C, dt = self.selection(a_j)
            u = self.pred_in(a_j)
            dA, dB = discretize(A, B, dt)
            h = dA * h + dB * u
            y = C * h + self.D * u
        return self.decoder(self.pred_to_z(y))          # (B,2,A,S) predicted beam channel

    def _prior(self, z, h, a, anchor, obs_beam, beam):
        """World-model prior for the current step, honoring ablations.
          full/action/beamspace : roll the predictor from p steps back on clean history
          prior                  : no prior (zeros) — gate sees only obs + noise
          ssm                    : previous clean frame echoed (persistence; no learned dynamics)
        """
        if self.ablate == "prior":
            return torch.zeros_like(obs_beam)
        p = min(2, anchor)
        if self.ablate == "ssm":
            return beam[:, anchor - 1] if anchor >= 1 else torch.zeros_like(obs_beam)
        if p >= 1:
            return self.predict_beam(z[:, anchor - p], h[:, anchor - p], a[:, anchor - p:anchor])
        return torch.zeros_like(obs_beam)

    def estimate(self, obs_beam, prior_beam, noise_var):
        b, _, na, ns = obs_beam.shape
        nv = noise_var.reshape(b, 1, 1, 1).expand(b, 1, na, ns)
        g = self.gate(torch.cat([obs_beam, prior_beam, nv], 1))       # (B,2,A,S) in [0,1]
        base = g * obs_beam + (1 - g) * prior_beam                    # Kalman fusion
        return base + self.refine(torch.cat([base, prior_beam, nv], 1))

    @staticmethod
    def comb_mask(n_sub, stride, device):
        """1-in-`stride` comb pilot mask over subcarriers: (1,1,1,n_sub), 1=pilot, 0=unobserved."""
        m = torch.zeros(n_sub, device=device)
        m[::stride] = 1.0
        return m.reshape(1, 1, 1, n_sub)

    def estimate_sparse(self, Y_masked, prior_H, mask, noise_var):
        """Sparse-pilot fusion in the ANTENNA-SUBCARRIER (H) domain.

        On observed subcarriers the gate may trust the (masked) observation; on UNOBSERVED
        subcarriers there is no observation, so the fused estimate is FORCED onto the world-model
        prior via g_eff = g * mask. This is where a good prior structurally wins.
          Y_masked : (B,2,A,S) noisy obs with unobserved subcarriers zeroed
          prior_H  : (B,2,A,S) world-model prior, decoded to H-domain (dense)
          mask     : (1,1,1,S) or (B,1,A,S) comb mask
        """
        b, _, na, ns = Y_masked.shape
        nv = noise_var.reshape(b, 1, 1, 1).expand(b, 1, na, ns)
        mfull = mask.expand(b, 1, na, ns)
        if self.deep_fusion:
            # ReEsNet-capacity residual body over [Y_masked, prior_H, mask, nv], anchored on an
            # SNR-adaptive gated fusion so the head only CORRECTS a sane base (stable at SNR extremes).
            x = torch.cat([Y_masked, prior_H, mfull, nv], 1)                 # (B,6,A,S)
            g = self.fuse_gate(x) * mask            # mask-aware: 0 on unobserved carriers -> prior
            base = g * Y_masked + (1 - g) * prior_H
            h = self.fuse_in(x)
            h = self.fuse_mid(self.fuse_body(h)) + h
            return base + self.fuse_out(h)          # zero-init out => starts exactly at base
        g = self.sparse_gate(torch.cat([Y_masked, prior_H, mfull, nv], 1))   # (B,2,A,S)
        g_eff = g * mask                                                     # 0 on unobserved carriers
        base = g_eff * Y_masked + (1 - g_eff) * prior_H
        return base + self.sparse_refine(torch.cat([base, prior_H, mfull, nv], 1))

    def losses(self, o, a, snr_range=(0.0, 20.0), horizon_range=(1, 6), noise_gen=None,
               chan_only=False):
        cfg = self.config
        import random
        T = o.shape[1]
        beam = self._to(o)                      # (B,T,2,A,S) -> working domain (beamspace or raw)
        k = random.randint(horizon_range[0], min(horizon_range[1], T - 2))
        anchor = T - 1 - k

        # chan_only (estimation-only fine-tune): encoder/SSM/predictor are FROZEN -> run them under
        # no_grad so only the fusion head gets gradients; prediction stays exactly as trained.
        if chan_only:
            with torch.no_grad():
                x, z, h = self.encode(beam, a)
            loss_pred = torch.zeros((), device=o.device)
        else:
            x, z, h = self.encode(beam, a)
            # --- prediction loss: predict future channel k ahead (in the working domain) ---
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
        obs_beam = self._to(Y)
        prior_beam = self._prior(z, h, a, anchor, obs_beam, beam)
        if self.sparse_stride > 1:
            # SPARSE PILOTS: only 1-in-stride subcarriers observed; fuse in H-domain (mask-aware).
            mask = self.comb_mask(cfg.n_subcarriers, self.sparse_stride, o.device)
            Y_masked = Y * mask
            prior_H = self._from(prior_beam).detach()
            H_hat = self.estimate_sparse(Y_masked, prior_H, mask, nvar.reshape(-1))
        else:
            est_beam = self.estimate(obs_beam, prior_beam.detach(), nvar.reshape(-1))
            H_hat = self._from(est_beam)
        loss_chan = F.mse_loss(H_hat, H_clean)

        total = loss_chan if chan_only else (loss_pred + 5.0 * loss_chan)
        with torch.no_grad():
            met = {"total": total.item(), "pred": loss_pred.item(), "chan": loss_chan.item(),
                   "chan_nmse": (F.mse_loss(H_hat, H_clean) / H_clean.pow(2).mean()).item()}
        return total, met

    @torch.no_grad()
    def estimate_channel(self, o_seq, a, snr_db, noise_gen=None):
        """Inference: estimate the current (last-frame) channel from a noisy obs + history prior."""
        beam = self._to(o_seq)
        x, z, h = self.encode(beam, a)
        t = o_seq.shape[1] - 1
        H_clean = o_seq[:, t]
        power = H_clean.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        nvar = power / (10 ** (snr_db / 10))
        Y = H_clean + torch.randn(H_clean.shape, generator=noise_gen, device=o_seq.device) * nvar.sqrt()
        obs_beam = self._to(Y)
        prior_beam = self._prior(z, h, a, t, obs_beam, beam)
        est_beam = self.estimate(obs_beam, prior_beam, nvar.reshape(-1))
        return self._from(est_beam), H_clean

    @torch.no_grad()
    def estimate_channel_sparse(self, o_seq, a, snr_db, stride, noise_gen=None):
        """Sparse-pilot inference: observe only 1-in-`stride` subcarriers, fuse with the prior."""
        beam = self._to(o_seq)
        x, z, h = self.encode(beam, a)
        t = o_seq.shape[1] - 1
        H_clean = o_seq[:, t]
        power = H_clean.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        nvar = power / (10 ** (snr_db / 10))
        Y = H_clean + torch.randn(H_clean.shape, generator=noise_gen, device=o_seq.device) * nvar.sqrt()
        mask = self.comb_mask(self.config.n_subcarriers, stride, o_seq.device)
        Y_masked = Y * mask
        prior_beam = self._prior(z, h, a, t, self._to(Y), beam)
        prior_H = self._from(prior_beam)
        H_hat = self.estimate_sparse(Y_masked, prior_H, mask, nvar.reshape(-1))
        return H_hat, H_clean, Y_masked, mask

    def freeze_for_finetune(self):
        """Freeze everything except the sparse fusion head, for estimation-only fine-tuning.
        Keeps encoder/SSM/predictor (hence the prediction task + the prior) exactly as trained;
        only the head that maps (obs, prior) -> estimate keeps learning."""
        head = {"fuse_in", "fuse_body", "fuse_mid", "fuse_out", "fuse_gate",
                "sparse_gate", "sparse_refine"}
        n_train = n_frozen = 0
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            p.requires_grad = top in head
            n_train += p.numel() if p.requires_grad else 0
            n_frozen += 0 if p.requires_grad else p.numel()
        return n_train, n_frozen

    def forward(self, o, a, **kw):
        return self.losses(o, a, **kw)
