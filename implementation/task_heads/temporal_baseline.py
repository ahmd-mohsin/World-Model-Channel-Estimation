"""Learned temporal channel predictors (baselines for the world model's PREDICTION claim).

The fairest strong baseline is the WM predictor's exact pipeline — per-frame beamspace conv encoder,
roll k steps under planned actions, conv decoder — but with a GENERIC sequence model (GRU or
Transformer) in place of the action-conditioned selective SSM. If the WM still beats these, the win
is attributable to the selective-SSM dynamics, not merely "a sequence model over history."

Both predict the future beamspace channel; trained with the same MSE loss and multi-horizon schedule.
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from ..wireless_data.beamspace import to_beamspace, from_beamspace
    from ..beam_wm.beam_world_model import BeamEncoder, BeamDecoder
except ImportError:
    from wireless_data.beamspace import to_beamspace, from_beamspace
    from beam_wm.beam_world_model import BeamEncoder, BeamDecoder


class TemporalPredictor(nn.Module):
    """Generic learned predictor. backbone in {'gru','lstm','transformer'}.

    encode each frame -> latent x_t; sequence model over [x_t; a_t] -> state; roll k steps feeding
    planned actions; decode final state -> future beamspace channel.
    """

    def __init__(self, config, backbone: str = "gru"):
        super().__init__()
        self.config = config; self.backbone = backbone
        d, na, ns = config.latent_dim, config.n_antennas, config.n_subcarriers
        self.encoder = BeamEncoder(na, ns, d, base=config.unet_base_ch)
        self.decoder = BeamDecoder(na, ns, d, base=config.unet_base_ch)
        self.in_proj = nn.Linear(d + config.action_dim, d)
        self.act_proj = nn.Linear(config.action_dim, d)
        if backbone == "gru":
            self.rnn = nn.GRU(d, d, num_layers=2, batch_first=True)
        elif backbone == "lstm":
            self.rnn = nn.LSTM(d, d, num_layers=2, batch_first=True)
        elif backbone == "transformer":
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                               batch_first=True, dropout=0.0)
            self.tf = nn.TransformerEncoder(layer, num_layers=3)
            self.roll_cell = nn.GRUCell(d, d)   # autoregressive roll head for the transformer
        else:
            raise ValueError(backbone)

    def _encode_seq(self, beam_seq, a):
        b, t = beam_seq.shape[:2]
        x = self.encoder(beam_seq.reshape(b * t, *beam_seq.shape[2:])).reshape(b, t, -1)
        u = self.in_proj(torch.cat([x, a], dim=-1))
        return u

    def predict(self, beam_hist, a_hist, planned_acts):
        """beam_hist:(B,Th,2,A,S) a_hist:(B,Th,da) planned_acts:(B,k,da) -> future beam (B,2,A,S)."""
        u = self._encode_seq(beam_hist, a_hist)                     # (B,Th,d)
        if self.backbone in ("gru", "lstm"):
            out, state = self.rnn(u)                                # state carries history
            hcur = out[:, -1]
            # roll k steps: feed planned actions as pseudo-inputs
            for j in range(planned_acts.shape[1]):
                step_in = self.act_proj(planned_acts[:, j]).unsqueeze(1)
                out, state = self.rnn(step_in, state)
                hcur = out[:, -1]
            return self.decoder(hcur)
        else:  # transformer encodes history; GRUCell rolls forward on planned actions
            ctx = self.tf(u)[:, -1]                                 # (B,d) history summary
            h = ctx
            for j in range(planned_acts.shape[1]):
                h = self.roll_cell(self.act_proj(planned_acts[:, j]), h)
            return self.decoder(h)

    def losses(self, o, a, horizon_range=(1, 6), **kw):
        import random, torch.nn.functional as F
        T = o.shape[1]; beam = to_beamspace(o)
        k = random.randint(horizon_range[0], min(horizon_range[1], T - 2))
        anchor = T - 1 - k
        b_pred = self.predict(beam[:, :anchor + 1], a[:, :anchor + 1], a[:, anchor:anchor + k])
        loss = F.mse_loss(b_pred, beam[:, anchor + k])
        return loss, {"total": loss.item(), "pred": loss.item(), "chan": 0.0, "chan_nmse": 0.0}

    def forward(self, o, a, **kw):
        return self.losses(o, a, **kw)
