"""Deep stacked selective-SSM channel predictor (the FAIR strong SSM, vs the single-layer SSM in the
world model and vs the Transformer baseline).

A single diagonal selective-SSM layer (as the WM predictor uses) is a weak instantiation of the SSM
family; real Mamba/S4 models STACK many gated SSM blocks with residual connections. This module is that
proper deep SSM, matched in capacity (~7M) to the Transformer baseline and sharing the SAME beamspace
encoder/decoder — so a head-to-head isolates "deep selective-SSM dynamics" vs "attention" on channel
prediction, with everything else held fixed.

Each block:  x -> LayerNorm -> action-conditioned selective SSM (A,B,C,Δ from a_t) -> SiLU gate -> +res
Rollout: carry per-block SSM states; for each planned action advance one step, propagate up the stack.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..config import SSWMConfig
    from ..selection_net import SelectionNet
    from ..selective_ssm import discretize
    from ..wireless_data.beamspace import to_beamspace
    from ..beam_wm.beam_world_model import BeamEncoder, BeamDecoder
except ImportError:
    from config import SSWMConfig
    from selection_net import SelectionNet
    from selective_ssm import discretize
    from wireless_data.beamspace import to_beamspace
    from beam_wm.beam_world_model import BeamEncoder, BeamDecoder


class SSMBlock(nn.Module):
    """Gated selective-SSM block with a residual connection (Mamba-style)."""

    def __init__(self, d: int, state: int, action_dim: int, sel_hidden: int):
        super().__init__()
        self.d, self.state = d, state
        cfg = SSWMConfig(state_dim=state, action_dim=action_dim, selection_hidden=sel_hidden)
        self.selection = SelectionNet(cfg)
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, state)
        self.gate_proj = nn.Linear(d, d)
        self.D = nn.Parameter(torch.ones(state))
        self.out_proj = nn.Linear(state, d)

    def forward(self, x, a):
        """x:(B,T,d) tokens, a:(B,T,da) actions -> (out:(B,T,d), final_state:(B,state))."""
        b, t, _ = x.shape
        xn = self.norm(x)
        u = self.in_proj(xn)                                   # (B,T,state)
        A, B, C, dt = self.selection(a)                        # each (B,T,state)
        dA, dB = discretize(A, B, dt)
        h = torch.zeros(b, self.state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(t):
            h = dA[:, i] * h + dB[:, i] * u[:, i]
            ys.append(C[:, i] * h + self.D * u[:, i])
        y = torch.stack(ys, 1)                                 # (B,T,state)
        out = self.out_proj(y) * F.silu(self.gate_proj(xn))    # gated
        return x + out, h

    def step(self, x_t, a_t, h):
        """single-step rollout. x_t:(B,d), a_t:(B,da), h:(B,state) -> (out:(B,d), h_new)."""
        xn = self.norm(x_t)
        u = self.in_proj(xn)
        A, B, C, dt = self.selection(a_t)
        dA, dB = discretize(A, B, dt)
        h = dA * h + dB * u
        y = C * h + self.D * u
        out = self.out_proj(y) * F.silu(self.gate_proj(xn))
        return x_t + out, h


class DeepSSMPredictor(nn.Module):
    def __init__(self, config, n_layers: int = 4, state: int | None = None):
        super().__init__()
        self.config = config
        d, na, ns = config.latent_dim, config.n_antennas, config.n_subcarriers
        state = state or config.state_dim
        self.encoder = BeamEncoder(na, ns, d, base=config.unet_base_ch)
        self.decoder = BeamDecoder(na, ns, d, base=config.unet_base_ch)
        self.act_embed = nn.Linear(config.action_dim, d)       # token input during rollout
        self.blocks = nn.ModuleList(
            [SSMBlock(d, state, config.action_dim, config.selection_hidden) for _ in range(n_layers)])

    def predict(self, beam_hist, a_hist, planned_acts):
        """beam_hist:(B,Th,2,A,S) a_hist:(B,Th,da) planned_acts:(B,k,da) -> future beam (B,2,A,S)."""
        b, th = beam_hist.shape[:2]
        x = self.encoder(beam_hist.reshape(b * th, *beam_hist.shape[2:])).reshape(b, th, -1)
        # run stack over history, collect per-block final states
        states = []
        cur = x
        for blk in self.blocks:
            cur, h = blk(cur, a_hist)
            states.append(h)
        tok = cur[:, -1]                                       # top-block last output
        # roll k steps: bottom token from planned action, propagate up the stack
        for j in range(planned_acts.shape[1]):
            a_j = planned_acts[:, j]
            tok = self.act_embed(a_j)
            new_states = []
            for blk, h in zip(self.blocks, states):
                tok, h = blk.step(tok, a_j, h)
                new_states.append(h)
            states = new_states
        return self.decoder(tok)

    def losses(self, o, a, horizon_range=(1, 6), **kw):
        import random
        T = o.shape[1]; beam = to_beamspace(o)
        k = random.randint(horizon_range[0], min(horizon_range[1], T - 2))
        anchor = T - 1 - k
        b_pred = self.predict(beam[:, :anchor + 1], a[:, :anchor + 1], a[:, anchor:anchor + k])
        loss = F.mse_loss(b_pred, beam[:, anchor + k])
        return loss, {"total": loss.item(), "pred": loss.item(), "chan": 0.0, "chan_nmse": 0.0}

    def forward(self, o, a, **kw):
        return self.losses(o, a, **kw)
