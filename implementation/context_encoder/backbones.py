from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..config import SSWMConfig
except ImportError:
    from config import SSWMConfig

_LWM_DIR = os.path.join(os.path.dirname(__file__), "lwm")


class StubBackbone(nn.Module):
    def __init__(self, hidden: int, patch: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Conv2d(2, hidden, kernel_size=patch, stride=patch)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        feat = self.proj(channel)
        return feat.flatten(2).transpose(1, 2)


class _RandomViT(nn.Module):
    def __init__(self, hidden: int, image_size: int, patch: int = 14) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)

    def forward(self, pixel_values: torch.Tensor):
        tokens = self.proj(pixel_values).flatten(2).transpose(1, 2)

        class _Out:
            pass

        out = _Out()
        out.last_hidden_state = tokens
        return out


class IJepaBackbone(nn.Module):
    def __init__(self, config: SSWMConfig) -> None:
        super().__init__()
        self.hidden = config.jepa_hidden
        self.image_size = config.jepa_image_size
        self.adapter = nn.Sequential(
            nn.Conv2d(config.obs_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )
        self.vit = self._load_vit(config)

    @staticmethod
    def _load_vit(config: SSWMConfig) -> nn.Module:
        try:
            from transformers import IJepaModel

            return IJepaModel.from_pretrained(config.jepa_checkpoint)
        except Exception as exc:
            print(f"[IJepaBackbone] pretrained ViT unavailable ({exc}); using random ViT-shaped stub.")
            return _RandomViT(config.jepa_hidden, config.jepa_image_size)

    def frozen_modules(self):
        return [self.vit]

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        x = self.adapter(channel)
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.vit(pixel_values=x).last_hidden_state


class LWMBackbone(nn.Module):
    def __init__(self, config: SSWMConfig) -> None:
        super().__init__()
        self.hidden = config.lwm_hidden
        self.patch_rows = config.lwm_patch_rows
        self.patch_cols = config.lwm_patch_cols
        self.element_length = config.lwm_element_length
        self.register_buffer("cls_token", 0.2 * torch.ones(self.element_length))
        self.net, self.is_pretrained = self._load_lwm(config)

    def _load_lwm(self, config: SSWMConfig):
        import sys

        try:
            if _LWM_DIR not in sys.path:
                sys.path.insert(0, _LWM_DIR)
            from lwm_model import lwm

            ckpt = config.lwm_checkpoint
            if not os.path.isabs(ckpt):
                ckpt = os.path.join(_LWM_DIR, os.path.basename(ckpt))
            if not os.path.exists(ckpt):
                from huggingface_hub import hf_hub_download

                ckpt = hf_hub_download("wi-lab/lwm-v1.1", "models/model.pth")
            state = torch.load(ckpt, map_location="cpu")
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model = lwm(d_model=config.lwm_hidden, element_length=config.lwm_element_length)
            model.load_state_dict(state)
            return model, True
        except Exception as exc:
            print(f"[LWMBackbone] pretrained LWM unavailable ({exc}); using offline stub.")
            return StubBackbone(config.lwm_hidden, patch=4), False

    def frozen_modules(self):
        return [self.net] if self.is_pretrained else []

    def _tokenize(self, channel: torch.Tensor) -> torch.Tensor:
        n, _, n_rows, n_cols = channel.shape
        real = channel[:, 0]
        imag = channel[:, 1]
        interleaved = torch.empty(n, n_rows, n_cols * 2, dtype=channel.dtype, device=channel.device)
        interleaved[:, :, 0::2] = real
        interleaved[:, :, 1::2] = imag

        pr, pc = self.patch_rows, self.patch_cols
        pad_r = (int(np.ceil(n_rows / pr)) * pr) - n_rows
        pad_c = (int(np.ceil(n_cols / pc)) * pc) - n_cols
        if pad_r > 0 or pad_c > 0:
            interleaved = F.pad(interleaved, (0, pad_c * 2, 0, pad_r))

        _, padded_rows, padded_cols2 = interleaved.shape
        padded_cols = padded_cols2 // 2
        patches = []
        for i in range(0, padded_rows, pr):
            for j in range(0, padded_cols, pc):
                patch = interleaved[:, i:i + pr, j * 2:(j + pc) * 2]
                patches.append(patch.reshape(n, -1))
        tokens = torch.stack(patches, dim=1)

        cls = self.cls_token.to(tokens.dtype).expand(n, 1, self.element_length)
        return torch.cat([cls, tokens], dim=1)

    def forward(self, channel: torch.Tensor) -> torch.Tensor:
        if not self.is_pretrained:
            return self.net(channel)
        input_ids = self._tokenize(channel)
        output, _ = self.net(input_ids)
        return output


def build_backbone(config: SSWMConfig) -> nn.Module:
    if not config.use_pretrained or config.backbone == "stub":
        return StubBackbone(config.backbone_hidden, patch=4)
    if config.backbone == "lwm":
        return LWMBackbone(config)
    if config.backbone == "ijepa":
        return IJepaBackbone(config)
    raise ValueError(f"unknown backbone: {config.backbone!r}")
