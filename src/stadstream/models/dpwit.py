from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import AnchorHead, make_detector_output


class SignedMaskAttention(nn.Module):
    """Compact DPWiT-style signed mask attention for sensor sequences."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mask = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        signed = tokens * self.mask(tokens)
        out, _ = self.attn(signed, signed, tokens, need_weights=False)
        return self.norm(tokens + out).transpose(1, 2)


class DPWiTDetector(nn.Module):
    """DPWiT/WiFiTAD-compatible detector for [B, C, T] WiFi CSI signals."""

    def __init__(self, in_channels: int = 60, num_classes: int = 8, hidden_dim: int = 128, num_anchors: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.low_band = nn.Sequential(
            nn.AvgPool1d(5, stride=1, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.high_band = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=8),
            nn.GELU(),
        )
        self.attn = SignedMaskAttention(hidden_dim)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.head = AnchorHead(hidden_dim, num_classes, num_anchors)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.stem(x)
        low = self.low_band(base)
        high = self.high_band(base - low)
        feat = self.fuse(torch.cat([self.attn(low), high], dim=1))
        return make_detector_output(self.head(feat))

