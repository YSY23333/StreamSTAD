from __future__ import annotations

import torch
from torch import nn

from .common import AnchorHead, make_detector_output


class ChannelGate(nn.Module):
    def __init__(self, hidden_dim: int, reduction: int = 4):
        super().__init__()
        mid = max(hidden_dim // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SlimSTADDetector(nn.Module):
    """SlimSTAD-style decoupled channel modeling with a shared detector head."""

    def __init__(self, in_channels: int = 60, num_classes: int = 8, hidden_dim: int = 128, num_anchors: int = 128):
        super().__init__()
        self.channel_embed = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.depthwise_temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, groups=hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=8, dilation=2, groups=hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.channel_gate = ChannelGate(hidden_dim)
        self.mix = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.head = AnchorHead(hidden_dim, num_classes, num_anchors)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.channel_embed(x)
        feat = self.depthwise_temporal(feat)
        feat = self.mix(self.channel_gate(feat))
        return make_detector_output(self.head(feat))

