from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class AnchorHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, num_anchors: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc = nn.Conv1d(hidden_dim, 2, kernel_size=3, padding=1)
        self.conf = nn.Conv1d(hidden_dim, num_classes, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = F.adaptive_avg_pool1d(feat, self.num_anchors)
        loc = F.softplus(self.loc(pooled)).transpose(1, 2).contiguous()
        conf = self.conf(pooled).transpose(1, 2).contiguous()
        centers = torch.linspace(
            0.5 / self.num_anchors,
            1.0 - 0.5 / self.num_anchors,
            self.num_anchors,
            device=feat.device,
        ).view(1, self.num_anchors, 1)
        return {"loc": loc, "conf": conf, "priors": centers}


def make_detector_output(head_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "loc": head_out["loc"],
        "conf": head_out["conf"],
        "priors": head_out["priors"],
    }

