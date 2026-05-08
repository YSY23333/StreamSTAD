from __future__ import annotations

import torch
from torch import nn


class STADStreamNet(nn.Module):
    """STADStream v1: chunk-level sensor online TAD.

    This is the first runnable version of the proposal: a channel-aware temporal
    encoder plus action/class/boundary/offset heads. The recurrent core is GRU
    for portability; it can be replaced by Mamba later without changing labels
    or online decoding.
    """

    def __init__(self, in_channels: int = 30, num_classes: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.channel_filter = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=8, dilation=2, groups=in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.actionness_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.offset_head = nn.Linear(hidden_dim, 2)
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.channel_filter(x).transpose(1, 2)
        seq, _ = self.temporal(feat)
        pooled = torch.cat([seq[:, -1], seq.mean(dim=1)], dim=-1)
        z = self.fuse(pooled)
        return {
            "actionness_logit": self.actionness_head(z).squeeze(-1),
            "cls_logits": self.cls_head(z),
            "offsets": self.offset_head(z),
            "start_logit": self.start_head(z).squeeze(-1),
            "end_logit": self.end_head(z).squeeze(-1),
            "feature": z,
        }

