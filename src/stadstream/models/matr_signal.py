from __future__ import annotations

import torch
from torch import nn

from .common import AnchorHead, make_detector_output


class MATRSignalBaseline(nn.Module):
    """Memory-augmented online TAL baseline adapted to sensor [B, C, T] input.

    This is the integration layer for the official MATR baseline idea: segment
    encoder + memory encoder + query-style prediction. It keeps our detector
    output contract so it can be trained with the same WiFiTAD pipeline.
    """

    def __init__(
        self,
        in_channels: int = 60,
        num_classes: int = 8,
        hidden_dim: int = 128,
        num_anchors: int = 128,
        memory_segments: int = 4,
        segment_tokens: int = 64,
    ):
        super().__init__()
        self.segment_tokens = segment_tokens
        self.memory_segments = memory_segments
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.segment_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.memory_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.memory_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.head = AnchorHead(hidden_dim, num_classes, num_anchors)
        self.register_buffer("_memory", torch.empty(0), persistent=False)

    def reset_stream(self) -> None:
        self._memory = torch.empty(0, device=self._memory.device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.proj(x)
        feat = torch.nn.functional.adaptive_avg_pool1d(feat, self.segment_tokens)
        tokens = feat.transpose(1, 2)
        encoded = self.segment_encoder(tokens)

        if self._memory.numel() and self._memory.shape[0] == x.shape[0]:
            memory = torch.cat([self._memory, encoded.detach()], dim=1)
            memory = memory[:, -self.memory_segments * self.segment_tokens :]
        else:
            memory = encoded.detach()
        self._memory = memory

        mem_context = self.memory_encoder(memory).mean(dim=1, keepdim=True)
        gated = encoded + self.memory_gate(mem_context) * mem_context
        return make_detector_output(self.head(gated.transpose(1, 2)))

