from __future__ import annotations

import torch
from torch import nn

from .common import AnchorHead, make_detector_output


class MOADSignalBaseline(nn.Module):
    """MOAD-style Mamba OAD baseline adapted to sensor [B, C, T] signals.

    The public MOAD paper describes Backtrace Mamba with hierarchical memory
    compression, memory quantization, and temporal soft pruning. This module
    keeps those experimental knobs in our unified detector interface. It uses a
    GRU recurrence as the default portable stand-in for Mamba; swapping in
    mamba_ssm later only needs to replace ``self.recurrent``.
    """

    def __init__(
        self,
        in_channels: int = 60,
        num_classes: int = 8,
        hidden_dim: int = 128,
        num_anchors: int = 128,
        memory_size: int = 64,
        quant_levels: int = 16,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.quant_levels = quant_levels
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.recurrent = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.scene_memory = nn.Parameter(torch.zeros(1, memory_size, hidden_dim), requires_grad=False)
        self.action_memory = nn.Parameter(torch.zeros(1, memory_size, hidden_dim), requires_grad=False)
        self.memory_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.head = AnchorHead(hidden_dim, num_classes, num_anchors)

    def reset_stream(self) -> None:
        self.scene_memory.zero_()
        self.action_memory.zero_()

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_levels <= 1:
            return x
        scale = x.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        levels = float(self.quant_levels - 1)
        return torch.round((x / scale + 1.0) * 0.5 * levels) / levels * 2.0 * scale - scale

    def _soft_prune(self, memory: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if memory.shape[1] <= self.memory_size:
            return memory
        keep = scores.squeeze(-1).topk(self.memory_size, dim=1).indices
        keep = keep.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        return memory.gather(dim=1, index=keep)

    def _update_memory(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        scene_memory = self.scene_memory.expand(batch, -1, -1)
        action_memory = self.action_memory.expand(batch, -1, -1)

        token_scores = self.score_head(tokens).sigmoid()
        scene_tokens = tokens[:, :: max(tokens.shape[1] // 8, 1)]
        action_tokens = self._soft_prune(tokens, token_scores)

        scene_memory = torch.cat([scene_memory, scene_tokens.detach()], dim=1)[:, -self.memory_size :]
        action_memory = torch.cat([action_memory, action_tokens.detach()], dim=1)
        action_scores = self.score_head(action_memory).sigmoid()
        action_memory = self._soft_prune(action_memory, action_scores)
        return self._quantize(scene_memory), self._quantize(action_memory)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.input_proj(x).transpose(1, 2)
        encoded, _ = self.recurrent(tokens)
        scene_memory, action_memory = self._update_memory(encoded)

        scene_ctx = scene_memory.mean(dim=1, keepdim=True).expand_as(encoded)
        action_ctx = action_memory.mean(dim=1, keepdim=True).expand_as(encoded)
        fused = self.memory_fuse(torch.cat([encoded, scene_ctx, action_ctx], dim=-1))
        return make_detector_output(self.head(fused.transpose(1, 2)))

