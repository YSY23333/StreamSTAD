from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn


@dataclass(frozen=True)
class MATRImportStatus:
    ok: bool
    matr_root: str
    message: str


def check_matr_import(matr_root: str | Path) -> MATRImportStatus:
    """Check whether the official MATR codebase is importable.

    This does not modify official code. It only appends ``matr_root`` to
    ``sys.path`` for the current process and imports the official builder.
    """
    root = Path(matr_root).resolve()
    if not root.exists():
        return MATRImportStatus(False, str(root), "MATR root does not exist")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from models import build_model  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostic path
        return MATRImportStatus(False, str(root), f"import failed: {exc}")
    return MATRImportStatus(True, str(root), "official MATR import ok")


class SensorToMATRFeatureAdapter(nn.Module):
    """Project WiFi/sensor signal chunks to official MATR feature tensors.

    Official MATR expects a sequence feature tensor shaped ``[B, S, D]``. This
    adapter is the only model-side transform allowed for faithful transfer:
    it converts our raw sensor signal ``[B, C, T]`` into that feature shape.
    """

    def __init__(self, in_channels: int, feat_dim: int, num_frame: int):
        super().__init__()
        self.num_frame = int(num_frame)
        self.proj = nn.Conv1d(in_channels, feat_dim, kernel_size=1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(signal.shape)}")
        feat = self.proj(signal)
        feat = torch.nn.functional.adaptive_avg_pool1d(feat, self.num_frame)
        return feat.transpose(1, 2).contiguous()

