from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .wifitad import load_class_index


def _read_info(path: str | Path) -> dict[str, int]:
    df = pd.read_csv(path)
    return {str(row.video): int(row.sample_count) for _, row in df.iterrows()}


def _read_annos(path: str | Path, origin_to_dense: dict[int, int]) -> dict[str, list[dict]]:
    df = pd.read_csv(path)
    annos: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        video = str(row.video)
        annos.setdefault(video, []).append(
            {
                "start": float(row.startFrame),
                "end": float(row.endFrame),
                "label": int(origin_to_dense[int(row.type_idx)]),
            }
        )
    return annos


class StreamWindowDataset(Dataset):
    """Chunk/window labels for sensor-native online TAD.

    A window is positive when it overlaps an action enough. This avoids MATR's
    sparse "action end near detection point" supervision.
    """

    def __init__(
        self,
        info_path: str | Path,
        anno_path: str | Path,
        data_path: str | Path,
        class_index_path: str | Path,
        window_size: int,
        stride: int,
        num_classes: int,
        overlap_thresh: float = 0.15,
        training: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.num_classes = int(num_classes)
        self.overlap_thresh = float(overlap_thresh)
        self.training = training
        self.origin_to_dense, self.dense_to_name = load_class_index(class_index_path)
        self.video_len = _read_info(info_path)
        self.annos = _read_annos(anno_path, self.origin_to_dense)
        self.samples: list[tuple[str, int]] = []
        for video, length in self.video_len.items():
            ends = list(range(max(1, self.stride), length + 1, self.stride))
            if not ends or ends[-1] != length:
                ends.append(length)
            self.samples.extend((video, ed) for ed in ends)
        self._cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_signal(self, video: str) -> torch.Tensor:
        if video not in self._cache:
            x = np.load(self.data_path / f"{video}.npy").astype(np.float32)
            if x.shape[0] < x.shape[-1]:
                x = x.T
            self._cache[video] = torch.from_numpy((x / 40.0).T.copy()).float()
        return self._cache[video]

    def _target_for_window(self, video: str, win_start: int, win_end: int) -> dict[str, torch.Tensor]:
        best = None
        best_score = 0.0
        for ann in self.annos.get(video, []):
            inter = max(0.0, min(win_end, ann["end"]) - max(win_start, ann["start"]))
            if inter <= 0:
                continue
            ioa = inter / max(ann["end"] - ann["start"], 1.0)
            iow = inter / max(win_end - win_start, 1.0)
            score = max(ioa, iow)
            if score > best_score:
                best = ann
                best_score = score

        cls = 0
        actionness = 0.0
        start_offset = 0.0
        end_offset = 0.0
        start_boundary = 0.0
        end_boundary = 0.0
        if best is not None and best_score >= self.overlap_thresh:
            cls = int(best["label"])
            actionness = 1.0
            ref = float(win_end)
            norm = float(self.window_size)
            start_offset = (ref - best["start"]) / norm
            end_offset = (ref - best["end"]) / norm
            radius = max(self.stride, self.window_size // 8)
            start_boundary = 1.0 if abs(best["start"] - ref) <= radius or (win_start <= best["start"] <= win_end) else 0.0
            end_boundary = 1.0 if abs(best["end"] - ref) <= radius or (win_start <= best["end"] <= win_end) else 0.0

        return {
            "actionness": torch.tensor(actionness, dtype=torch.float32),
            "cls": torch.tensor(cls, dtype=torch.long),
            "offsets": torch.tensor([start_offset, end_offset], dtype=torch.float32),
            "start_boundary": torch.tensor(start_boundary, dtype=torch.float32),
            "end_boundary": torch.tensor(end_boundary, dtype=torch.float32),
        }

    def __getitem__(self, index: int):
        video, win_end = self.samples[index]
        win_start = win_end - self.window_size
        signal = self._load_signal(video)
        clip = signal[:, max(0, win_start) : win_end]
        if clip.shape[-1] < self.window_size:
            clip = torch.nn.functional.pad(clip, (self.window_size - clip.shape[-1], 0))
        target = self._target_for_window(video, win_start, win_end)
        meta = {"video": video, "window_start": win_start, "window_end": win_end}
        return clip, target, meta


def stream_window_collate(batch):
    clips = torch.stack([item[0] for item in batch], dim=0)
    targets = {
        "actionness": torch.stack([item[1]["actionness"] for item in batch]),
        "cls": torch.stack([item[1]["cls"] for item in batch]),
        "offsets": torch.stack([item[1]["offsets"] for item in batch]),
        "start_boundary": torch.stack([item[1]["start_boundary"] for item in batch]),
        "end_boundary": torch.stack([item[1]["end_boundary"] for item in batch]),
    }
    metas = [item[2] for item in batch]
    return clips, targets, metas

