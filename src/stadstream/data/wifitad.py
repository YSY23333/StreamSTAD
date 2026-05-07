from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClipSpec:
    video: str
    offset: int


def load_class_index(path: str | Path) -> tuple[dict[int, int], dict[int, str]]:
    rows = np.loadtxt(path, dtype=str)
    origin_to_dense: dict[int, int] = {}
    dense_to_name: dict[int, str] = {}
    for i, row in enumerate(rows):
        dense_id = i + 1
        origin_to_dense[int(row[0])] = dense_id
        dense_to_name[dense_id] = str(row[1])
    return origin_to_dense, dense_to_name


def _read_info(path: str | Path) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path)
    return {
        str(row.video): {
            "fps": int(row.fps),
            "sample_fps": int(row.sample_fps),
            "count": int(row["count"]),
            "sample_count": int(row.sample_count),
        }
        for _, row in df.iterrows()
    }


def _read_annos(
    path: str | Path,
    video_infos: dict[str, dict[str, int]],
    origin_to_dense: dict[int, int],
) -> dict[str, list[list[float]]]:
    df = pd.read_csv(path)
    annos: dict[str, list[list[float]]] = {}
    for _, row in df.iterrows():
        video = str(row.video)
        if video not in video_infos:
            continue
        ratio = video_infos[video]["sample_count"] / max(video_infos[video]["count"], 1)
        start = float(row.startFrame) * ratio
        end = float(row.endFrame) * ratio
        cls = origin_to_dense[int(row.type_idx)]
        annos.setdefault(video, []).append([start, end, cls])
    return annos


def _make_clips(
    video_infos: dict[str, dict[str, int]],
    video_annos: dict[str, list[list[float]]],
    clip_length: int,
    stride: int,
) -> list[ClipSpec]:
    clips: list[ClipSpec] = []
    for video, annos in video_annos.items():
        sample_count = video_infos[video]["sample_count"]
        if sample_count <= clip_length:
            offsets = [0]
        else:
            offsets = list(range(0, sample_count - clip_length + 1, stride))
            if (sample_count - clip_length) % stride:
                offsets.append(sample_count - clip_length)
        for offset in offsets:
            left, right = offset + 1, offset + clip_length
            keep = False
            for start, end, _ in annos:
                inter = max(0.0, min(right, end) - max(left, start))
                if inter / max(end - start, 1.0) >= 0.5:
                    keep = True
                    break
            if keep:
                clips.append(ClipSpec(video=video, offset=offset))
    return clips


class WiFiTADClips(Dataset):
    """WiFiTAD clip loader with a normalized [C, T] sensor signal interface."""

    def __init__(
        self,
        info_path: str | Path,
        anno_path: str | Path,
        data_path: str | Path,
        class_index_path: str | Path,
        clip_length: int,
        clip_stride: int,
    ) -> None:
        self.data_path = Path(data_path)
        self.clip_length = int(clip_length)
        self.origin_to_dense, self.dense_to_name = load_class_index(class_index_path)
        self.video_infos = _read_info(info_path)
        self.video_annos = _read_annos(anno_path, self.video_infos, self.origin_to_dense)
        self.clips = _make_clips(
            self.video_infos,
            self.video_annos,
            clip_length=self.clip_length,
            stride=int(clip_stride),
        )
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.clips)

    def _load_video(self, video: str) -> np.ndarray:
        if video not in self._cache:
            x = np.load(self.data_path / f"{video}.npy").astype(np.float32)
            if x.shape[0] < x.shape[-1]:
                x = x.T
            self._cache[video] = x / 40.0
        return self._cache[video]

    def __getitem__(self, index: int):
        spec = self.clips[index]
        video = self._load_video(spec.video)
        clip = video[spec.offset : spec.offset + self.clip_length]
        if clip.shape[0] < self.clip_length:
            pad = self.clip_length - clip.shape[0]
            clip = np.pad(clip, ((0, pad), (0, 0)), mode="constant")

        targets = []
        for start, end, cls in self.video_annos[spec.video]:
            start -= spec.offset
            end -= spec.offset
            if end <= 0 or start >= self.clip_length:
                continue
            targets.append([
                max(start, 1.0) / self.clip_length,
                min(end, float(self.clip_length)) / self.clip_length,
                float(cls),
            ])
        if not targets:
            targets = [[0.0, 0.0, 0.0]]

        signal = torch.from_numpy(clip.T.copy()).float()
        target = torch.tensor(targets, dtype=torch.float32)
        return signal, target, {"video": spec.video, "offset": spec.offset}


def detection_collate(batch):
    signals = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    meta = [item[2] for item in batch]
    return signals, targets, meta

