from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class OnlineProposal:
    video: str
    start: float
    end: float
    label: int
    score: float
    output_time: int
    latency: float


class SlidingWindowOnlineDetector:
    """Turn an offline anchor detector into a past-only streaming detector.

    The wrapped model only receives the current chunk plus historical context in
    ``self.buffer``. This is a pseudo-online baseline for offline models such as
    DPWiT and SlimSTAD: it is causal at inference time even if the detector
    architecture was originally trained on full clips.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window_size: int = 4096,
        stride: int = 256,
        score_thresh: float = 0.3,
        top_k: int = 20,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.score_thresh = float(score_thresh)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.reset_stream()

    def reset_stream(self, video: str = "") -> None:
        if hasattr(self.model, "reset_stream"):
            self.model.reset_stream()
        self.video = video
        self.buffer: torch.Tensor | None = None
        self.buffer_start = 0
        self.current_time = 0
        self.emitted: list[OnlineProposal] = []

    @torch.no_grad()
    def stream_step(self, chunk: torch.Tensor, video: str | None = None) -> list[OnlineProposal]:
        """Process one incoming chunk.

        Args:
            chunk: Tensor shaped [C, T] or [1, C, T].
            video: Optional video id stored in emitted proposals.
        """
        if video is not None and video != self.video:
            self.reset_stream(video)
        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(0)
        chunk = chunk.to(self.device)
        chunk_len = int(chunk.shape[-1])

        if self.buffer is None:
            self.buffer = chunk
            self.buffer_start = self.current_time
        else:
            self.buffer = torch.cat([self.buffer, chunk], dim=-1)
        if self.buffer.shape[-1] > self.window_size:
            overflow = self.buffer.shape[-1] - self.window_size
            self.buffer = self.buffer[..., overflow:]
            self.buffer_start += overflow

        self.current_time += chunk_len
        self.model.eval()
        outputs = self.model(self.buffer)
        proposals = self._decode(outputs)
        new_props = self._dedupe(proposals)
        self.emitted.extend(new_props)
        return new_props

    def _decode(self, outputs: dict[str, torch.Tensor]) -> list[OnlineProposal]:
        loc = outputs["loc"][0]
        conf = F.softmax(outputs["conf"][0], dim=-1)
        priors = outputs["priors"][0, :, 0]
        window_len = int(self.buffer.shape[-1]) if self.buffer is not None else self.window_size

        scores, labels = conf[:, 1:].max(dim=-1)
        keep = scores > self.score_thresh
        if keep.sum() == 0:
            return []
        kept_scores = scores[keep]
        kept_labels = labels[keep] + 1
        kept_loc = loc[keep]
        kept_priors = priors[keep]
        order = kept_scores.argsort(descending=True)[: self.top_k]

        proposals: list[OnlineProposal] = []
        for idx in order:
            center = float(kept_priors[idx].item() * window_len + self.buffer_start)
            start = center - float(kept_loc[idx, 0].item() * window_len)
            end = center + float(kept_loc[idx, 1].item() * window_len)
            start = max(float(self.buffer_start), start)
            end = min(float(self.current_time), max(start + 1.0, end))
            output_time = self.current_time
            proposals.append(
                OnlineProposal(
                    video=self.video,
                    start=start,
                    end=end,
                    label=int(kept_labels[idx].item()),
                    score=float(kept_scores[idx].item()),
                    output_time=output_time,
                    latency=max(0.0, float(output_time) - end),
                )
            )
        return proposals

    def _dedupe(self, proposals: list[OnlineProposal]) -> list[OnlineProposal]:
        fresh: list[OnlineProposal] = []
        for prop in proposals:
            duplicate = False
            for old in self.emitted[-100:]:
                if old.label != prop.label:
                    continue
                inter = max(0.0, min(old.end, prop.end) - max(old.start, prop.start))
                union = max(old.end, prop.end) - min(old.start, prop.start)
                if union > 0 and inter / union > 0.7:
                    duplicate = True
                    break
            if not duplicate:
                fresh.append(prop)
        return fresh


class ChunkOnlineDetector:
    """Native chunk streaming wrapper for online models with internal memory.

    MATR/MOAD-style baselines should not receive the whole historical window on
    every step. They receive only the newly arrived chunk and keep history in
    their own memory modules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        score_thresh: float = 0.3,
        top_k: int = 20,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.score_thresh = float(score_thresh)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self.reset_stream()

    def reset_stream(self, video: str = "") -> None:
        if hasattr(self.model, "reset_stream"):
            self.model.reset_stream()
        self.video = video
        self.current_time = 0
        self.emitted: list[OnlineProposal] = []

    @torch.no_grad()
    def stream_step(self, chunk: torch.Tensor, video: str | None = None) -> list[OnlineProposal]:
        if video is not None and video != self.video:
            self.reset_stream(video)
        if chunk.dim() == 2:
            chunk = chunk.unsqueeze(0)
        chunk = chunk.to(self.device)
        chunk_len = int(chunk.shape[-1])
        span_start = self.current_time
        self.current_time += chunk_len

        self.model.eval()
        outputs = self.model(chunk)
        proposals = self._decode(outputs, span_start=span_start, span_len=chunk_len)
        new_props = self._dedupe(proposals)
        self.emitted.extend(new_props)
        return new_props

    def _decode(
        self,
        outputs: dict[str, torch.Tensor],
        span_start: int,
        span_len: int,
    ) -> list[OnlineProposal]:
        loc = outputs["loc"][0]
        conf = F.softmax(outputs["conf"][0], dim=-1)
        priors = outputs["priors"][0, :, 0]

        scores, labels = conf[:, 1:].max(dim=-1)
        keep = scores > self.score_thresh
        if keep.sum() == 0:
            return []
        kept_scores = scores[keep]
        kept_labels = labels[keep] + 1
        kept_loc = loc[keep]
        kept_priors = priors[keep]
        order = kept_scores.argsort(descending=True)[: self.top_k]

        proposals: list[OnlineProposal] = []
        for idx in order:
            center = float(kept_priors[idx].item() * span_len + span_start)
            start = center - float(kept_loc[idx, 0].item() * span_len)
            end = center + float(kept_loc[idx, 1].item() * span_len)
            start = max(float(span_start), start)
            end = min(float(self.current_time), max(start + 1.0, end))
            output_time = self.current_time
            proposals.append(
                OnlineProposal(
                    video=self.video,
                    start=start,
                    end=end,
                    label=int(kept_labels[idx].item()),
                    score=float(kept_scores[idx].item()),
                    output_time=output_time,
                    latency=max(0.0, float(output_time) - end),
                )
            )
        return proposals

    def _dedupe(self, proposals: list[OnlineProposal]) -> list[OnlineProposal]:
        fresh: list[OnlineProposal] = []
        for prop in proposals:
            duplicate = False
            for old in self.emitted[-100:]:
                if old.label != prop.label:
                    continue
                inter = max(0.0, min(old.end, prop.end) - max(old.start, prop.start))
                union = max(old.end, prop.end) - min(old.start, prop.start)
                if union > 0 and inter / union > 0.7:
                    duplicate = True
                    break
            if not duplicate:
                fresh.append(prop)
        return fresh
