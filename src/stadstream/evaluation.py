from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def load_ground_truth(
    anno_path: str | Path,
    origin_to_dense: dict[int, int],
) -> dict[str, list[dict[str, Any]]]:
    df = pd.read_csv(anno_path)
    gt: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        video = str(row.video)
        gt.setdefault(video, []).append(
            {
                "segment": [float(row.startFrame), float(row.endFrame)],
                "label": int(origin_to_dense[int(row.type_idx)]),
            }
        )
    return gt


def load_predictions(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["results"]


def evaluate_map(
    predictions: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, list[dict[str, Any]]],
    tious: list[float],
    latency_budget: float | None = None,
) -> dict[str, float]:
    if latency_budget is not None:
        predictions = {
            video: [p for p in props if float(p.get("latency", 0.0)) <= latency_budget]
            for video, props in predictions.items()
        }

    classes = sorted({int(g["label"]) for props in ground_truth.values() for g in props})
    metrics: dict[str, float] = {}
    for tiou in tious:
        aps: list[float] = []
        for cls in classes:
            cls_gt: list[tuple[str, float, float]] = []
            cls_pred: list[tuple[str, float, float, float]] = []
            for video, props in ground_truth.items():
                for gt in props:
                    if int(gt["label"]) == cls:
                        cls_gt.append((video, float(gt["segment"][0]), float(gt["segment"][1])))
            for video, props in predictions.items():
                for pred in props:
                    if int(pred["label"]) == cls:
                        cls_pred.append(
                            (
                                video,
                                float(pred["segment"][0]),
                                float(pred["segment"][1]),
                                float(pred["score"]),
                            )
                        )

            if not cls_gt:
                continue
            if not cls_pred:
                aps.append(0.0)
                continue

            cls_pred.sort(key=lambda item: item[3], reverse=True)
            matched = np.zeros(len(cls_gt), dtype=bool)
            tp = np.zeros(len(cls_pred), dtype=np.float32)
            fp = np.zeros(len(cls_pred), dtype=np.float32)
            for i, (video, start, end, _) in enumerate(cls_pred):
                best_iou = 0.0
                best_j = -1
                for j, (gt_video, gt_start, gt_end) in enumerate(cls_gt):
                    if matched[j] or video != gt_video:
                        continue
                    iou = interval_iou((start, end), (gt_start, gt_end))
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= tiou and best_j >= 0:
                    tp[i] = 1
                    matched[best_j] = True
                else:
                    fp[i] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(len(cls_gt), 1)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            aps.append(voc_ap(rec, prec))
        metrics[f"mAP@{tiou:.2f}"] = float(np.mean(aps)) if aps else 0.0
    metrics["mAP@Avg"] = float(np.mean([metrics[f"mAP@{t:.2f}"] for t in tious]))
    return metrics


def summarize_streaming(
    predictions: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, list[dict[str, Any]]],
    tious: list[float],
    latency_budgets: list[float],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "standard": evaluate_map(predictions, ground_truth, tious),
        "latency": {},
    }
    for budget in latency_budgets:
        out["latency"][str(budget)] = evaluate_map(
            predictions,
            ground_truth,
            tious,
            latency_budget=budget,
        )
    return out

