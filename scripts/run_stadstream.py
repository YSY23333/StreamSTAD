from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stadstream.config import load_config, resolve_path
from stadstream.data.stream_windows import StreamWindowDataset, stream_window_collate
from stadstream.data.wifitad import load_class_index
from stadstream.evaluation import load_ground_truth, summarize_streaming
from stadstream.models.stadstream import STADStreamNet
from stadstream.training.stadstream_loss import stadstream_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_dataset(cfg: dict, split: str, window_size: int, stride: int, overlap_thresh: float):
    base = cfg["_config_dir"]
    ds = cfg["dataset"]["train" if split == "train" else "test"]
    return StreamWindowDataset(
        info_path=resolve_path(ds["info_path"], base),
        anno_path=resolve_path(ds["anno_path"], base),
        data_path=resolve_path(ds["data_path"], base),
        class_index_path=resolve_path(cfg["dataset"]["class_index_path"], base),
        window_size=window_size,
        stride=stride,
        num_classes=int(cfg["dataset"]["num_classes"]),
        overlap_thresh=overlap_thresh,
        training=(split == "train"),
    )


def train(cfg: dict, model: STADStreamNet, loader: DataLoader, out_dir: Path, device: torch.device) -> Path:
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        totals = {}
        seen = 0
        for clips, targets, _ in tqdm(loader, desc=f"stadstream train {epoch}"):
            clips = clips.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(clips)
            losses = stadstream_loss(outputs, targets, int(cfg["dataset"]["num_classes"]))
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch = clips.shape[0]
            seen += batch
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * batch
        row = {"epoch": epoch, **{k: v / max(seen, 1) for k, v in totals.items()}}
        history.append(row)
        print(" ".join([f"epoch={epoch}"] + [f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"]))
        torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, out_dir / "checkpoint_last.pt")
    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    return out_dir / "checkpoint_last.pt"


def merge_proposals(props: list[dict], iou_thresh: float = 0.7) -> list[dict]:
    out: list[dict] = []
    for prop in sorted(props, key=lambda x: x["score"], reverse=True):
        keep = True
        for old in out:
            if old["label"] != prop["label"]:
                continue
            inter = max(0.0, min(old["segment"][1], prop["segment"][1]) - max(old["segment"][0], prop["segment"][0]))
            union = max(old["segment"][1], prop["segment"][1]) - min(old["segment"][0], prop["segment"][0])
            if union > 0 and inter / union > iou_thresh:
                keep = False
                break
        if keep:
            out.append(prop)
    return sorted(out, key=lambda x: x["segment"][0])


@torch.no_grad()
def stream_test(cfg: dict, model: STADStreamNet, dataset: StreamWindowDataset, out_dir: Path, device: torch.device, score_thresh: float) -> Path:
    model.eval()
    results: dict[str, list[dict]] = {video: [] for video in dataset.video_len}
    for clip, _, meta in tqdm(dataset, desc="stadstream test"):
        video = meta["video"]
        win_end = int(meta["window_end"])
        out = model(clip.unsqueeze(0).to(device))
        action = torch.sigmoid(out["actionness_logit"])[0].item()
        probs = torch.softmax(out["cls_logits"], dim=-1)[0]
        score, label = probs[1:].max(dim=0)
        label = int(label.item()) + 1
        score = float(score.item() * action)
        if score < score_thresh:
            continue
        offsets = out["offsets"][0].detach().cpu().numpy()
        start = float(win_end - offsets[0] * dataset.window_size)
        end = float(win_end - offsets[1] * dataset.window_size)
        if end <= start:
            end = start + 1.0
        start = max(0.0, start)
        end = min(float(dataset.video_len[video]), end)
        results[video].append(
            {
                "label": label,
                "score": score,
                "segment": [start, end],
                "output_time": win_end,
                "latency": max(0.0, win_end - end),
            }
        )
    results = {video: merge_proposals(props) for video, props in results.items()}
    pred_path = out_dir / "stadstream_predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump({"version": "STADStream-v1", "results": results, "external_data": {}}, f, indent=2)
    print(f"saved_predictions={pred_path}")
    return pred_path


def evaluate(cfg: dict, pred_path: Path, out_dir: Path) -> None:
    base = cfg["_config_dir"]
    origin_to_dense, _ = load_class_index(resolve_path(cfg["dataset"]["class_index_path"], base))
    gt = load_ground_truth(resolve_path(cfg["dataset"]["test"]["anno_path"], base), origin_to_dense)
    with pred_path.open("r", encoding="utf-8") as f:
        pred = json.load(f)["results"]
    eval_cfg = cfg.get("evaluation", {})
    metrics = summarize_streaming(
        pred,
        gt,
        [float(x) for x in eval_cfg.get("tious", [0.3, 0.4, 0.5, 0.6, 0.7])],
        [float(x) for x in eval_cfg.get("latency_budgets", [50, 100, 200, 500, 1000])],
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wifitad_mini.yaml")
    parser.add_argument("--mode", default="train_test_eval", choices=["train", "stream_test", "eval", "train_test_eval"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pred", default=None)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--overlap-thresh", type=float, default=0.15)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 2020)))
    device = device_from_config(cfg.get("device", "auto"))
    out_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / f"stadstream-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = STADStreamNet(int(cfg["model"]["in_channels"]), int(cfg["dataset"]["num_classes"]), int(cfg["model"]["hidden_dim"])).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])

    train_ds = build_dataset(cfg, "train", args.window_size, args.stride, args.overlap_thresh)
    test_ds = build_dataset(cfg, "test", args.window_size, args.stride, args.overlap_thresh)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0, collate_fn=stream_window_collate)

    ckpt_path = args.checkpoint
    pred_path = Path(args.pred) if args.pred else None
    print(f"STADStream device={device} train_windows={len(train_ds)} test_windows={len(test_ds)}")
    if args.mode in {"train", "train_test_eval"}:
        ckpt_path = train(cfg, model, train_loader, out_dir, device)
    if args.mode in {"stream_test", "train_test_eval"}:
        pred_path = stream_test(cfg, model, test_ds, out_dir, device, args.score_thresh)
    if args.mode in {"eval", "train_test_eval"}:
        if pred_path is None:
            raise ValueError("--pred is required for eval mode")
        evaluate(cfg, Path(pred_path), out_dir)


if __name__ == "__main__":
    main()

