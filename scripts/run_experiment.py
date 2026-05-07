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
from stadstream.data.wifitad import WiFiTADClips, detection_collate, load_class_index
from stadstream.evaluation import load_ground_truth, load_predictions, summarize_streaming
from stadstream.models import build_model
from stadstream.online import ChunkOnlineDetector, SlidingWindowOnlineDetector
from stadstream.training.losses import detection_loss


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg: dict, model_name: str, device: torch.device) -> torch.nn.Module:
    model = build_model(
        model_name,
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["dataset"]["num_classes"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_anchors=int(cfg["model"]["num_anchors"]),
    )
    return model.to(device)


def run_dir(cfg: dict, model_name: str, override: str | None = None) -> Path:
    if override:
        out = Path(override)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = ROOT / "runs" / f"{model_name}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def train(cfg: dict, model_name: str, out_dir: Path, device: torch.device) -> Path:
    base = cfg["_config_dir"]
    ds_cfg = cfg["dataset"]["train"]
    dataset = WiFiTADClips(
        info_path=resolve_path(ds_cfg["info_path"], base),
        anno_path=resolve_path(ds_cfg["anno_path"], base),
        data_path=resolve_path(ds_cfg["data_path"], base),
        class_index_path=resolve_path(cfg["dataset"]["class_index_path"], base),
        clip_length=ds_cfg["clip_length"],
        clip_stride=ds_cfg["clip_stride"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        drop_last=True,
    )
    model = build(cfg, model_name, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    best_path = out_dir / "checkpoint_last.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"total": 0.0, "cls": 0.0, "loc": 0.0}
        seen = 0
        for signals, targets, _ in tqdm(loader, desc=f"train {model_name} epoch {epoch}/{epochs}"):
            signals = signals.to(device)
            outputs = model(signals)
            losses = detection_loss(outputs, targets, num_classes=int(cfg["dataset"]["num_classes"]))
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch = signals.shape[0]
            seen += batch
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch

        row = {"epoch": epoch, **{k: v / max(seen, 1) for k, v in totals.items()}}
        history.append(row)
        print(
            f"epoch={epoch} total={row['total']:.4f} "
            f"cls={row['cls']:.4f} loc={row['loc']:.4f}"
        )
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": model_name,
                "config": cfg,
                "epoch": epoch,
            },
            best_path,
        )

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    return best_path


def make_online(model: torch.nn.Module, model_name: str, cfg: dict, device: torch.device):
    stream_cfg = cfg.get("streaming", {})
    score_thresh = float(stream_cfg.get("score_thresh", 0.05))
    top_k = int(stream_cfg.get("top_k", 20))
    if model_name in {"matr_signal", "moad_signal"}:
        return ChunkOnlineDetector(model, score_thresh=score_thresh, top_k=top_k, device=device)
    return SlidingWindowOnlineDetector(
        model,
        window_size=int(stream_cfg.get("window_size", cfg["dataset"]["test"]["clip_length"])),
        stride=int(stream_cfg.get("chunk_size", 256)),
        score_thresh=score_thresh,
        top_k=top_k,
        device=device,
    )


def load_signal(path: Path) -> torch.Tensor:
    x = np.load(path).astype(np.float32)
    if x.shape[0] < x.shape[-1]:
        x = x.T
    return torch.from_numpy((x / 40.0).T.copy()).float()


def stream_test(
    cfg: dict,
    model_name: str,
    out_dir: Path,
    device: torch.device,
    checkpoint: str | Path | None,
) -> Path:
    base = cfg["_config_dir"]
    model = build(cfg, model_name, device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
    online = make_online(model, model_name, cfg, device)

    ds_cfg = cfg["dataset"]["test"]
    info_path = resolve_path(ds_cfg["info_path"], base)
    data_path = resolve_path(ds_cfg["data_path"], base)
    chunk_size = int(cfg.get("streaming", {}).get("chunk_size", 256))
    info = pd.read_csv(info_path)

    results: dict[str, list[dict]] = {}
    for _, row in tqdm(info.iterrows(), total=len(info), desc=f"stream_test {model_name}"):
        video = str(row.video)
        signal = load_signal(data_path / f"{video}.npy")
        online.reset_stream(video=video)
        props = []
        for start in range(0, signal.shape[-1], chunk_size):
            chunk = signal[:, start : start + chunk_size]
            props.extend(online.stream_step(chunk, video=video))
        results[video] = [
            {
                "label": p.label,
                "score": p.score,
                "segment": [p.start, p.end],
                "output_time": p.output_time,
                "latency": p.latency,
            }
            for p in online.emitted
        ]

    pred_path = out_dir / f"{model_name}_stream_predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "STADStream",
                "model": model_name,
                "results": results,
                "external_data": {},
            },
            f,
            indent=2,
        )
    print(f"saved_predictions={pred_path}")
    return pred_path


def evaluate(cfg: dict, pred_path: str | Path, out_dir: Path) -> Path:
    base = cfg["_config_dir"]
    origin_to_dense, _ = load_class_index(resolve_path(cfg["dataset"]["class_index_path"], base))
    gt = load_ground_truth(resolve_path(cfg["dataset"]["test"]["anno_path"], base), origin_to_dense)
    pred = load_predictions(pred_path)
    eval_cfg = cfg.get("evaluation", {})
    tious = [float(x) for x in eval_cfg.get("tious", [0.3, 0.4, 0.5, 0.6, 0.7])]
    latency_budgets = [float(x) for x in eval_cfg.get("latency_budgets", [50, 100, 200, 500, 1000])]
    metrics = summarize_streaming(pred, gt, tious=tious, latency_budgets=latency_budgets)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved_metrics={metrics_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wifitad_mini.yaml")
    parser.add_argument(
        "--model",
        default="slimstad",
        choices=["dpwit", "slimstad", "matr_signal", "moad_signal"],
    )
    parser.add_argument(
        "--mode",
        default="train_test_eval",
        choices=["train", "stream_test", "eval", "train_test_eval"],
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pred", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 2020)))
    device = get_device(cfg.get("device", "auto"))
    out_dir = run_dir(cfg, args.model, args.run_dir)
    print(f"model={args.model} mode={args.mode} device={device} run_dir={out_dir}")

    ckpt = args.checkpoint
    pred = args.pred
    if args.mode in {"train", "train_test_eval"}:
        ckpt = train(cfg, args.model, out_dir, device)
    if args.mode in {"stream_test", "train_test_eval"}:
        pred = stream_test(cfg, args.model, out_dir, device, ckpt)
    if args.mode in {"eval", "train_test_eval"}:
        if pred is None:
            raise ValueError("--pred is required for eval mode")
        evaluate(cfg, pred, out_dir)


if __name__ == "__main__":
    main()

