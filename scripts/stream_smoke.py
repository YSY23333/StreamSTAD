from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stadstream.config import load_config, resolve_path
from stadstream.data.wifitad import WiFiTADClips
from stadstream.models import build_model
from stadstream.online import ChunkOnlineDetector, SlidingWindowOnlineDetector


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wifitad_mini.yaml")
    parser.add_argument(
        "--model",
        default="dpwit",
        choices=["dpwit", "slimstad", "matr_signal", "moad_signal"],
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--max-chunks", type=int, default=4)
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    base = cfg["_config_dir"]
    ds_cfg = cfg["dataset"]["test"]
    dataset = WiFiTADClips(
        info_path=resolve_path(ds_cfg["info_path"], base),
        anno_path=resolve_path(ds_cfg["anno_path"], base),
        data_path=resolve_path(ds_cfg["data_path"], base),
        class_index_path=resolve_path(cfg["dataset"]["class_index_path"], base),
        clip_length=ds_cfg["clip_length"],
        clip_stride=ds_cfg["clip_stride"],
    )

    device = get_device(cfg.get("device", "auto"))
    model = build_model(
        args.model,
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["dataset"]["num_classes"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_anchors=int(cfg["model"]["num_anchors"]),
    ).to(device)

    signal, _, meta = dataset[0]
    if args.model in {"matr_signal", "moad_signal"}:
        online = ChunkOnlineDetector(
            model=model,
            score_thresh=args.score_thresh,
            top_k=5,
            device=device,
        )
        online_kind = "chunk-native"
    else:
        online = SlidingWindowOnlineDetector(
            model=model,
            window_size=args.window_size,
            stride=args.chunk_size,
            score_thresh=args.score_thresh,
            top_k=5,
            device=device,
        )
        online_kind = "sliding-window"
    online.reset_stream(video=meta["video"])

    total_props = 0
    max_t = min(signal.shape[-1], args.max_chunks * args.chunk_size)
    for start in range(0, max_t, args.chunk_size):
        chunk = signal[:, start : start + args.chunk_size]
        proposals = online.stream_step(chunk, video=meta["video"])
        total_props += len(proposals)
        buffer_start = getattr(online, "buffer_start", start)
        buffer = getattr(online, "buffer", chunk.unsqueeze(0))
        print(
            f"chunk_end={start + chunk.shape[-1]} "
            f"online={online_kind} "
            f"buffer_start={buffer_start} "
            f"buffer_len={buffer.shape[-1]} "
            f"new_props={len(proposals)}"
        )
        for prop in proposals[:2]:
            print(
                f"  label={prop.label} score={prop.score:.3f} "
                f"seg=[{prop.start:.1f}, {prop.end:.1f}] "
                f"output={prop.output_time} latency={prop.latency:.1f}"
            )

    print(f"stream_smoke_ok model={args.model} total_new_props={total_props}")


if __name__ == "__main__":
    main()
