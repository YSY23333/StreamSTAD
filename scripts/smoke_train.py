from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stadstream.config import load_config, resolve_path
from stadstream.data.wifitad import WiFiTADClips, detection_collate
from stadstream.models import build_model
from stadstream.training.losses import detection_loss


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wifitad_mini.yaml")
    parser.add_argument("--model", default="dpwit", choices=["dpwit", "slimstad", "matr_signal", "moad_signal"])
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    base = cfg["_config_dir"]
    seed = int(cfg.get("seed", 2020))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    device = get_device(cfg.get("device", "auto"))
    model = build_model(
        args.model,
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["dataset"]["num_classes"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_anchors=int(cfg["model"]["num_anchors"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    print(f"model={args.model} device={device} clips={len(dataset)}")
    model.train()
    last = None
    for step, (signals, targets, meta) in enumerate(tqdm(loader, total=min(len(loader), args.max_batches))):
        if step >= args.max_batches:
            break
        signals = signals.to(device)
        outputs = model(signals)
        losses = detection_loss(outputs, targets, num_classes=int(cfg["dataset"]["num_classes"]))
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        last = {k: float(v.detach().cpu()) for k, v in losses.items()}
        print(
            f"step={step + 1} total={last['total']:.4f} "
            f"cls={last['cls']:.4f} loc={last['loc']:.4f} "
            f"loc_shape={tuple(outputs['loc'].shape)} conf_shape={tuple(outputs['conf'].shape)}"
        )

    if last is None:
        raise RuntimeError("No training batch was produced. Check dataset paths and clip filtering.")
    print("smoke_train_ok")


if __name__ == "__main__":
    main()
