from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stadstream.config import load_config
from stadstream.models.faithful_matr import SensorToMATRFeatureAdapter, check_matr_import


def default_matr_args(feat_dim: int, num_frame: int, num_classes: int) -> SimpleNamespace:
    """Minimal args object matching official MATR model constructor fields."""
    return SimpleNamespace(
        training=True,
        feat_dim=feat_dim,
        num_of_class=num_classes,
        hidden_dim=128,
        enc_layers=1,
        dec_layers=1,
        e_nheads=4,
        d_nheads=4,
        ffn_dim=256,
        num_frame=num_frame,
        num_queries=10,
        rgb=True,
        flow=False,
        dropout=0.1,
        drop_rate=0.1,
        use_flag=False,
        flag_threshold=0.5,
        max_memory_len=4,
        memory_sampler="gap2",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matr-root", default="../MATR_codebase")
    parser.add_argument("--config", default="configs/wifitad_mini.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--num-frame", type=int, default=64)
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    status = check_matr_import((ROOT / args.matr_root).resolve() if not Path(args.matr_root).is_absolute() else args.matr_root)
    print(status)
    if not status.ok:
        raise SystemExit(1)

    from models import build_model as build_official_matr

    adapter = SensorToMATRFeatureAdapter(
        in_channels=int(cfg["model"]["in_channels"]),
        feat_dim=args.feat_dim,
        num_frame=args.num_frame,
    )
    official_args = default_matr_args(
        feat_dim=args.feat_dim,
        num_frame=args.num_frame,
        num_classes=int(cfg["dataset"]["num_classes"]),
    )
    matr = build_official_matr(official_args)
    signal = torch.randn(args.batch_size, int(cfg["model"]["in_channels"]), args.timesteps)
    features = adapter(signal)
    inputs = {
        "inputs": features,
        "infos": {
            "st": torch.zeros(args.batch_size),
            "ed": torch.full((args.batch_size,), args.num_frame),
            "video_name": ["smoke"] * args.batch_size,
            "current_frame": torch.zeros(args.batch_size),
            "segment_flag": torch.ones(args.batch_size, dtype=torch.long),
        },
    }
    with torch.no_grad():
        out = matr(inputs, device=torch.device("cpu"))
    print("faithful_matr_smoke_ok")
    print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})


if __name__ == "__main__":
    main()

