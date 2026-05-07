from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stadstream.config import load_config, resolve_path
from stadstream.data.wifitad import load_class_index


def add_matr_root(matr_root: str | Path) -> Path:
    root = Path(matr_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"MATR root does not exist: {root}")
    required = [root / "models", root / "criterion", root / "util", root / "eval.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"MATR root is missing official files: {missing}")
    for name in [
        "criterion",
        "criterion.criterion",
        "criterion.matcher",
        "models",
        "models.models",
        "models.transformer",
        "util",
        "util.utils",
        "eval",
    ]:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "Evaluation"))
    return root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_official_args(cfg: dict, run_dir: Path, gt_json: Path, mode: str, args) -> SimpleNamespace:
    num_classes = int(cfg["dataset"]["num_classes"])
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    stream_cfg = cfg.get("streaming", {})
    return SimpleNamespace(
        video_anno=str(gt_json),
        video_len_file=str(run_dir / "matr_video_len_{}.json"),
        num_of_class=num_classes,
        num_frame=int(args.num_frame),
        rgb=True,
        flow=False,
        video_feature_all_train="",
        video_feature_all_test="",
        ontal_label_file=str(run_dir / "matr_label_{}_{}_{}_{}_{}_{}_{}.h5"),
        num_queries=int(args.num_queries),
        p_videos=1,
        detect_len=int(args.detect_len),
        anti_len=int(args.anti_len),
        dropout=float(args.dropout),
        training=(mode == "train"),
        feat_dim=int(args.feat_dim),
        sensor_input=True,
        sensor_in_channels=int(model_cfg["in_channels"]),
        hidden_dim=int(args.hidden_dim),
        ffn_dim=int(args.ffn_dim),
        e_nheads=int(args.e_nheads),
        enc_layers=int(args.enc_layers),
        d_nheads=int(args.d_nheads),
        dec_layers=int(args.dec_layers),
        pre_norm=False,
        activation="gelu",
        max_memory_len=int(args.max_memory_len),
        memory_sampler=args.memory_sampler,
        use_flag=bool(args.use_flag),
        flag_threshold=float(args.flag_threshold),
        epochs=int(train_cfg["epochs"]),
        batch=int(train_cfg["batch_size"]),
        mode=mode,
        min_lr=float(args.min_lr),
        max_lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        lr_gamma=0.9,
        lr_Tup=3,
        lr_Tcycle=10,
        test_freq=1,
        save_freq=1,
        drop_rate=float(args.dropout),
        use_empty_weight=False,
        eos_coef=1e-8,
        use_focal=False,
        reduce=1,
        cls_threshold=float(stream_cfg.get("score_thresh", 0.05)),
        cls_coef=1,
        flag_coef=1,
        reg_l1_coef=1,
        reg_diou_coef=1,
        reg_stcls_coef=1,
        nms_threshold=float(args.nms_threshold),
        make_output=True,
        proposal_path="proposal_{}_{}_{}",
        wandb=False,
        code_testing=False,
        task="ontal",
        dataset="thumos14",
        random_seed=int(cfg.get("seed", 2020)),
        save_path=str(run_dir),
        load_model=False,
        model_path=None,
        train_eval_step=1,
        test_eval_step=1,
        device="0",
        num_workers=0,
    )


def patch_official_matr_sensor_encoder(model: nn.Module, in_channels: int) -> nn.Module:
    """Attach a sensor feature encoder to official MATR without touching other logic."""
    model.sensor_input = True
    model.sensor_in_channels = int(in_channels)
    model.sensor_feature_encoder = nn.Sequential(
        nn.Conv1d(in_channels, model.n_embedding_dim, kernel_size=1),
        nn.GELU(),
        nn.Conv1d(model.n_embedding_dim, model.n_embedding_dim, kernel_size=3, padding=1),
    )
    old_input_projection = model.input_projection

    def input_projection(self, inputs):
        if getattr(self, "sensor_input", False):
            if inputs.dim() != 3:
                raise ValueError(f"sensor input expects [B, C, T], got {tuple(inputs.shape)}")
            if inputs.shape[1] != self.sensor_in_channels:
                raise ValueError(
                    f"sensor input channel mismatch: expected {self.sensor_in_channels}, got {inputs.shape[1]}"
                )
            base_x = self.sensor_feature_encoder(inputs.float())
            base_x = torch.nn.functional.adaptive_avg_pool1d(base_x, self.n_seglen)
            return base_x.permute([2, 0, 1])
        return old_input_projection(inputs)

    model.input_projection = MethodType(input_projection, model)
    return model


def ensure_matr_position_capacity(model: nn.Module, num_frame: int) -> None:
    """Expand official MATR positional encodings when sensor windows are long."""
    if model.segment_pos_encoding.pos_embedding.shape[0] < num_frame:
        emb = model.n_embedding_dim
        den = torch.exp(-torch.arange(0, emb, 2) * torch.log(torch.tensor(10000.0)) / emb)
        pos = torch.arange(0, num_frame).reshape(num_frame, 1)
        pos_embedding = torch.zeros((num_frame, emb), device=model.segment_pos_encoding.pos_embedding.device)
        pos_embedding[:, 0::2] = torch.sin(pos.to(pos_embedding.device) * den.to(pos_embedding.device))
        pos_embedding[:, 1::2] = torch.cos(pos.to(pos_embedding.device) * den.to(pos_embedding.device))
        model.segment_pos_encoding.pos_embedding = pos_embedding.unsqueeze(-2)
        model.segment_pos_encoding.position_ids = torch.arange(num_frame, device=pos_embedding.device).expand((1, -1))

    if model.memory_pos_encoding.pos_embedding.shape[0] < num_frame:
        maxlen = max(num_frame, model.max_memory_len + 3)
        emb = model.n_embedding_dim // 2
        den = torch.exp(-torch.arange(0, emb, 2) * torch.log(torch.tensor(10000.0)) / emb)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb), device=model.memory_pos_encoding.pos_embedding.device)
        pos_embedding[:, 0::2] = torch.sin(pos.to(pos_embedding.device) * den.to(pos_embedding.device))
        pos_embedding[:, 1::2] = torch.cos(pos.to(pos_embedding.device) * den.to(pos_embedding.device))
        model.memory_pos_encoding.pos_embedding = pos_embedding
        model.memory_pos_encoding.position_ids = torch.arange(maxlen, device=pos_embedding.device)


def read_info(path: Path) -> dict[str, dict[str, int]]:
    df = pd.read_csv(path)
    return {
        str(row.video): {
            "count": int(row["count"]),
            "sample_count": int(row.sample_count),
            "sample_fps": int(row.sample_fps),
        }
        for _, row in df.iterrows()
    }


def read_annos(path: Path, origin_to_dense: dict[int, int]) -> dict[str, list[dict]]:
    df = pd.read_csv(path)
    out: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        video = str(row.video)
        out.setdefault(video, []).append(
            {
                "start": float(row.startFrame),
                "end": float(row.endFrame),
                "class0": int(origin_to_dense[int(row.type_idx)]) - 1,
            }
        )
    return out


class WiFiTADMATRDataset(Dataset):
    """WiFiTAD sensor loader that returns official MATR target/info fields."""

    def __init__(
        self,
        info_path: Path,
        anno_path: Path,
        data_path: Path,
        class_index_path: Path,
        label_names: list[str],
        num_frame: int,
        num_queries: int,
        num_classes: int,
        detect_len: int,
        anti_len: int,
        max_memory_len: int,
        sample_stride: int,
        subset: str,
    ) -> None:
        self.info_path = info_path
        self.anno_path = anno_path
        self.data_path = data_path
        self.label_name = label_names
        self.num_frame = num_frame
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.detect_len = detect_len
        self.anti_len = anti_len
        self.max_memory_len = max_memory_len
        self.sample_stride = sample_stride
        self.subset = subset
        self.origin_to_dense, _ = load_class_index(class_index_path)
        self.video_len = {k: v["sample_count"] for k, v in read_info(info_path).items()}
        self.video_list = sorted(self.video_len)
        self.annos = read_annos(anno_path, self.origin_to_dense)
        self.video_dict = {
            video: {
                "duration": float(self.video_len[video]),
                "subset": subset,
                "annotations": [
                    {
                        "segment": [ann["start"], ann["end"]],
                        "label": self.label_name[ann["class0"]],
                    }
                    for ann in self.annos.get(video, [])
                ],
            }
            for video in self.video_list
        }
        self.samples = []
        for video in self.video_list:
            duration = self.video_len[video]
            for ed in range(1, duration + 1, self.sample_stride):
                self.samples.append((video, max(1, ed)))
            if self.samples[-1][1] != duration:
                self.samples.append((video, duration))
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

    def _labels(self, video: str, ed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_rows = []
        reg_rows = []
        stcls_rows = []
        target_sel = [ed - self.detect_len + 1, ed + self.anti_len]
        flag = 0.0
        for ann in self.annos.get(video, []):
            start, end, cls = ann["start"], ann["end"], ann["class0"]
            if start < ed and end > ed - self.num_frame:
                flag = 1.0
            if end > target_sel[0] and target_sel[1] > end:
                cls_vec = np.zeros(self.num_classes, dtype=np.float32)
                cls_vec[cls] = 1.0
                stcls = np.zeros(self.max_memory_len + 2, dtype=np.float32)
                st_off = ed - start
                ed_off = (ed - end) / self.num_frame
                st_sel = int(min(max(0, st_off // self.num_frame), self.max_memory_len + 1))
                stcls[st_sel] = 1.0
                reg = np.zeros(2, dtype=np.float32)
                reg[0] = st_off / self.num_frame - st_sel
                reg[1] = ed_off
                cls_rows.append(cls_vec)
                reg_rows.append(reg)
                stcls_rows.append(stcls)
        cls_rows = cls_rows[: self.num_queries]
        reg_rows = reg_rows[: self.num_queries]
        stcls_rows = stcls_rows[: self.num_queries]
        while len(cls_rows) < self.num_queries:
            cls_vec = np.zeros(self.num_classes, dtype=np.float32)
            cls_vec[-1] = 1.0
            reg = np.zeros(2, dtype=np.float32)
            reg[-1] = -1e4
            stcls = np.zeros(self.max_memory_len + 2, dtype=np.float32)
            cls_rows.append(cls_vec)
            reg_rows.append(reg)
            stcls_rows.append(stcls)
        return (
            torch.tensor(np.stack(cls_rows), dtype=torch.float32),
            torch.tensor(np.stack(reg_rows), dtype=torch.float32),
            torch.tensor(np.stack(stcls_rows), dtype=torch.float32),
            torch.tensor(flag, dtype=torch.float32),
        )

    def __getitem__(self, index: int):
        video, ed = self.samples[index]
        signal = self._load_signal(video)
        st = ed - self.num_frame
        clip = signal[:, max(0, st) : ed]
        if clip.shape[-1] < self.num_frame:
            pad = self.num_frame - clip.shape[-1]
            clip = torch.nn.functional.pad(clip, (pad, 0))
        cls, reg, stcls, flag = self._labels(video, ed)
        target = {"cls_label": cls, "reg_label": reg, "stcls_label": stcls}
        info = {
            "video_name": video,
            "current_frame": ed - 1,
            "ed": ed,
            "st": st,
            "duration": self.video_len[video],
            "video_time": float(self.video_len[video]),
            "frame_to_time": 1.0,
            "segment_flag": flag,
        }
        return clip, target, info


def matr_collate(batch):
    signals = torch.stack([x[0] for x in batch], dim=0)
    targets = {
        "cls_label": torch.stack([x[1]["cls_label"] for x in batch], dim=0),
        "reg_label": torch.stack([x[1]["reg_label"] for x in batch], dim=0),
        "stcls_label": torch.stack([x[1]["stcls_label"] for x in batch], dim=0),
    }
    infos = {}
    for key in batch[0][2]:
        vals = [x[2][key] for x in batch]
        if key == "video_name":
            infos[key] = vals
        else:
            infos[key] = torch.tensor(vals)
    return signals, targets, infos


def print_label_stats(dataset: WiFiTADMATRDataset, name: str) -> None:
    bg_idx = dataset.num_classes - 1
    counts = {label: 0 for label in dataset.label_name}
    positives = 0
    for i in range(len(dataset)):
        _, target, _ = dataset[i]
        for row in target["cls_label"]:
            cls = int(row.argmax().item())
            if cls == bg_idx:
                continue
            if 0 <= cls < len(dataset.label_name):
                counts[dataset.label_name[cls]] += 1
                positives += 1
    print(f"{name} samples: {len(dataset)}")
    print(f"{name} positive query rows: {positives}")
    print(f"{name} positive class distribution: {counts}")


def write_gt_json(dataset: WiFiTADMATRDataset, path: Path) -> None:
    payload = {"database": dataset.video_dict}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_dataset(cfg: dict, split: str, args, label_names: list[str]) -> WiFiTADMATRDataset:
    base = cfg["_config_dir"]
    ds_cfg = cfg["dataset"]["train" if split == "train" else "test"]
    return WiFiTADMATRDataset(
        info_path=resolve_path(ds_cfg["info_path"], base),
        anno_path=resolve_path(ds_cfg["anno_path"], base),
        data_path=resolve_path(ds_cfg["data_path"], base),
        class_index_path=resolve_path(cfg["dataset"]["class_index_path"], base),
        label_names=label_names,
        num_frame=args.num_frame,
        num_queries=args.num_queries,
        num_classes=int(cfg["dataset"]["num_classes"]),
        detect_len=args.detect_len,
        anti_len=args.anti_len,
        max_memory_len=args.max_memory_len,
        sample_stride=args.sample_stride,
        subset=split,
    )


def run_epoch(args, official_args, model, criterion, loader, device, optimizer=None, output_path: Path | None = None, label_names=None):
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)
    model.memory_queue = None
    model.memory_queue_index = None
    totals = {}
    seen = 0
    if output_path is not None and output_path.exists():
        output_path.unlink()
    for signals, targets, infos in tqdm(loader, desc="train" if is_train else "eval"):
        signals = signals.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        infos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in infos.items()}
        outputs = model({"inputs": signals, "infos": infos}, device)
        losses = criterion(outputs, targets, infos, device)
        weight = criterion.weight_dict
        loss = sum(losses[k] * weight[k] for k in losses if k in weight)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch = signals.shape[0]
        seen += batch
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * batch
        if output_path is not None:
            safe_make_txt(official_args, infos, outputs, str(output_path), label_names)
    return {k: v / max(seen, 1) for k, v in totals.items()}


def safe_make_txt(args, infos, outputs, file_path, label_map):
    """Official MATR make_txt with a batch-size-1-safe offset selection."""
    from util.utils import non_max_suppression

    pred_path = file_path.format("pred")
    Path(pred_path).touch()
    pred_inst_clses = outputs["pred_cls"]
    pred_inst_reges = outputs["pred_reg"]
    pred_inst_stcls = outputs["pred_stcls"]
    act_func = nn.Softmax(dim=1).to(pred_inst_clses.device)

    stsel_all = torch.argmax(pred_inst_stcls, dim=2)
    stregs_all = pred_inst_reges[:, :, : args.max_memory_len + 2]
    streg_all = torch.gather(stregs_all, 2, stsel_all.unsqueeze(-1)).squeeze(-1)
    edreg_all = pred_inst_reges[:, :, -1]
    pred_inst_reges = torch.stack([streg_all, edreg_all], dim=-1)

    for b in range(len(pred_inst_clses)):
        vid = infos["video_name"][b]
        fid = infos["current_frame"][b]
        frame_to_time = infos["frame_to_time"][b]
        f_inst_clses = pred_inst_clses[b]
        f_inst_reges = pred_inst_reges[b]
        f_inst_stcls = pred_inst_stcls[b]
        f_cls_prob = act_func(f_inst_clses)
        f_preds = []
        for idx in range(args.num_queries):
            cls = torch.argmax(f_cls_prob[idx][:-1], dim=0).reshape(-1)
            if f_cls_prob[idx][cls] < args.cls_threshold:
                continue
            cls_idx = int(cls.item())
            if cls_idx >= len(label_map):
                continue
            st_reg, ed_reg = f_inst_reges[idx]
            stsel = torch.argmax(f_inst_stcls[idx], dim=0).reshape(-1)
            st_offset = (stsel + st_reg) * args.num_frame
            ed_offset = ed_reg * args.num_frame
            st = fid - st_offset
            ed = fid - ed_offset
            st = st * frame_to_time
            ed = ed * frame_to_time
            cur = fid * frame_to_time
            cls_label = label_map[cls_idx]
            cls_prob = f_cls_prob[idx][cls]
            f_preds.append([
                str(vid),
                round(float(cur), 2),
                round(float(st), 2),
                round(float(ed), 2),
                str(cls_label),
                round(float(cls_prob), 4),
            ])
        if not f_preds:
            continue
        f_preds = non_max_suppression(f_preds, args.nms_threshold)
        with open(pred_path, "a+", encoding="utf-8") as f:
            for f_pred in f_preds:
                f.write("   ".join(str(k) for k in f_pred) + "\r\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matr-root", default="../MATR_codebase")
    parser.add_argument("--config", default="configs/wifitad_full.yaml")
    parser.add_argument("--mode", default="train_test_eval", choices=["train", "eval", "train_test_eval"])
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--num-frame", type=int, default=64)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--feat-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--e-nheads", type=int, default=4)
    parser.add_argument("--d-nheads", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=2)
    parser.add_argument("--dec-layers", type=int, default=2)
    parser.add_argument("--max-memory-len", type=int, default=7)
    parser.add_argument("--memory-sampler", default="gap2")
    parser.add_argument("--detect-len", type=int, default=16)
    parser.add_argument("--anti-len", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--flag-threshold", type=float, default=0.5)
    parser.add_argument("--use-flag", action="store_true")
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--print-label-stats", action="store_true")
    parsed = parser.parse_args()

    cfg_path = (ROOT / parsed.config).resolve() if not Path(parsed.config).is_absolute() else Path(parsed.config)
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 2020)))
    add_matr_root((ROOT / parsed.matr_root).resolve() if not Path(parsed.matr_root).is_absolute() else parsed.matr_root)

    from criterion import build_criterion
    from eval import evaluation_detection
    from models import build_model
    from util.utils import online_nms

    origin_to_dense, dense_to_name = load_class_index(resolve_path(cfg["dataset"]["class_index_path"], cfg["_config_dir"]))
    label_names = [dense_to_name[i] for i in sorted(dense_to_name)]
    run_dir = Path(parsed.run_dir) if parsed.run_dir else ROOT / "runs" / f"faithful-matr-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get("device", "auto"))

    train_dataset = build_dataset(cfg, "train", parsed, label_names)
    test_dataset = build_dataset(cfg, "test", parsed, label_names)
    if parsed.print_label_stats:
        print_label_stats(train_dataset, "train")
        print_label_stats(test_dataset, "test")
        if parsed.mode == "eval":
            return
    gt_json = run_dir / "wifitad_matr_gt.json"
    write_gt_json(test_dataset, gt_json)
    official_args = build_official_args(cfg, run_dir, gt_json, "train", parsed)
    model = build_model(official_args)
    model = patch_official_matr_sensor_encoder(model, int(cfg["model"]["in_channels"])).to(device)
    ensure_matr_position_capacity(model, parsed.num_frame)
    criterion = build_criterion(official_args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))

    if parsed.checkpoint:
        ckpt = torch.load(parsed.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])

    train_loader = DataLoader(train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=matr_collate)
    test_loader = DataLoader(test_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=matr_collate)

    if parsed.mode in {"train", "train_test_eval"}:
        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            log = run_epoch(parsed, official_args, model, criterion, train_loader, device, optimizer=optimizer)
            print(f"epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
            torch.save({"model": model.state_dict(), "criterion": criterion.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, run_dir / "checkpoint_last.pt")

    if parsed.mode in {"eval", "train_test_eval"}:
        official_args.training = False
        official_args.mode = "eval"
        official_args.save_path = str(run_dir)
        proposal_file = official_args.proposal_path.format("{}", "test", "final")
        proposal_txt_path = run_dir / f"{proposal_file}.txt"
        run_epoch(parsed, official_args, model, criterion, test_loader, device, optimizer=None, output_path=proposal_txt_path, label_names=label_names)
        pred_txt = str(proposal_txt_path).format("pred")
        result_dict = online_nms(official_args, pred_txt, test_dataset)
        pred_json = run_dir / "faithful_matr_predictions.json"
        with pred_json.open("w", encoding="utf-8") as f:
            json.dump({"version": "VERSION 1", "results": result_dict, "external_data": {}}, f, indent=2)
        mAP = evaluation_detection(official_args, str(pred_json), subset="test", tiou_thresholds=np.linspace(0.3, 0.70, 5), verbose=True)
        metrics = {
            "mAP@0.30": float(mAP[0]),
            "mAP@0.40": float(mAP[1]),
            "mAP@0.50": float(mAP[2]),
            "mAP@0.60": float(mAP[3]),
            "mAP@0.70": float(mAP[4]),
            "mAP@Avg": float(np.mean(mAP)),
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
