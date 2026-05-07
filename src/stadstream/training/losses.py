from __future__ import annotations

import torch
from torch.nn import functional as F


def anchor_targets(priors: torch.Tensor, targets: list[torch.Tensor], num_classes: int):
    centers = priors[0, :, 0]
    batch_cls = []
    batch_dist = []
    for target in targets:
        cls = torch.zeros_like(centers, dtype=torch.long)
        dist = torch.full((centers.numel(), 2), 0.05, device=centers.device)
        target = target.to(centers.device)
        for start, end, label in target:
            label_i = int(label.item())
            if label_i <= 0:
                continue
            inside = (centers >= start) & (centers <= end)
            cls[inside] = min(label_i, num_classes - 1)
            dist[inside, 0] = (centers[inside] - start).clamp_min(0.0)
            dist[inside, 1] = (end - centers[inside]).clamp_min(0.0)
        batch_cls.append(cls)
        batch_dist.append(dist)
    return torch.stack(batch_cls), torch.stack(batch_dist)


def detection_loss(outputs: dict[str, torch.Tensor], targets: list[torch.Tensor], num_classes: int):
    cls_t, dist_t = anchor_targets(outputs["priors"], targets, num_classes)
    conf = outputs["conf"]
    loc = outputs["loc"]
    cls_loss = F.cross_entropy(conf.reshape(-1, num_classes), cls_t.reshape(-1))
    pos = cls_t > 0
    if pos.any():
        loc_loss = F.smooth_l1_loss(loc[pos], dist_t[pos])
    else:
        loc_loss = loc.sum() * 0.0
    return {"total": cls_loss + loc_loss, "cls": cls_loss, "loc": loc_loss}

