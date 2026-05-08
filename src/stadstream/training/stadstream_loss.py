from __future__ import annotations

import torch
from torch.nn import functional as F


def stadstream_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], num_classes: int):
    action = targets["actionness"].to(outputs["actionness_logit"].device)
    cls = targets["cls"].to(outputs["cls_logits"].device)
    offsets = targets["offsets"].to(outputs["offsets"].device)
    start_b = targets["start_boundary"].to(outputs["start_logit"].device)
    end_b = targets["end_boundary"].to(outputs["end_logit"].device)

    pos_weight = ((action == 0).float().sum() / (action.sum() + 1.0)).clamp(max=20.0)
    loss_action = F.binary_cross_entropy_with_logits(outputs["actionness_logit"], action, pos_weight=pos_weight)
    loss_cls = F.cross_entropy(outputs["cls_logits"], cls)
    pos = action > 0.5
    if pos.any():
        loss_offsets = F.smooth_l1_loss(outputs["offsets"][pos], offsets[pos])
    else:
        loss_offsets = outputs["offsets"].sum() * 0.0
    loss_start = F.binary_cross_entropy_with_logits(outputs["start_logit"], start_b)
    loss_end = F.binary_cross_entropy_with_logits(outputs["end_logit"], end_b)
    total = loss_action + loss_cls + loss_offsets + 0.25 * (loss_start + loss_end)
    return {
        "total": total,
        "action": loss_action,
        "cls": loss_cls,
        "offset": loss_offsets,
        "start": loss_start,
        "end": loss_end,
    }

