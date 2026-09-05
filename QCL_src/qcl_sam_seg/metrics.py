from __future__ import annotations

import torch
import torch.nn.functional as F


class SegmentationMetrics:
    def __init__(self, classes: int, ignore_index: int = 255):
        self.classes, self.ignore_index = classes, ignore_index
        self.cm = torch.zeros(classes, classes, dtype=torch.float64)
        self.boundary_intersection = torch.zeros(classes, dtype=torch.float64)
        self.boundary_union = torch.zeros(classes, dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(1).detach()
        target = target.detach().to(prediction.device)
        if self.cm.device != prediction.device:
            self.cm = self.cm.to(prediction.device)
            self.boundary_intersection = self.boundary_intersection.to(prediction.device)
            self.boundary_union = self.boundary_union.to(prediction.device)
        valid = target != self.ignore_index
        encoded = self.classes * target[valid].long() + prediction[valid].long()
        self.cm += torch.bincount(encoded, minlength=self.classes ** 2).reshape(self.classes, self.classes)
        for cls in range(self.classes):
            pred, truth = prediction.eq(cls), target.eq(cls)
            pred_b = pred ^ F.max_pool2d(pred.float().unsqueeze(1), 3, 1, 1).squeeze(1).bool()
            truth_b = truth ^ F.max_pool2d(truth.float().unsqueeze(1), 3, 1, 1).squeeze(1).bool()
            self.boundary_intersection[cls] += (pred_b & truth_b & valid).sum()
            self.boundary_union[cls] += ((pred_b | truth_b) & valid).sum()

    def compute(self) -> dict:
        tp = self.cm.diag()
        denom_iou = self.cm.sum(0) + self.cm.sum(1) - tp
        denom_dice = self.cm.sum(0) + self.cm.sum(1)
        iou = torch.where(denom_iou > 0, tp / denom_iou, torch.nan)
        dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.nan)
        biou = torch.where(self.boundary_union > 0, self.boundary_intersection / self.boundary_union, torch.nan)
        return {"accuracy": float(tp.sum() / self.cm.sum().clamp_min(1)), "miou": float(torch.nanmean(iou)), "dice": float(torch.nanmean(dice)), "biou": float(torch.nanmean(biou)), "per_class_iou": iou.nan_to_num(-1).tolist(), "per_class_dice": dice.nan_to_num(-1).tolist(), "confusion_matrix": self.cm.tolist()}
