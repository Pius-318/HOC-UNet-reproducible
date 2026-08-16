"""Segmentation metrics."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: Optional[int] = 255) -> np.ndarray:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    hist = np.bincount(
        num_classes * target[valid].astype(int) + pred[valid].astype(int),
        minlength=num_classes**2,
    )
    return hist.reshape(num_classes, num_classes)


def metrics_from_confusion(hist: np.ndarray) -> dict[str, Union[float, list[float]]]:
    hist = hist.astype(np.float64)
    true_positive = np.diag(hist)
    pred_count = hist.sum(axis=0)
    true_count = hist.sum(axis=1)
    union = pred_count + true_count - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    acc = np.divide(true_positive, true_count, out=np.zeros_like(true_positive), where=true_count > 0)
    total = hist.sum()
    return {
        "mIoU": float(np.nanmean(iou)),
        "mAcc": float(np.nanmean(acc)),
        "aAcc": float(true_positive.sum() / total) if total > 0 else 0.0,
        "IoU": iou.tolist(),
        "Acc": acc.tolist(),
    }
