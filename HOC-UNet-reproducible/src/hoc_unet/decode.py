"""Mask-to-parameter decoding for time-frequency components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class DecodedComponent:
    label: int
    class_name: str
    t0: float
    duration: float
    center_frequency: float
    bandwidth: float
    area: int


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current
            while q:
                cy, cx = q.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        q.append((ny, nx))
    return labels, current


def decode_mask(
    mask: np.ndarray,
    class_names: Sequence[str],
    time_span: float = 1.8,
    freq_min: float = -1.2e6,
    freq_max: float = 1.2e6,
    min_area: int = 8,
) -> list[DecodedComponent]:
    """Decode rectangular time-frequency components from a semantic mask."""

    mask = np.asarray(mask)
    components, n_components = connected_components(mask > 0)
    h, w = mask.shape
    results: list[DecodedComponent] = []
    freq_span = freq_max - freq_min
    for cid in range(1, n_components + 1):
        ys, xs = np.where(components == cid)
        if len(xs) < min_area:
            continue
        values, counts = np.unique(mask[ys, xs], return_counts=True)
        values = values[values > 0]
        if len(values) == 0:
            continue
        label = int(values[np.argmax([np.sum(mask[ys, xs] == v) for v in values])])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        t0 = x0 / max(w - 1, 1) * time_span
        duration = (x1 - x0 + 1) / max(w, 1) * time_span
        f_high = freq_max - y0 / max(h - 1, 1) * freq_span
        f_low = freq_max - y1 / max(h - 1, 1) * freq_span
        center = 0.5 * (f_low + f_high)
        bandwidth = abs(f_high - f_low)
        class_name = class_names[label] if label < len(class_names) else str(label)
        results.append(DecodedComponent(label, class_name, t0, duration, center, bandwidth, int(len(xs))))
    return results
