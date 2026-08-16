"""Create a tiny synthetic time-frequency dataset for smoke tests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PALETTE = [
    (0, 0, 0),
    (255, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
    (255, 165, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 128, 128),
]

CLASS_NAMES = ["background", "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "FM", "AM-DSB", "AM-SSB", "MSK"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy")
    parser.add_argument("--num-train", type=int, default=16)
    parser.add_argument("--num-val", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_sample(path_img: Path, path_mask: Path, path_csv: Path, size: int) -> None:
    rng = random.Random(path_img.stem)
    image_arr = np.clip(rng.gauss(80, 8) + np.random.randn(size, size, 3) * 10, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_arr, "RGB")
    mask = Image.new("RGB", (size, size), PALETTE[0])
    draw_img = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    n_signals = rng.randint(1, 3)
    rows = []
    for _ in range(n_signals):
        label = rng.randint(1, 9)
        width = rng.randint(size // 8, size // 3)
        height = rng.randint(size // 12, size // 5)
        x0 = rng.randint(0, size - width - 1)
        y0 = rng.randint(0, size - height - 1)
        x1 = x0 + width
        y1 = y0 + height
        color = PALETTE[label]
        draw_mask.rectangle([x0, y0, x1, y1], fill=color)
        draw_img.rectangle([x0, y0, x1, y1], fill=tuple(min(255, c + rng.randint(-20, 20)) for c in color))
        t0 = x0 / size * 1.8
        duration = width / size * 1.8
        f_high = 1.2e6 - y0 / size * 2.4e6
        f_low = 1.2e6 - y1 / size * 2.4e6
        rows.append(
            {
                "Type": CLASS_NAMES[label],
                "Start_s": t0,
                "Duration_s": duration,
                "Freq_Hz": 0.5 * (f_high + f_low),
                "Bandwidth_Hz": abs(f_high - f_low),
            }
        )
    image.save(path_img)
    mask.save(path_mask)
    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Type", "Start_s", "Duration_s", "Freq_Hz", "Bandwidth_Hz"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.out)
    for split, count in (("train", args.num_train), ("val", args.num_val)):
        for sub in ("images", "masks", "params"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            stem = f"Signal_{idx + 1:04d}"
            make_sample(
                root / "images" / split / f"{stem}.jpg",
                root / "masks" / split / f"{stem}.png",
                root / "params" / split / f"{stem}.csv",
                args.size,
            )
    print(f"created toy dataset at: {root}")


if __name__ == "__main__":
    main()
