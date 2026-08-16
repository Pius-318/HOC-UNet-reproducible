"""Run inference and decode time-frequency parameters."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config
from .data import DEFAULT_MULTICLASS_NAMES, labels_to_color
from .decode import decode_mask
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer HOC-UNet masks and decoded parameters.")
    parser.add_argument("--config", default="configs/hoc_unet_multiclass.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="runs/hoc_unet/infer")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def preprocess(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    image = image.convert("RGB").resize(image_size, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [512, 512]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    image_path = Path(args.image)
    with Image.open(image_path) as image:
        x = preprocess(image, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        mask = model(x).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels_to_color(mask)).save(out_dir / f"{image_path.stem}_mask.png")
    components = decode_mask(
        mask,
        class_names=cfg.get("classes", DEFAULT_MULTICLASS_NAMES),
        time_span=data_cfg.get("time_span", 1.8),
        freq_min=data_cfg.get("freq_min", -1.2e6),
        freq_max=data_cfg.get("freq_max", 1.2e6),
    )

    rows = [c.__dict__ for c in components]
    with (out_dir / f"{image_path.stem}_params.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / f"{image_path.stem}_params.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "class_name", "t0", "duration", "center_frequency", "bandwidth", "area"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
