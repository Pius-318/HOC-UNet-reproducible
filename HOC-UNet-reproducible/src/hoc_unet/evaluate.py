"""Evaluate HOC-UNet segmentation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import SignalSegmentationDataset
from .metrics import confusion_matrix, metrics_from_confusion
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HOC-UNet.")
    parser.add_argument("--config", default="configs/hoc_unet_multiclass.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default="runs/hoc_unet/eval_metrics.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SignalSegmentationDataset(
        args.data_root,
        split=args.split,
        image_size=tuple(cfg.get("data", {}).get("image_size", [512, 512])),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(cfg["model"]).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    hist = np.zeros((cfg["model"]["num_classes"], cfg["model"]["num_classes"]), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].numpy()
            pred = model(images).argmax(dim=1).cpu().numpy()
            for p, t in zip(pred, masks):
                hist += confusion_matrix(p, t, cfg["model"]["num_classes"])
    metrics = metrics_from_confusion(hist)
    metrics["confusion_matrix"] = hist.tolist()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ("mIoU", "mAcc", "aAcc")}, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
