"""Train HOC-UNet."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config
from .data import SignalSegmentationDataset
from .model import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOC-UNet for time-frequency segmentation.")
    parser.add_argument("--config", default="configs/hoc_unet_multiclass.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="runs/hoc_unet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg["model"]

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset = SignalSegmentationDataset(
        args.data_root,
        split="train",
        image_size=tuple(data_cfg.get("image_size", [512, 512])),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr or train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=train_cfg.get("ignore_index", 255))
    epochs = args.epochs or train_cfg.get("epochs", 80)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
        avg_loss = total_loss / len(dataset)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"epoch={epoch:03d} loss={avg_loss:.6f}")

    checkpoint = {
        "model": model.state_dict(),
        "config": cfg,
        "num_parameters": count_parameters(model),
    }
    torch.save(checkpoint, output / "hoc_unet_last.pt")
    with (output / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"saved: {output / 'hoc_unet_last.pt'}")


if __name__ == "__main__":
    main()
