#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
python examples/create_toy_dataset.py --out data/toy --num-train 4 --num-val 2 --size 64
python -m hoc_unet.train --config configs/hoc_unet_smoke.yaml --data-root data/toy --epochs 1 --batch-size 1 --device cpu --output runs/smoke
python -m hoc_unet.evaluate --config configs/hoc_unet_smoke.yaml --checkpoint runs/smoke/hoc_unet_last.pt --data-root data/toy --split val --device cpu --output runs/smoke/eval_metrics.json
python -m hoc_unet.infer --config configs/hoc_unet_smoke.yaml --checkpoint runs/smoke/hoc_unet_last.pt --image data/toy/images/val/Signal_0001.jpg --device cpu --output-dir runs/smoke/infer
