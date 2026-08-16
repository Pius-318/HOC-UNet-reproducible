# HOC-UNet reproducible code

This repository provides a compact, reviewer-friendly PyTorch implementation of
HOC-UNet for wireless wideband received-signal time-frequency segmentation and
mask-based parameter decoding.

The code is intentionally smaller than the original research engineering folder:
it keeps the HOC-UNet model, training loop, evaluation metrics, inference, and
geometric parameter decoding, while removing historical experiments, generated
figures, logs, pretrained weights, IDE files, and intermediate manuscript files.

## Repository structure

```text
configs/                  Model and training configuration files
examples/create_toy_dataset.py
                           Tiny synthetic dataset generator for smoke tests
src/hoc_unet/model.py      Standalone PyTorch HOC-UNet implementation
src/hoc_unet/train.py      Training entry point
src/hoc_unet/evaluate.py   mIoU, mAcc, aAcc, and confusion matrix evaluation
src/hoc_unet/infer.py      Mask prediction and decoded parameter export
src/hoc_unet/decode.py     Mask-to-parameter decoding
scripts/smoke_test.*       End-to-end CPU smoke test
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

For DCT-based HFP experiments, install the optional dependency:

```bash
python -m pip install -e ".[dct]"
```

The default configuration sets `use_hfp_dct: false`, matching the fast training
setting used for reproducible review runs.

## Dataset layout

Use this layout for a multiclass or binary dataset:

```text
data/my_dataset/
  images/train/*.jpg
  masks/train/*.png
  images/val/*.jpg
  masks/val/*.png
```

Masks may be single-channel class-index PNG files or RGB masks using the palette
defined in `src/hoc_unet/data.py`. The multiclass labels are:

```text
0 background, 1 BPSK, 2 QPSK, 3 8PSK, 4 16QAM, 5 64QAM,
6 FM, 7 AM-DSB, 8 AM-SSB, 9 MSK
```

## Quick reproducibility check

Run a complete CPU smoke test on a tiny generated dataset:

```bash
bash scripts/smoke_test.sh
```

On Windows PowerShell:

```powershell
.\scripts\smoke_test.ps1
```

The smoke test creates `data/toy`, trains for one epoch, evaluates segmentation
metrics, and exports an inferred mask plus decoded parameters. It is not intended
to reproduce paper-level accuracy; it verifies that the code path is executable.
It uses `configs/hoc_unet_smoke.yaml`, a small CPU-friendly model configuration.

## Train HOC-UNet

```bash
python -m hoc_unet.train \
  --config configs/hoc_unet_multiclass.yaml \
  --data-root data/my_dataset \
  --output runs/hoc_unet_multiclass
```

For binary segmentation, use:

```bash
python -m hoc_unet.train \
  --config configs/hoc_unet_binary.yaml \
  --data-root data/my_binary_dataset \
  --output runs/hoc_unet_binary
```

## Evaluate

```bash
python -m hoc_unet.evaluate \
  --config configs/hoc_unet_multiclass.yaml \
  --checkpoint runs/hoc_unet_multiclass/hoc_unet_last.pt \
  --data-root data/my_dataset \
  --split val \
  --output runs/hoc_unet_multiclass/eval_metrics.json
```

The evaluation JSON includes mIoU, mAcc, aAcc, per-class IoU/accuracy, and the
confusion matrix.

## Inference and parameter decoding

```bash
python -m hoc_unet.infer \
  --config configs/hoc_unet_multiclass.yaml \
  --checkpoint runs/hoc_unet_multiclass/hoc_unet_last.pt \
  --image data/my_dataset/images/val/Signal_0001.jpg \
  --output-dir runs/hoc_unet_multiclass/infer
```

This writes:

- `*_mask.png`: predicted semantic mask
- `*_params.csv`: decoded signal type, start time, duration, center frequency,
  and bandwidth
- `*_params.json`: same decoded parameters in JSON format

## Notes for manuscript review

No user login, password, or personal information is required to run this code.
Large generated datasets and trained weights are intentionally excluded from the
repository. Place the dataset locally using the layout above, or run the included
toy-data smoke test to verify the full software path.
