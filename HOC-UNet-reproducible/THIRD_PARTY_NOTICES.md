# Third-party notices

This repository contains a compact reimplementation of modules used in the
HOC-UNet manuscript workflow.

- Coordinate Attention is adapted from the design described in "Coordinate
  Attention for Efficient Mobile Network Design" and the public implementation
  at https://github.com/Andrew-Qibin/CoordAttention.
- Omni-dimensional attention is adapted from the attention mechanism in
  "Omni-Dimensional Dynamic Convolution" and the public implementation at
  https://github.com/OSVAI/ODConv.
- The original research engineering folder was based on an Apache-2.0 licensed
  RepViT/mmseg-style project. This cleaned repository removes the training
  artifacts and rewrites the HOC-UNet path as standalone PyTorch code for
  reproducibility.
