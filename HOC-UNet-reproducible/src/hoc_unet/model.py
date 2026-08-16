"""Standalone PyTorch implementation of HOC-UNet.

The implementation keeps the computational pipeline used in the manuscript:
high-frequency perception blocks in the encoder and early skip connections,
omni-dimensional attention in the decoder fusion blocks, and coordinate
attention on selected skip features. It intentionally avoids the mmseg/mmcv
registry layer so that reviewers can run the model with a compact dependency
set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_dct as DCT
except Exception:  # pragma: no cover - optional dependency
    DCT = None


@dataclass
class HOCUNetConfig:
    in_channels: int = 3
    num_classes: int = 10
    base_channels: int = 64
    bilinear: bool = True
    use_hfp: bool = True
    use_hfp_dct: bool = False
    use_hfp_skip: bool = True
    use_omni_layers: Sequence[int] = field(default_factory=lambda: (0, 1, 2, 3))
    use_ca_skip: bool = True
    ca_reduction: int = 32
    omni_kernel_size: int = 3
    omni_reduction: float = 0.0625
    omni_kernel_num: int = 4
    omni_residual_weight: float = 0.5


def _valid_group_count(channels: int, preferred: int = 32) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate attention for preserving height/width positional cues."""

    def __init__(self, inp: int, oup: Optional[int] = None, reduction: int = 32):
        super().__init__()
        oup = oup or inp
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class ODConvAttention(nn.Module):
    """Compact omni-dimensional attention module adapted for feature fusion."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        min_channel: int = 16,
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.norm = nn.GroupNorm(1, attention_channel)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = None if in_planes == groups and in_planes == out_planes else nn.Conv2d(
            attention_channel, out_planes, 1, bias=True
        )
        self.spatial_fc = None if kernel_size == 1 else nn.Conv2d(
            attention_channel, kernel_size * kernel_size, 1, bias=True
        )
        self.kernel_fc = None if kernel_num == 1 else nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.relu(self.norm(self.fc(self.avgpool(x))))
        channel_attention = torch.sigmoid(self.channel_fc(pooled) / self.temperature)
        if self.filter_fc is None:
            filter_attention = torch.ones_like(channel_attention)
        else:
            filter_attention = torch.sigmoid(self.filter_fc(pooled) / self.temperature)
        return channel_attention, filter_attention


class OmniAttention(nn.Module):
    """Feature-level omni-dimensional attention with residual blending."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        residual_weight: float = 0.5,
    ):
        super().__init__()
        self.residual_weight = residual_weight
        self.attention = ODConvAttention(
            in_planes=channels,
            out_planes=channels,
            kernel_size=kernel_size,
            reduction=reduction,
            kernel_num=kernel_num,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention, filter_attention = self.attention(x)
        enhanced = x * channel_attention * filter_attention
        return x * (1.0 - self.residual_weight) + enhanced * self.residual_weight


class DctSpatialInteraction(nn.Module):
    def __init__(self, in_channels: int, ratio: tuple[float, float], isdct: bool = False):
        super().__init__()
        self.ratio = ratio
        self.isdct = isdct
        self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        if not self.isdct or DCT is None:
            return x * torch.sigmoid(self.spatial1x1(x))
        spectrum = DCT.dct_2d(x, norm="ortho")
        weight = self._compute_weight(h, w, self.ratio).to(x.device, x.dtype)
        filtered = DCT.idct_2d(spectrum * weight.view(1, 1, h, w), norm="ortho")
        return x * filtered

    @staticmethod
    def _compute_weight(h: int, w: int, ratio: tuple[float, float]) -> torch.Tensor:
        weight = torch.ones((h, w), requires_grad=False)
        weight[: int(h * ratio[0]), : int(w * ratio[1])] = 0
        return weight


class DctChannelInteraction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch: tuple[int, int],
        ratio: tuple[float, float],
        isdct: bool = False,
    ):
        super().__init__()
        groups = _valid_group_count(in_channels)
        self.patch = patch
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.isdct or DCT is None:
            pooled = self.relu(F.adaptive_max_pool2d(x, 1)) + self.relu(F.adaptive_avg_pool2d(x, 1))
            return x * torch.sigmoid(self.channel2x1(self.channel1x1(pooled)))
        _, _, h, w = x.size()
        spectrum = DCT.dct_2d(x, norm="ortho")
        weight = DctSpatialInteraction._compute_weight(h, w, self.ratio).to(x.device, x.dtype)
        filtered = DCT.idct_2d(spectrum * weight.view(1, 1, h, w), norm="ortho")
        pooled = self.relu(F.adaptive_max_pool2d(filtered, 1)) + self.relu(F.adaptive_avg_pool2d(filtered, 1))
        return x * torch.sigmoid(self.channel2x1(self.channel1x1(pooled)))


class HFP(nn.Module):
    """High-frequency perception block."""

    def __init__(
        self,
        in_channels: int,
        ratio: tuple[float, float] = (0.25, 0.25),
        patch: tuple[int, int] = (8, 8),
        isdct: bool = False,
    ):
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.GroupNorm(_valid_group_count(in_channels), in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.spatial(x) + self.channel(x))


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        use_omni: bool = False,
        use_ca: bool = False,
        ca_reduction: int = 32,
        omni_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.omni = OmniAttention(in_channels, **(omni_kwargs or {})) if use_omni else None
        self.ca = CoordAtt(in_channels, in_channels, reduction=ca_reduction) if use_ca else None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        if self.omni is not None:
            x = self.omni(x)
        if self.ca is not None:
            x = self.ca(x)
        return self.conv(x)


class HOCUNet(nn.Module):
    """HOC-UNet segmentation network."""

    def __init__(self, config: Optional[HOCUNetConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = HOCUNetConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either a HOCUNetConfig or keyword arguments, not both.")
        self.config = config
        b = config.base_channels
        factor = 2 if config.bilinear else 1
        omni_kwargs = {
            "kernel_size": config.omni_kernel_size,
            "reduction": config.omni_reduction,
            "kernel_num": config.omni_kernel_num,
            "residual_weight": config.omni_residual_weight,
        }

        self.inc = DoubleConv(config.in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16 // factor)

        self.hfp1 = HFP(b, patch=(16, 16), isdct=config.use_hfp_dct) if config.use_hfp else None
        self.hfp2 = HFP(b * 2, patch=(8, 8), isdct=config.use_hfp_dct) if config.use_hfp else None
        self.hfp3 = HFP(b * 4, patch=(4, 4), isdct=config.use_hfp_dct) if config.use_hfp else None
        self.hfp_skip1 = HFP(b, patch=(16, 16), isdct=config.use_hfp_dct) if config.use_hfp_skip else None
        self.hfp_skip2 = HFP(b * 2, patch=(8, 8), isdct=config.use_hfp_dct) if config.use_hfp_skip else None
        self.ca_skip1 = CoordAtt(b, b, reduction=config.ca_reduction) if config.use_ca_skip else None
        self.ca_skip2 = CoordAtt(b * 2, b * 2, reduction=config.ca_reduction) if config.use_ca_skip else None

        omni_layers = set(config.use_omni_layers)
        self.up1 = Up(b * 16 // factor + b * 8, b * 8 // factor, config.bilinear, 0 in omni_layers, False, config.ca_reduction, omni_kwargs)
        self.up2 = Up(b * 8 // factor + b * 4, b * 4 // factor, config.bilinear, 1 in omni_layers, False, config.ca_reduction, omni_kwargs)
        self.up3 = Up(b * 4 // factor + b * 2, b * 2 // factor, config.bilinear, 2 in omni_layers, False, config.ca_reduction, omni_kwargs)
        self.up4 = Up(b * 2 // factor + b, b, config.bilinear, 3 in omni_layers, False, config.ca_reduction, omni_kwargs)
        self.outc = nn.Conv2d(b, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        if self.hfp1 is not None:
            x1 = self.hfp1(x1)
        x2 = self.down1(x1)
        if self.hfp2 is not None:
            x2 = self.hfp2(x2)
        x3 = self.down2(x2)
        if self.hfp3 is not None:
            x3 = self.hfp3(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.hfp_skip2 is not None:
            x2 = self.hfp_skip2(x2)
        if self.ca_skip2 is not None:
            x2 = self.ca_skip2(x2)
        if self.hfp_skip1 is not None:
            x1 = self.hfp_skip1(x1)
        if self.ca_skip1 is not None:
            x1 = self.ca_skip1(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(config: Union[dict, HOCUNetConfig]) -> HOCUNet:
    if isinstance(config, HOCUNetConfig):
        return HOCUNet(config)
    return HOCUNet(HOCUNetConfig(**config))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_modules(model: nn.Module, module_names: Iterable[str]) -> None:
    for name, module in model.named_modules():
        if name in module_names:
            for param in module.parameters():
                param.requires_grad_(False)
