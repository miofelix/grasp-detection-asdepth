"""DPT fusion decoder，参数命名与源 `rgbddepth.dpt.DPTHead` 一致。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...vendor.dpt_blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(
    features: int,
    use_batch_norm: bool,
    size: tuple[int, int] | None = None,
) -> FeatureFusionBlock:
    return FeatureFusionBlock(  # type: ignore[no-untyped-call]
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_batch_norm,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv_block(inputs))


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int = 256,
        use_bn: bool = False,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        use_clstoken: bool = False,
        sigact_out: bool = False,
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError("DPTHead requires four output channel levels")
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                    for _ in self.projects
                ]
            )

        scratch: Any = _make_scratch(  # type: ignore[no-untyped-call]
            list(out_channels), features, groups=1, expand=False
        )
        scratch.stem_transpose = None
        scratch.refinenet1 = _make_fusion_block(features, use_bn)
        scratch.refinenet2 = _make_fusion_block(features, use_bn)
        scratch.refinenet3 = _make_fusion_block(features, use_bn)
        scratch.refinenet4 = _make_fusion_block(features, use_bn)
        scratch.output_conv1 = nn.Conv2d(
            features,
            features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if sigact_out:
            scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )
        else:
            scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
        self.scratch = scratch

    def forward(
        self,
        out_features: Sequence[tuple[torch.Tensor, torch.Tensor]],
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for index, feature in enumerate(out_features):
            tokens, class_token = feature
            if self.use_clstoken:
                readout = class_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, readout), -1))
            tokens = tokens.permute(0, 2, 1).reshape(
                (tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            )
            outputs.append(self.resize_layers[index](self.projects[index](tokens)))

        layer_1, layer_2, layer_3, layer_4 = outputs
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        output = self.scratch.output_conv1(path_1)
        output = functional.interpolate(
            output,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        return cast(torch.Tensor, self.scratch.output_conv2(output))


__all__ = ["ConvBlock", "DPTHead"]
