"""DeFM + ConvStack 的 depth、mask、normal 与 ablation variants。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from loguru import logger

from ..components.adapters import DepthAdapter
from ..components.backbones.defm import create_defm_backbone, preprocess_defm_depth
from ..components.fusion import aggregate_feature_tokens, fuse_cross_attention
from ..components.heads import create_convstack_head, create_convstack_neck
from .dino import RGBDDepth


def _initialize_defm(
    model: Any,
    *,
    depth_pretrained: bool,
    depth_pretrained_path: str | None,
    with_adapters: bool,
) -> None:
    model.depth_pretrained = create_defm_backbone(
        pretrained=depth_pretrained,
        pretrained_path=depth_pretrained_path,
    )
    if with_adapters:
        model.depth_adapters = nn.ModuleList(
            [DepthAdapter(model.pretrained.embed_dim, reduction=4) for _ in range(4)]
        )


def _initialize_stack(model: Any) -> None:
    del model.depth_head_rgbd
    model.neck = create_convstack_neck()
    model.depth_head_rgbd = create_convstack_head(1)


def _defm_features(model: Any, inputs: torch.Tensor, *, with_adapters: bool) -> list[Any]:
    rgb, depth = inputs[:, :3], inputs[:, 3:]
    with torch.no_grad():
        rgb_features = model.pretrained.get_intermediate_layers(
            rgb,
            model.intermediate_layer_idx[model.encoder],
            return_class_token=True,
        )
    depth_features = model.depth_pretrained.get_intermediate_layers(
        preprocess_defm_depth(depth),
        model.intermediate_layer_idx[model.encoder],
        return_class_token=True,
    )
    adapters = model.depth_adapters if with_adapters else None
    return cast(
        list[Any],
        fuse_cross_attention(
            rgb_features,
            depth_features,
            model.crossAtts,
            adapters,
        ),
    )


def _stack_features(model: Any, inputs: torch.Tensor, *, with_adapters: bool) -> list[Any]:
    fused = _defm_features(model, inputs, with_adapters=with_adapters)
    feature_map = aggregate_feature_tokens(
        (feature for feature, _ in fused),
        patch_height=inputs.shape[-2] // 14,
        patch_width=inputs.shape[-1] // 14,
    )
    return cast(list[Any], model.neck([feature_map, None, None, None, None]))


def _resize_output(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(
        output,
        (inputs.shape[-2], inputs.shape[-1]),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


class DeFMStackConvRGBDDepth(RGBDDepth):
    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        use_bn: bool = False,
        use_clstoken: bool = False,
        max_depth: float = 20.0,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        depth_pretrained: bool = False,
        depth_pretrained_path: str | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            max_depth=max_depth,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
        _initialize_defm(
            self,
            depth_pretrained=depth_pretrained,
            depth_pretrained_path=depth_pretrained_path,
            with_adapters=True,
        )
        _initialize_stack(self)
        logger.info("DeFMStackConvRGBDDepth initialized")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = _stack_features(self, inputs, with_adapters=True)
        depth = _resize_output(self.depth_head_rgbd(features)[-1], inputs)
        return functional.relu(depth).squeeze(1)
