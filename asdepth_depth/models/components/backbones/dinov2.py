"""Vendored DINOv2 的稳定构建边界。"""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn

from ...vendor.dinov2 import DINOv2, DinoVisionTransformer


def create_dinov2_backbone(
    model_name: str,
    *,
    pretrained: bool = False,
    pretrained_path: str | None = None,
) -> nn.Module:
    builder: Any = DINOv2
    return cast(
        nn.Module,
        builder(
            model_name=model_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        ),
    )


__all__ = ["DINOv2", "DinoVisionTransformer", "create_dinov2_backbone"]
