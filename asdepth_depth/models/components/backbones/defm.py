"""Vendored DeFM encoder 与 metric-depth preprocessing 的稳定边界。"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn

from ...vendor.defm.model_factory import create_defm_model
from ...vendor.defm.utils.utils import preprocess_depth_batch


def create_defm_backbone(
    model_name: str = "defm_vit_l14",
    *,
    pretrained: bool = False,
    pretrained_path: str | None = None,
) -> nn.Module:
    builder: Any = create_defm_model
    return cast(
        nn.Module,
        builder(
            model_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        ),
    )


def preprocess_defm_depth(depth: torch.Tensor, *, patch_size: int = 14) -> torch.Tensor:
    preprocess: Any = preprocess_depth_batch
    return cast(
        torch.Tensor,
        preprocess(
            depth,
            target_size=(depth.shape[-2], depth.shape[-1]),
            patch_size=patch_size,
            device=depth.device,
        ),
    )


__all__ = ["create_defm_backbone", "preprocess_defm_depth"]
