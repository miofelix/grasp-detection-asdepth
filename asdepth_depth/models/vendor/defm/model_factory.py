# Copyright (c) 2026, ETH Zurich, Manthan Patel
#
# This source code is licensed under the Apache License, Version 2.0.

"""只构建 AS-Depth-2 使用的 DeFM ViT-L/14 backbone。"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from .models import vision_transformer as vits

BASE_PATH = Path(__file__).parent.resolve()


def get_defm_config(model_name: str):
    if model_name != "defm_vit_l14":
        raise ValueError(f"unsupported migrated DeFM model: {model_name}")
    return OmegaConf.load(BASE_PATH / "configs" / f"{model_name}.yaml")


def create_defm_model(
    model_name: str,
    *,
    pretrained: bool = False,
    pretrained_path: str | None = None,
):
    """构建参数层级与 AS-Depth ``defm_vit_l14`` 一致的模型。"""

    cfg = get_defm_config(model_name)
    model = vits.__dict__[cfg.arch](
        img_size=cfg.global_crops_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        init_values=cfg.layerscale,
        ffn_layer=cfg.ffn_layer,
        block_chunks=cfg.block_chunks,
        qkv_bias=cfg.qkv_bias,
        proj_bias=cfg.proj_bias,
        ffn_bias=cfg.ffn_bias,
        num_register_tokens=cfg.num_register_tokens,
        interpolate_offset=cfg.interpolate_offset,
        interpolate_antialias=cfg.interpolate_antialias,
    )
    if pretrained:
        if not pretrained_path:
            raise ValueError("migrated DeFM backbone requires an explicit pretrained_path")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and isinstance(state_dict.get("model"), dict):
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    return model
