"""原始双 DINOv2 RGB/depth 模型；只包含参数构建与 forward。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..components.backbones.dinov2 import create_dinov2_backbone
from ..components.decoders import DPTHead
from ..components.fusion import fuse_cross_attention


class RGBDDepth(nn.Module):
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
        load_depth_encoder: bool = True,
        freeze_rgb: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_rgb = freeze_rgb
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.max_depth = max_depth
        self.encoder = encoder
        self.pretrained: Any = create_dinov2_backbone(
            encoder,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
        if self.freeze_rgb:
            for parameter in self.pretrained.parameters():
                parameter.requires_grad = False
        if load_depth_encoder:
            self.depth_pretrained: Any = create_dinov2_backbone(
                encoder,
                pretrained=pretrained,
                pretrained_path=pretrained_path,
            )
        else:
            self.depth_pretrained = None
        embed_dim = cast(int, self.pretrained.embed_dim)
        self.depth_head_rgbd: Any = DPTHead(
            embed_dim * 2,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            sigact_out=False,
        )
        self.crossAtts = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim, 4, batch_first=True) for _ in range(4)]
        )

    def forward(self, inputs: torch.Tensor) -> Any:
        rgb, depth = inputs[:, :3], inputs[:, 3:]
        patch_h, patch_w = inputs.shape[-2] // 14, inputs.shape[-1] // 14
        context_manager = torch.no_grad() if self.freeze_rgb else nullcontext()
        with context_manager:
            rgb_features = self.pretrained.get_intermediate_layers(
                rgb,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True,
            )
        if self.depth_pretrained is None:
            raise RuntimeError("RGBDDepth depth encoder is not configured")
        depth_features = self.depth_pretrained.get_intermediate_layers(
            depth.repeat(1, 3, 1, 1),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        fused = fuse_cross_attention(rgb_features, depth_features, self.crossAtts)
        output = self.depth_head_rgbd(fused, patch_h, patch_w)
        return functional.relu(output).squeeze(1)
