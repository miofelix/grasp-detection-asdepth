"""DeFM ViT-L/14 depth encoder + DPT decoder variant。"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from ..components.adapters import DepthAdapter
from ..components.backbones.defm import create_defm_backbone, preprocess_defm_depth
from ..components.fusion import fuse_cross_attention
from .dino import RGBDDepth


class DeFMRGBDDepth(RGBDDepth):
    """DeFM ViT-L/14 + DPT decoder RGB-D 深度模型。"""

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
        freeze_rgb: bool = False,
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
            freeze_rgb=freeze_rgb,
            load_depth_encoder=False,
        )
        self.depth_pretrained: Any = create_defm_backbone(
            pretrained=depth_pretrained,
            pretrained_path=depth_pretrained_path,
        )
        self.depth_adapters = nn.ModuleList(
            [DepthAdapter(self.pretrained.embed_dim, reduction=4) for _ in range(4)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rgb, depth = inputs[:, :3], inputs[:, 3:]
        patch_h, patch_w = inputs.shape[-2] // 14, inputs.shape[-1] // 14
        context_manager = torch.no_grad() if self.freeze_rgb else nullcontext()
        with context_manager:
            rgb_features: Any = self.pretrained.get_intermediate_layers(
                rgb,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True,
            )
        depth_features: Any = self.depth_pretrained.get_intermediate_layers(
            preprocess_defm_depth(depth),
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True,
        )
        fused = fuse_cross_attention(
            rgb_features,
            depth_features,
            self.crossAtts,
            self.depth_adapters,
        )
        output = self.depth_head_rgbd(fused, patch_h, patch_w)
        return functional.relu(output).squeeze(1)
