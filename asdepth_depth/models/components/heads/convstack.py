"""保持源参数层级的 ConvStack neck/head presets。"""

from __future__ import annotations

from typing import Any, cast

from ..decoders import ConvStack


def create_convstack_neck() -> ConvStack:
    constructor: Any = ConvStack
    return cast(
        ConvStack,
        constructor(
            dim_in=[2048, None, None, None, None],
            dim_out=None,
            dim_res_blocks=[1024, 256, 128, 64, 32],
            num_res_blocks=[0, 2, 2, 2, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        ),
    )


def create_convstack_head(output_channels: int) -> ConvStack:
    if output_channels <= 0:
        raise ValueError("output_channels must be positive")
    constructor: Any = ConvStack
    return cast(
        ConvStack,
        constructor(
            dim_in=[1024, 256, 128, 64, 32],
            dim_out=[None, None, None, None, output_channels],
            dim_res_blocks=[1024, 256, 128, 64, 32],
            num_res_blocks=[0, 1, 1, 1, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        ),
    )
