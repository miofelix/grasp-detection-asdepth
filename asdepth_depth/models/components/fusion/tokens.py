"""保持源数学计算的 token-level RGB/depth fusion。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

TokenFeature = tuple[torch.Tensor, torch.Tensor]


def fuse_cross_attention(
    rgb_features: Sequence[TokenFeature],
    depth_features: Sequence[TokenFeature],
    cross_attentions: Iterable[nn.Module],
    depth_adapters: Iterable[nn.Module] | None = None,
) -> list[TokenFeature]:
    """逐空间 token 执行源 2-token self-attention fusion。"""

    attentions = tuple(cross_attentions)
    adapters = (None,) * len(attentions) if depth_adapters is None else tuple(depth_adapters)
    if not (len(rgb_features) == len(depth_features) == len(attentions) == len(adapters)):
        raise ValueError("RGB/depth/fusion layer counts must match")

    fused_features: list[TokenFeature] = []
    for rgb_feature, depth_feature, cross_attention, adapter in zip(
        rgb_features,
        depth_features,
        attentions,
        adapters,
        strict=True,
    ):
        rgb_tokens, rgb_cls = rgb_feature
        depth_tokens, depth_cls = depth_feature
        adapted_tokens = depth_tokens if adapter is None else adapter(depth_tokens)
        batch, token_count, channels = rgb_tokens.shape
        token_features = torch.concat(
            (
                rgb_tokens.reshape(batch * token_count, 1, channels),
                adapted_tokens.reshape(batch * token_count, 1, channels),
            ),
            dim=1,
        )
        attention_features, _ = cross_attention(
            token_features,
            token_features,
            token_features,
        )
        attention_features = (
            attention_features.reshape(batch * token_count, 2, channels)
            .sum(axis=1)
            .reshape(batch, token_count, channels)
        )
        fused_features.append(
            (
                torch.concat((rgb_tokens, attention_features), dim=2),
                torch.concat((rgb_cls, depth_cls), dim=1),
            )
        )
    return fused_features


def fuse_gated_tokens(
    rgb_features: Sequence[TokenFeature],
    depth_features: Sequence[TokenFeature],
    gates: Iterable[nn.Module],
) -> list[torch.Tensor]:
    """DINOv2 new-fusion variant 的逐层 channel gate。"""

    gate_layers = tuple(gates)
    if not (len(rgb_features) == len(depth_features) == len(gate_layers)):
        raise ValueError("RGB/depth/gate layer counts must match")
    outputs: list[torch.Tensor] = []
    for rgb_feature, depth_feature, gate_projection in zip(
        rgb_features,
        depth_features,
        gate_layers,
        strict=True,
    ):
        rgb_tokens = rgb_feature[0]
        depth_tokens = depth_feature[0]
        gate = torch.sigmoid(gate_projection(torch.cat((rgb_tokens, depth_tokens), dim=-1)))
        fused = gate * depth_tokens + (1.0 - gate) * rgb_tokens
        outputs.append(torch.cat((rgb_tokens, fused), dim=-1))
    return outputs


def aggregate_feature_tokens(
    features: Iterable[torch.Tensor],
    *,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    """按源顺序求和并恢复 BCHW feature map。"""

    values = tuple(features)
    if not values:
        raise ValueError("at least one feature tensor is required")
    total = values[0]
    for value in values[1:]:
        total = total + value
    feature_map = total.permute(0, 2, 1).reshape(
        total.shape[0],
        total.shape[2],
        patch_height,
        patch_width,
    )
    return feature_map
