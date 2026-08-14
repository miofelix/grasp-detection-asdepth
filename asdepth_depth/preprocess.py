"""RealSense BGR/raw depth 到四通道深度模型输入的预处理。"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
PATCH_SIZE = 14


def _constrain(
    value: float,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    result = int(np.round(value / PATCH_SIZE) * PATCH_SIZE)
    if maximum is not None and result > maximum:
        result = int(np.floor(value / PATCH_SIZE) * PATCH_SIZE)
    if minimum is not None and result < minimum:
        result = int(np.ceil(value / PATCH_SIZE) * PATCH_SIZE)
    if result <= 0:
        raise ValueError(f"resize produced a non-positive dimension from {value}")
    return result


def output_size(
    width: int,
    height: int,
    *,
    input_size: int,
    resize_method: Literal["lower_bound", "upper_bound"],
) -> tuple[int, int]:
    if width <= 0 or height <= 0 or input_size <= 0:
        raise ValueError("image dimensions and input_size must be positive")
    if resize_method not in {"lower_bound", "upper_bound"}:
        raise ValueError(f"unsupported resize_method: {resize_method}")
    scales = (input_size / height, input_size / width)
    scale = max(scales) if resize_method == "lower_bound" else min(scales)
    constraint = (
        {"minimum": input_size} if resize_method == "lower_bound" else {"maximum": input_size}
    )
    return (
        _constrain(scale * width, **constraint),
        _constrain(scale * height, **constraint),
    )


def metric_depth_from_raw(
    raw_depth: np.ndarray,
    *,
    depth_scale: float,
    max_depth_m: float,
) -> np.ndarray:
    if depth_scale <= 0 or max_depth_m <= 0:
        raise ValueError("depth_scale and max_depth_m must be positive")
    raw = np.asarray(raw_depth)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"raw depth must be HxW, got {raw.shape}")
    values = raw.astype(np.float32) / np.float32(depth_scale)
    invalid = (~np.isfinite(values)) | (values <= 0.0) | (values >= max_depth_m)
    values[invalid] = 0.0
    return np.ascontiguousarray(values, dtype=np.float32)


def prepare_rgbd_input(
    color_bgr: np.ndarray,
    raw_depth: np.ndarray,
    *,
    depth_scale: float = 1000.0,
    max_depth_m: float = 10.0,
    input_size: int = 518,
    resize_method: Literal["lower_bound", "upper_bound"] = "lower_bound",
) -> torch.Tensor:
    """生成 ``1x4xHxW`` float32 tensor；深度通道单位为 meter。"""

    color = np.asarray(color_bgr)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color_bgr must be HxWx3, got {color.shape}")
    if not np.issubdtype(color.dtype, np.integer):
        raise ValueError("color_bgr must use integer 0-255 values")
    metric = metric_depth_from_raw(
        raw_depth,
        depth_scale=depth_scale,
        max_depth_m=max_depth_m,
    )
    if metric.shape != color.shape[:2]:
        raise ValueError(f"RGB/depth spatial mismatch: rgb={color.shape[:2]}, depth={metric.shape}")

    size = output_size(
        color.shape[1],
        color.shape[0],
        input_size=input_size,
        resize_method=resize_method,
    )
    rgb = cv2.cvtColor(color.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb *= np.float32(1.0 / 255.0)
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb_chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    resized_depth = np.ascontiguousarray(
        cv2.resize(metric, size, interpolation=cv2.INTER_NEAREST),
        dtype=np.float32,
    )
    rgbd = np.concatenate((rgb_chw, resized_depth[None]), axis=0)[None]
    return torch.from_numpy(np.ascontiguousarray(rgbd, dtype=np.float32))
