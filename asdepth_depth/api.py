"""RGB-D 深度模型加载与内存推理 API。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from .checkpoint import CheckpointLoadReport, load_checkpoint
from .models import DeFMRGBDDepth, DeFMStackConvRGBDDepth
from .preprocess import prepare_rgbd_input

DepthModelId = Literal["defm_vit_l14_depth", "defm_stackconv_depth"]
SUPPORTED_MODEL_IDS: tuple[DepthModelId, ...] = (
    "defm_vit_l14_depth",
    "defm_stackconv_depth",
)


@dataclass(frozen=True, slots=True)
class LoadedDepthModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint: CheckpointLoadReport
    model_id: DepthModelId


def _device(value: str | torch.device | None) -> torch.device:
    if value is not None and str(value).lower() != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _requested_model_id(model_id: str) -> DepthModelId:
    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(
            f"unsupported depth model {model_id!r}; choose one of {', '.join(SUPPORTED_MODEL_IDS)}"
        )
    return cast(DepthModelId, model_id)


def _create_model(model_id: DepthModelId) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "encoder": "vitl",
        "features": 256,
        "out_channels": (256, 512, 1024, 1024),
        "pretrained": False,
        "depth_pretrained": False,
    }
    if model_id == "defm_vit_l14_depth":
        return DeFMRGBDDepth(**kwargs)
    return DeFMStackConvRGBDDepth(**kwargs)


def load_depth_model(
    checkpoint: str | Path,
    *,
    model_id: str,
    device: str | torch.device | None = "auto",
    trusted_pickle: bool = False,
) -> LoadedDepthModel:
    """按显式 model ID 严格加载受支持的 DeFM checkpoint。"""

    active_device = _device(device)
    resolved_model_id = _requested_model_id(model_id)
    state_dict, report = load_checkpoint(checkpoint, trusted_pickle=trusted_pickle)
    model = _create_model(resolved_model_id)
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"depth checkpoint is incompatible with {resolved_model_id}: "
                f"missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
    finally:
        del state_dict
        gc.collect()
    return LoadedDepthModel(
        model=model.to(active_device).eval(),
        device=active_device,
        checkpoint=report,
        model_id=resolved_model_id,
    )


def _primary_depth(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        if not value:
            raise ValueError("depth model returned an empty mapping")
        value = next(iter(value.values()))
    elif isinstance(value, tuple | list):
        if not value:
            raise ValueError("depth model returned an empty sequence")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"depth model output must be a tensor, got {type(value).__name__}")
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"depth model output must be 1xHxW or 1x1xHxW, got {tuple(value.shape)}")
    return value


def predict_depth(
    loaded_model: LoadedDepthModel,
    color_bgr: np.ndarray,
    raw_depth: np.ndarray,
    *,
    depth_scale: float = 1000.0,
    max_depth_m: float = 10.0,
    input_size: int = 518,
    resize_method: Literal["lower_bound", "upper_bound"] = "lower_bound",
) -> np.ndarray:
    """输出与相机帧同尺寸的 finite、非负、meter ``float32`` 深度。"""

    original_shape = tuple(int(value) for value in np.asarray(color_bgr).shape[:2])
    inputs = prepare_rgbd_input(
        color_bgr,
        raw_depth,
        depth_scale=depth_scale,
        max_depth_m=max_depth_m,
        input_size=input_size,
        resize_method=resize_method,
    ).to(loaded_model.device, non_blocking=loaded_model.device.type == "cuda")
    with torch.inference_mode():
        prediction = _primary_depth(loaded_model.model(inputs))[0]
    if tuple(prediction.shape) != original_shape:
        prediction = torch.nn.functional.interpolate(
            prediction[None, None],
            original_shape,
            mode="nearest",
        )[0, 0]
    result = np.ascontiguousarray(prediction.detach().cpu().numpy(), dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    result[result < 0.0] = 0.0
    return result
