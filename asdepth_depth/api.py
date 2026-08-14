"""AS-Depth catalog 模型加载与内存 RGB-D 推理兼容 API。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from .preprocess import metric_depth_from_raw, prepare_rgbd_input


DEFAULT_MODEL_ID = "defm_stackconv_depth"


@dataclass(frozen=True, slots=True)
class DepthModelInfo:
    """可供当前抓取项目选择的 canonical AS-Depth 模型信息。"""

    model_id: str
    model_version: str
    config_hash: str
    entrypoint: str
    native_depth: str
    sparse_raw_depth: bool


@dataclass(frozen=True, slots=True)
class DepthCheckpointReport:
    """把正式 `asdepth` checkpoint 报告收敛为本项目稳定元数据。"""

    path: Path
    source_key: str
    tensor_count: int
    stripped_prefixes: tuple[str, ...] = ()
    step: int | None = None
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    resolution_source: str = "explicit_model"


@dataclass(frozen=True, slots=True)
class LoadedDepthModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint: Any
    model_id: str = DEFAULT_MODEL_ID
    model_version: str = "legacy-snapshot"
    config_hash: str | None = None
    native_depth: str = "metric_depth"
    sparse_raw_depth: bool = False
    resolution_source: str = "legacy_snapshot"
    spec: Any | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "native_depth": self.native_depth,
            "sparse_raw_depth": self.sparse_raw_depth,
            "resolution_source": self.resolution_source,
        }


def list_depth_models() -> tuple[DepthModelInfo, ...]:
    """列出随 `as-depth` 0.3.0 wheel 安装的全部活跃模型。"""

    from asdepth.models import ModelCatalog

    result: list[DepthModelInfo] = []
    for entry in ModelCatalog.from_package().entries:
        spec = entry.to_model_spec()
        result.append(
            DepthModelInfo(
                model_id=spec.model_id,
                model_version=spec.model_version,
                config_hash=spec.config_hash,
                entrypoint=spec.entrypoint,
                native_depth=spec.depth.representation.value,
                sparse_raw_depth=bool(spec.metadata.get("is_sparse_raw_depth", False)),
            )
        )
    return tuple(result)


def load_depth_model(
    checkpoint: str | Path,
    *,
    model_id: str | None = DEFAULT_MODEL_ID,
    device: str | torch.device | None = "auto",
    trusted_pickle: bool = False,
    verify_checksums: bool = True,
    strict: bool = True,
) -> LoadedDepthModel:
    """通过 canonical catalog 加载任意活跃 AS-Depth RGB-D 模型。"""

    from asdepth.inference import load_inference_model

    requested_model = None if model_id in {None, "", "auto"} else model_id
    loaded = load_inference_model(
        checkpoint,
        model_id=requested_model,
        device=device,
        strict=strict,
        verify_checksums=verify_checksums,
        trusted_pickle=trusted_pickle,
    )
    source_checkpoint = loaded.report.checkpoint
    checkpoint_report = DepthCheckpointReport(
        path=source_checkpoint.path,
        source_key=source_checkpoint.checkpoint_format.value,
        tensor_count=len(source_checkpoint.state_dict),
        step=source_checkpoint.step,
        missing_keys=loaded.report.missing_keys,
        unexpected_keys=loaded.report.unexpected_keys,
        resolution_source=loaded.report.resolution.source.value,
    )
    gc.collect()
    return LoadedDepthModel(
        model=loaded.model,
        device=loaded.device,
        checkpoint=checkpoint_report,
        model_id=loaded.spec.model_id,
        model_version=loaded.spec.model_version,
        config_hash=loaded.spec.config_hash,
        native_depth=loaded.spec.depth.representation.value,
        sparse_raw_depth=loaded.is_sparse_raw_depth,
        resolution_source=loaded.report.resolution.source.value,
        spec=loaded.spec,
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


def _legacy_predict_depth(
    loaded_model: LoadedDepthModel,
    color_bgr: np.ndarray,
    raw_depth: np.ndarray,
    *,
    depth_scale: float,
    max_depth_m: float,
    input_size: int,
    resize_method: Literal["lower_bound", "upper_bound"],
) -> np.ndarray:
    """兼容旧测试和调用方直接构造 ``LoadedDepthModel`` 的推理行为。"""

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
    return np.ascontiguousarray(prediction.detach().cpu().numpy(), dtype=np.float32)


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
    """统一输出与相机帧同尺寸的 finite、非负、meter `float32` 深度。"""

    color = np.asarray(color_bgr)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"color_bgr must be HxWx3, got {color.shape}")
    if not np.issubdtype(color.dtype, np.integer):
        raise ValueError("color_bgr must use integer 0-255 values")

    if loaded_model.spec is None:
        result = _legacy_predict_depth(
            loaded_model,
            color,
            raw_depth,
            depth_scale=depth_scale,
            max_depth_m=max_depth_m,
            input_size=input_size,
            resize_method=resize_method,
        )
    else:
        from asdepth.core import DepthMap, DepthRepresentation
        from asdepth.inference import infer_rgbd_images

        metric_input = metric_depth_from_raw(
            raw_depth,
            depth_scale=depth_scale,
            max_depth_m=max_depth_m,
        )
        if metric_input.shape != color.shape[:2]:
            raise ValueError(
                f"RGB/depth spatial mismatch: rgb={color.shape[:2]}, depth={metric_input.shape}"
            )
        native_spec = loaded_model.spec.depth
        if native_spec.representation is DepthRepresentation.METRIC_DEPTH:
            native_input = metric_input
        elif native_spec.representation is DepthRepresentation.INVERSE_DEPTH:
            native_input = DepthMap.from_metric_array(metric_input).to_inverse().values
        else:
            raise ValueError(
                f"unsupported RGB-D model depth representation: {native_spec.representation.value}"
            )
        native_input = np.array(native_input, dtype=np.float32, copy=True, order="C")
        color_rgb = np.ascontiguousarray(
            cv2.cvtColor(color.astype(np.uint8, copy=False), cv2.COLOR_BGR2RGB)
        )
        native_prediction = infer_rgbd_images(
            loaded_model.model,
            [color_rgb],
            [native_input],
            input_size=input_size,
            resize_method=resize_method,
            sparse_raw_depth=loaded_model.sparse_raw_depth,
            device=loaded_model.device,
        )[0]
        result = np.array(
            DepthMap(native_prediction, native_spec).to_metric().values,
            dtype=np.float32,
            copy=True,
            order="C",
        )

    result[~np.isfinite(result)] = 0.0
    result[result < 0.0] = 0.0
    return np.ascontiguousarray(result, dtype=np.float32)
