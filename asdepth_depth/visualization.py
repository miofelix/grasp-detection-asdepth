"""米制深度图的彩色可视化与运行产物保存。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .preprocess import metric_depth_from_raw

VIS_PERCENTILE_MIN = 2.0
VIS_PERCENTILE_MAX = 98.0
VIS_COLORMAP = "turbo"


@dataclass(frozen=True)
class DepthVisualizationArtifacts:
    raw_depth_path: Path
    prediction_path: Path
    min_depth_m: float
    max_depth_m: float
    percentile_min: float = VIS_PERCENTILE_MIN
    percentile_max: float = VIS_PERCENTILE_MAX
    colormap: str = VIS_COLORMAP


def depth_visualization_range(
    *depth_maps: np.ndarray,
    max_depth_m: float,
    percentile_min: float = VIS_PERCENTILE_MIN,
    percentile_max: float = VIS_PERCENTILE_MAX,
) -> tuple[float, float]:
    """按多张深度图的联合有效像素返回共享可视化范围。"""

    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    if not 0.0 <= percentile_min <= percentile_max <= 100.0:
        raise ValueError("visualization percentiles must satisfy 0 <= min <= max <= 100")

    valid_parts: list[np.ndarray] = []
    for depth_map in depth_maps:
        values = np.asarray(depth_map, dtype=np.float32).squeeze()
        if values.ndim != 2:
            raise ValueError(f"depth must squeeze to 2D, got {values.shape}")
        valid = np.isfinite(values) & (values > 0.0) & (values < max_depth_m)
        if valid.any():
            valid_parts.append(values[valid])

    if not valid_parts:
        return 0.0, float(max_depth_m)

    valid_values = np.concatenate(valid_parts)
    lower, upper = np.percentile(valid_values, [percentile_min, percentile_max])
    minimum = max(float(lower), 0.0)
    maximum = min(float(upper), float(max_depth_m))
    if maximum <= minimum:
        maximum = min(float(max_depth_m), minimum + 1e-6)
    if maximum <= minimum:
        minimum = max(0.0, maximum - 1e-6)
    return minimum, maximum


def colorize_metric_depth(
    depth: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """把二维米制深度上色为 OpenCV BGR uint8，无效区域固定为黑色。"""

    values = np.asarray(depth, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"depth must squeeze to 2D, got {values.shape}")
    if max_depth_m <= min_depth_m:
        raise ValueError("max_depth_m must be greater than min_depth_m")

    valid = np.isfinite(values) & (values > 0.0)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[valid] = np.clip(
        (values[valid] - np.float32(min_depth_m)) / np.float32(max_depth_m - min_depth_m),
        0.0,
        1.0,
    )
    grayscale = np.rint(normalized * np.float32(255.0)).astype(np.uint8)
    colored = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return np.ascontiguousarray(colored, dtype=np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write depth visualization: {path}")


def save_depth_visualizations(
    run_dir: str | Path,
    raw_depth: np.ndarray,
    prediction: np.ndarray,
    *,
    depth_scale: float,
    max_depth_m: float,
) -> DepthVisualizationArtifacts:
    """用共享色标保存原始和预测深度图的彩色 PNG。"""

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_metric_depth = metric_depth_from_raw(
        raw_depth,
        depth_scale=depth_scale,
        max_depth_m=max_depth_m,
    )
    predicted_depth = np.asarray(prediction, dtype=np.float32).squeeze()
    if predicted_depth.ndim != 2:
        raise ValueError(f"prediction must squeeze to 2D, got {predicted_depth.shape}")
    if predicted_depth.shape != raw_metric_depth.shape:
        raise ValueError(
            "raw/predicted depth shapes differ: "
            f"raw={raw_metric_depth.shape}, prediction={predicted_depth.shape}"
        )

    vis_min_depth_m, vis_max_depth_m = depth_visualization_range(
        raw_metric_depth,
        predicted_depth,
        max_depth_m=max_depth_m,
    )
    raw_visualization = colorize_metric_depth(
        raw_metric_depth,
        min_depth_m=vis_min_depth_m,
        max_depth_m=vis_max_depth_m,
    )
    prediction_visualization = colorize_metric_depth(
        predicted_depth,
        min_depth_m=vis_min_depth_m,
        max_depth_m=vis_max_depth_m,
    )

    raw_depth_path = output_dir / "raw_depth_vis.png"
    prediction_path = output_dir / "pred_depth_vis.png"
    _write_png(raw_depth_path, raw_visualization)
    _write_png(prediction_path, prediction_visualization)
    return DepthVisualizationArtifacts(
        raw_depth_path=raw_depth_path,
        prediction_path=prediction_path,
        min_depth_m=vis_min_depth_m,
        max_depth_m=vis_max_depth_m,
    )


__all__ = [
    "DepthVisualizationArtifacts",
    "colorize_metric_depth",
    "depth_visualization_range",
    "save_depth_visualizations",
]
