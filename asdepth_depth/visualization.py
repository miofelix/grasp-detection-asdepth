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
POINT_CLOUD_PAD_RATIO = 0.3
POINT_CLOUD_ROT_X_DEG = 25.0
POINT_CLOUD_ROT_Y_DEG = 15.0
POINT_CLOUD_BACKGROUND_BGR = (255, 255, 255)
POINT_CLOUD_VIEW = "depth_reproject"


@dataclass(frozen=True)
class DepthVisualizationArtifacts:
    raw_depth_path: Path
    prediction_path: Path
    raw_point_cloud_path: Path
    prediction_point_cloud_path: Path
    raw_point_count: int
    prediction_point_count: int
    point_cloud_canvas_height: int
    point_cloud_canvas_width: int
    min_depth_m: float
    max_depth_m: float
    percentile_min: float = VIS_PERCENTILE_MIN
    percentile_max: float = VIS_PERCENTILE_MAX
    colormap: str = VIS_COLORMAP
    point_cloud_view: str = POINT_CLOUD_VIEW
    point_cloud_rot_x_deg: float = POINT_CLOUD_ROT_X_DEG
    point_cloud_rot_y_deg: float = POINT_CLOUD_ROT_Y_DEG


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


def stable_point_cloud_canvas_hw(height: int, width: int) -> tuple[int, int]:
    """Return the padded fixed canvas used by AS-Depth-Research point-cloud views."""

    if height < 1 or width < 1:
        raise ValueError("image height and width must be positive")
    pad = int(max(height, width) * POINT_CLOUD_PAD_RATIO)
    return height + 2 * pad, width + 2 * pad


def _depth_to_points_and_colors(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    color_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = depth.shape
    if color_bgr.shape != (height, width, 3):
        raise ValueError(
            f"RGB/depth shapes differ: rgb={color_bgr.shape[:2]}, depth={depth.shape}"
        )
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if not np.isfinite(fx) or not np.isfinite(fy) or fx == 0.0 or fy == 0.0:
        raise ValueError("intrinsics fx/fy must be finite and non-zero")
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    valid = np.isfinite(depth) & (depth > 1e-8)
    if not valid.any():
        return None
    z = depth[valid]
    x = (columns[valid] - cx) * z / fx
    y = (rows[valid] - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    colors = np.clip(color_bgr, 0, 255).astype(np.uint8, copy=False)[valid]
    return points, colors


def _rotate_point_cloud(
    points: np.ndarray,
    rot_x_deg: float,
    rot_y_deg: float,
) -> np.ndarray:
    center = points.mean(axis=0)
    centered = points - center
    rot_x, rot_y = np.radians(rot_x_deg), np.radians(rot_y_deg)
    cos_x, sin_x = np.cos(rot_x), np.sin(rot_x)
    cos_y, sin_y = np.cos(rot_y), np.sin(rot_y)
    x_after_x = centered[:, 0]
    y_after_x = centered[:, 1] * cos_x - centered[:, 2] * sin_x
    z_after_x = centered[:, 1] * sin_x + centered[:, 2] * cos_x
    x_after_y = x_after_x * cos_y + z_after_x * sin_y
    y_after_y = y_after_x
    z_after_y = -x_after_x * sin_y + z_after_x * cos_y
    return np.asarray(
        np.stack((x_after_y, y_after_y, z_after_y), axis=-1) + center,
        dtype=np.float32,
    )


def render_pointcloud_reproject(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    color_bgr: np.ndarray,
    *,
    rot_x_deg: float = POINT_CLOUD_ROT_X_DEG,
    rot_y_deg: float = POINT_CLOUD_ROT_Y_DEG,
    bg_color: tuple[int, int, int] = POINT_CLOUD_BACKGROUND_BGR,
) -> np.ndarray:
    """Back-project, rotate and reproject depth using AS-Depth-Research semantics."""

    depth = np.asarray(depth_map, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth must squeeze to 2D, got {depth.shape}")
    matrix = np.asarray(intrinsics, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got {matrix.shape}")
    colors = np.asarray(color_bgr, dtype=np.uint8)
    height, width = depth.shape
    canvas_height, canvas_width = stable_point_cloud_canvas_hw(height, width)
    pad = int(max(height, width) * POINT_CLOUD_PAD_RATIO)
    projected = _depth_to_points_and_colors(depth, matrix, colors)
    if projected is None:
        return np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
    points, point_colors = projected
    rotated = _rotate_point_cloud(points, rot_x_deg, rot_y_deg)

    z = rotated[:, 2]
    keep = z > 1e-4
    if not keep.any():
        return np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    projected_u = rotated[keep, 0] * fx / z[keep] + cx
    projected_v = rotated[keep, 1] * fy / z[keep] + cy
    projected_z = z[keep]
    projected_colors = point_colors[keep]
    pixel_u = np.rint(projected_u + pad).astype(np.int32)
    pixel_v = np.rint(projected_v + pad).astype(np.int32)
    in_bounds = (
        (pixel_u >= 0)
        & (pixel_u < canvas_width)
        & (pixel_v >= 0)
        & (pixel_v < canvas_height)
    )
    pixel_u = pixel_u[in_bounds]
    pixel_v = pixel_v[in_bounds]
    projected_z = projected_z[in_bounds]
    projected_colors = projected_colors[in_bounds]
    order = np.argsort(-projected_z)
    pixel_u = pixel_u[order]
    pixel_v = pixel_v[order]
    projected_colors = projected_colors[order]

    canvas = np.full((canvas_height, canvas_width, 3), bg_color, dtype=np.uint8)
    filled = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for delta_u in range(2):
        for delta_v in range(2):
            splat_u = np.clip(pixel_u + delta_u, 0, canvas_width - 1)
            splat_v = np.clip(pixel_v + delta_v, 0, canvas_height - 1)
            canvas[splat_v, splat_u] = projected_colors
            filled[splat_v, splat_u] = 255

    dilated = cv2.dilate(filled, np.ones((3, 3), dtype=np.uint8), iterations=1)
    holes = (dilated > 0) & (filled == 0)
    if holes.any():
        for channel in range(3):
            blurred = cv2.blur(canvas[:, :, channel].astype(np.float32), (5, 5))
            canvas[:, :, channel][holes] = blurred[holes].astype(np.uint8)
    return np.asarray(canvas, dtype=np.uint8)


def save_depth_visualizations(
    run_dir: str | Path,
    raw_depth: np.ndarray,
    prediction: np.ndarray,
    *,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    max_depth_m: float,
) -> DepthVisualizationArtifacts:
    """保存共享色标深度图和相机坐标系 RGB 三维点云 PNG。"""

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
    intrinsics = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    raw_point_depth = np.where(
        np.isfinite(raw_metric_depth)
        & (raw_metric_depth > 0.0)
        & (raw_metric_depth < max_depth_m),
        raw_metric_depth,
        0.0,
    ).astype(np.float32)
    prediction_point_depth = np.where(
        np.isfinite(predicted_depth)
        & (predicted_depth > 0.0)
        & (predicted_depth < max_depth_m),
        predicted_depth,
        0.0,
    ).astype(np.float32)
    raw_point_count = int(np.count_nonzero(raw_point_depth))
    prediction_point_count = int(np.count_nonzero(prediction_point_depth))
    raw_point_cloud = render_pointcloud_reproject(
        raw_point_depth,
        intrinsics,
        color_bgr,
    )
    prediction_point_cloud = render_pointcloud_reproject(
        prediction_point_depth,
        intrinsics,
        color_bgr,
    )

    raw_depth_path = output_dir / "raw_depth_vis.png"
    prediction_path = output_dir / "pred_depth_vis.png"
    raw_point_cloud_path = output_dir / "raw_point_cloud_vis.png"
    prediction_point_cloud_path = output_dir / "pred_point_cloud_vis.png"
    _write_png(raw_depth_path, raw_visualization)
    _write_png(prediction_path, prediction_visualization)
    _write_png(raw_point_cloud_path, raw_point_cloud)
    _write_png(prediction_point_cloud_path, prediction_point_cloud)
    return DepthVisualizationArtifacts(
        raw_depth_path=raw_depth_path,
        prediction_path=prediction_path,
        raw_point_cloud_path=raw_point_cloud_path,
        prediction_point_cloud_path=prediction_point_cloud_path,
        raw_point_count=raw_point_count,
        prediction_point_count=prediction_point_count,
        point_cloud_canvas_height=raw_point_cloud.shape[0],
        point_cloud_canvas_width=raw_point_cloud.shape[1],
        min_depth_m=vis_min_depth_m,
        max_depth_m=vis_max_depth_m,
    )


__all__ = [
    "DepthVisualizationArtifacts",
    "POINT_CLOUD_BACKGROUND_BGR",
    "POINT_CLOUD_PAD_RATIO",
    "POINT_CLOUD_ROT_X_DEG",
    "POINT_CLOUD_ROT_Y_DEG",
    "POINT_CLOUD_VIEW",
    "colorize_metric_depth",
    "depth_visualization_range",
    "render_pointcloud_reproject",
    "save_depth_visualizations",
    "stable_point_cloud_canvas_hw",
]
