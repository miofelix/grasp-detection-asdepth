"""RGB-D 深度模型推理桥接包。"""

import os

# DeFM 的非 xFormers SwiGLU 使用 checkpoint 中的 ``w12/w3`` 参数层级。
os.environ.setdefault("XFORMERS_DISABLED", "1")

from .api import (
    SUPPORTED_MODEL_IDS,
    LoadedDepthModel,
    load_depth_model,
    predict_depth,
)
from .checkpoint import CheckpointLoadReport, extract_state_dict, load_checkpoint
from .preprocess import prepare_rgbd_input
from .visualization import (
    POINT_CLOUD_BACKGROUND_BGR,
    POINT_CLOUD_PAD_RATIO,
    POINT_CLOUD_ROT_X_DEG,
    POINT_CLOUD_ROT_Y_DEG,
    POINT_CLOUD_VIEW,
    DepthVisualizationArtifacts,
    colorize_metric_depth,
    depth_visualization_range,
    render_pointcloud_reproject,
    save_depth_visualizations,
    stable_point_cloud_canvas_hw,
)

__all__ = [
    "SUPPORTED_MODEL_IDS",
    "CheckpointLoadReport",
    "DepthVisualizationArtifacts",
    "LoadedDepthModel",
    "POINT_CLOUD_BACKGROUND_BGR",
    "POINT_CLOUD_PAD_RATIO",
    "POINT_CLOUD_ROT_X_DEG",
    "POINT_CLOUD_ROT_Y_DEG",
    "POINT_CLOUD_VIEW",
    "colorize_metric_depth",
    "depth_visualization_range",
    "extract_state_dict",
    "load_checkpoint",
    "load_depth_model",
    "predict_depth",
    "prepare_rgbd_input",
    "render_pointcloud_reproject",
    "save_depth_visualizations",
    "stable_point_cloud_canvas_hw",
]
