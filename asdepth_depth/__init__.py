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
    DepthVisualizationArtifacts,
    colorize_metric_depth,
    depth_visualization_range,
    save_depth_visualizations,
)

__all__ = [
    "SUPPORTED_MODEL_IDS",
    "CheckpointLoadReport",
    "DepthVisualizationArtifacts",
    "LoadedDepthModel",
    "colorize_metric_depth",
    "depth_visualization_range",
    "extract_state_dict",
    "load_checkpoint",
    "load_depth_model",
    "predict_depth",
    "prepare_rgbd_input",
    "save_depth_visualizations",
]
