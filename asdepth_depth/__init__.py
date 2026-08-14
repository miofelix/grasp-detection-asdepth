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

__all__ = [
    "SUPPORTED_MODEL_IDS",
    "CheckpointLoadReport",
    "LoadedDepthModel",
    "extract_state_dict",
    "load_checkpoint",
    "load_depth_model",
    "predict_depth",
    "prepare_rgbd_input",
]
