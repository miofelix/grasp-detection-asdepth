"""AS-Depth canonical catalog 的 RGB-D 推理兼容桥接包。"""

import os

# DeFM 的非 xFormers SwiGLU 使用 checkpoint 中的 ``w12/w3`` 参数层级。
os.environ.setdefault("XFORMERS_DISABLED", "1")

from .api import (
    DEFAULT_MODEL_ID,
    DepthCheckpointReport,
    DepthModelInfo,
    LoadedDepthModel,
    list_depth_models,
    load_depth_model,
    predict_depth,
)
from .checkpoint import CheckpointLoadReport, extract_state_dict, load_checkpoint
from .preprocess import prepare_rgbd_input

__all__ = [
    "CheckpointLoadReport",
    "DEFAULT_MODEL_ID",
    "DepthCheckpointReport",
    "DepthModelInfo",
    "LoadedDepthModel",
    "extract_state_dict",
    "list_depth_models",
    "load_checkpoint",
    "load_depth_model",
    "predict_depth",
    "prepare_rgbd_input",
]
