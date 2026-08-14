"""旧 AS-Depth-2 单文件 checkpoint 的兼容加载器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

StateDict = dict[str, torch.Tensor]
STATE_KEYS = ("model", "state_dict", "model_state_dict", "net")
STRIP_PREFIXES = ("module.", "state_dict.")


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    path: Path
    source_key: str
    tensor_count: int
    stripped_prefixes: tuple[str, ...]


def _tensor_mapping(payload: Mapping[Any, Any], *, context: str) -> StateDict:
    state_dict: StateDict = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"{context} is not a pure string-to-tensor state_dict")
        state_dict[key] = value
    if not state_dict:
        raise ValueError(f"{context} state_dict is empty")
    return state_dict


def extract_state_dict(payload: Any) -> tuple[StateDict, str]:
    """从常见 PyTorch checkpoint envelope 提取纯 state_dict。"""

    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload).__name__}")
    for key in STATE_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return _tensor_mapping(candidate, context=key), key
    return _tensor_mapping(payload, context="root"), "root"


def _strip_prefixes(state_dict: StateDict) -> tuple[StateDict, tuple[str, ...]]:
    normalized: StateDict = {}
    used: list[str] = []
    for source_key, tensor in state_dict.items():
        target_key = source_key
        for prefix in STRIP_PREFIXES:
            if target_key.startswith(prefix):
                target_key = target_key[len(prefix) :]
                if prefix not in used:
                    used.append(prefix)
        if target_key in normalized:
            raise ValueError(
                f"checkpoint key collision after prefix normalization: {source_key!r} -> {target_key!r}"
            )
        normalized[target_key] = tensor
    return normalized, tuple(used)


def _torch_load(path: Path, *, trusted_pickle: bool) -> Any:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": not trusted_pickle,
        "mmap": True,
    }
    try:
        return torch.load(path, **kwargs)
    except RuntimeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def load_checkpoint(
    checkpoint: str | Path,
    *,
    trusted_pickle: bool = False,
) -> tuple[StateDict, CheckpointLoadReport]:
    """读取 checkpoint，并返回规范化 state_dict 与不持有 tensor 的报告。"""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_state, source_key = extract_state_dict(_torch_load(path, trusted_pickle=trusted_pickle))
    state_dict, stripped = _strip_prefixes(raw_state)
    return state_dict, CheckpointLoadReport(
        path=path,
        source_key=source_key,
        tensor_count=len(state_dict),
        stripped_prefixes=stripped,
    )
