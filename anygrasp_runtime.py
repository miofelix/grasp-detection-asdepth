"""AnyGrasp 2026 SDK 的本地二进制加载与推理适配。"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import sysconfig
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def matching_gsnet_path(project_root: str | Path = PROJECT_ROOT) -> Path:
    """返回与当前 CPython ABI 匹配的官方 Linux x86-64 GSNet 扩展。"""

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("AnyGrasp GSNet requires Linux x86-64")

    versions_dir = Path(project_root).expanduser().resolve() / "gsnet_versions"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if extension_suffix:
        candidate = versions_dir / f"gsnet{extension_suffix}"
        if candidate.is_file():
            return candidate

    python_tag = f"{sys.version_info.major}{sys.version_info.minor}"
    candidates = sorted(versions_dir.glob(f"gsnet.cpython-{python_tag}*.so"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "no matching AnyGrasp GSNet binary for CPython "
            f"{sys.version_info.major}.{sys.version_info.minor} under {versions_dir}"
        )
    raise RuntimeError(
        "multiple AnyGrasp GSNet binaries match CPython "
        f"{sys.version_info.major}.{sys.version_info.minor}: {candidates}"
    )


def load_gsnet_module(project_root: str | Path = PROJECT_ROOT) -> ModuleType:
    """以模块名 ``gsnet`` 加载当前解释器对应的版本化扩展。"""

    binary = matching_gsnet_path(project_root)
    existing = sys.modules.get("gsnet")
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if existing_file != binary.resolve():
            raise RuntimeError(
                f"gsnet is already loaded from {existing_file}; expected {binary.resolve()}"
            )
        return existing

    spec = importlib.util.spec_from_file_location("gsnet", binary)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for GSNet binary: {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gsnet"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("gsnet", None)
        raise
    return module


def validate_license_dir(project_root: str | Path = PROJECT_ROOT) -> Path:
    """校验新版 AnyGrasp 许可证是否位于仓库根目录的 ``license/``。"""

    license_dir = Path(project_root).expanduser().resolve() / "license"
    if not (license_dir / "licenseCfg.json").is_file():
        raise FileNotFoundError(
            "AnyGrasp license is missing; put the new license folder at "
            f"{license_dir} (it must contain licenseCfg.json)"
        )
    return license_dir


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def create_detector(config: Any, project_root: str | Path = PROJECT_ROOT) -> Any:
    """加载 SDK、校验许可证并创建新版 AnyGrasp detector。"""

    root = Path(project_root).expanduser().resolve()
    validate_license_dir(root)
    module = load_gsnet_module(root)
    factory = getattr(module, "create_detector", None)
    if not callable(factory):
        raise RuntimeError("unsupported GSNet binary: create_detector is unavailable")
    with _working_directory(root):
        detector = factory(config)
    if detector is None:
        raise RuntimeError("AnyGrasp create_detector failed; check the license and checkpoint")
    return detector


def workspace_mask(points: np.ndarray, limits: Sequence[float]) -> np.ndarray:
    """把旧版六轴 workspace limits 转成新版 region steering mask。"""

    point_array = np.asarray(points)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {point_array.shape}")
    if len(limits) != 6:
        raise ValueError(f"workspace limits must contain six values, got {len(limits)}")
    xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in limits)
    if xmin > xmax or ymin > ymax or zmin > zmax:
        raise ValueError(f"invalid workspace limits: {list(limits)}")
    return (
        (point_array[:, 0] >= xmin)
        & (point_array[:, 0] <= xmax)
        & (point_array[:, 1] >= ymin)
        & (point_array[:, 1] <= ymax)
        & (point_array[:, 2] >= zmin)
        & (point_array[:, 2] <= zmax)
    )


def predict_grasps(
    detector: Any,
    points: np.ndarray,
    limits: Sequence[float],
    *,
    top_down_grasp: bool,
    approach_thresh: float = np.pi / 6,
    dense_grasp: bool = False,
    collision_detection: bool = True,
) -> Any | None:
    """调用新版 steering API，并返回经过 NMS 和分数排序的 GraspGroup。"""

    region = workspace_mask(points, limits)
    if not np.any(region):
        raise RuntimeError("no valid points fall inside the configured grasp workspace")
    options = {
        "dense_grasp": bool(dense_grasp),
        "collision_detection": bool(collision_detection),
        "region_steering": region,
        "approach_steering": [0.0, 0.0, 1.0] if top_down_grasp else None,
        "approach_thresh": float(approach_thresh),
    }
    grasps = detector.get_grasp(np.asarray(points, dtype=np.float32), options)
    if grasps is None or len(grasps) == 0:
        return None
    if not dense_grasp:
        grasps = grasps.nms()
    grasps = grasps.sort_by_score()
    return grasps if len(grasps) > 0 else None
