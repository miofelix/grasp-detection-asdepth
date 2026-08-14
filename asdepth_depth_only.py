"""离线 RGB-D 深度预测入口。

该入口不导入 AnyGrasp、RealSense 或 Piper，适合在 macOS 上验证
深度模型与预处理。完整抓取流程仍使用 ``asdepth_pipeline.py``。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用离线 RGB-D 图像运行所选深度模型（不需要 AnyGrasp/RealSense/Piper）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--depth-checkpoint", required=True, help="深度模型 checkpoint 路径"
    )
    parser.add_argument(
        "--depth-model",
        choices=["defm_vit_l14_depth", "defm_stackconv_depth"],
        required=True,
        help="checkpoint 对应的模型架构",
    )
    parser.add_argument("--rgb-image", required=True, help="离线 RGB 图像路径")
    parser.add_argument("--depth-image", required=True, help="离线 raw depth 图像路径")
    parser.add_argument(
        "--save-dir", default="debug/asdepth-only", help="运行产物根目录"
    )
    parser.add_argument(
        "--device", default="auto", help="推理设备，例如 mps、cpu、cuda、cuda:0"
    )
    parser.add_argument(
        "--depth-scale", type=float, default=1000.0, help="raw depth 到 meter 的除数"
    )
    parser.add_argument(
        "--max-depth", type=float, default=10.0, help="raw depth 有效上限，单位 meter"
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--resize-method",
        choices=["lower_bound", "upper_bound"],
        default="lower_bound",
    )
    parser.add_argument(
        "--trusted-depth-checkpoint",
        action="store_true",
        help="允许使用 pickle 读取受信任的旧深度模型 checkpoint",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.depth_scale <= 0 or args.max_depth <= 0 or args.input_size <= 0:
        parser.error("--depth-scale, --max-depth and --input-size must be positive")


def _resolve_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _new_run_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir).expanduser().resolve()
    run_dir = root / f"run_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _load_rgbd_files(
    rgb_path: str | Path,
    depth_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"cannot read RGB image: {rgb_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"cannot read raw depth image: {depth_path}")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"raw depth image must be single-channel, got {depth.shape}")
    if depth.shape != color.shape[:2]:
        raise ValueError(
            f"RGB/depth spatial mismatch: rgb={color.shape[:2]}, depth={depth.shape}"
        )
    return (
        np.ascontiguousarray(color, dtype=np.uint8),
        np.ascontiguousarray(depth),
    )


def _clear_model_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
    except ImportError:
        return


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)


def run(args: argparse.Namespace) -> dict[str, str]:
    depth_checkpoint = _resolve_file(
        args.depth_checkpoint, label="depth model checkpoint"
    )
    rgb_path = _resolve_file(args.rgb_image, label="RGB image")
    depth_path = _resolve_file(args.depth_image, label="raw depth image")
    run_dir = _new_run_dir(args.save_dir)
    color_bgr, raw_depth = _load_rgbd_files(rgb_path, depth_path)

    from asdepth_depth import load_depth_model, predict_depth

    load_started = time.perf_counter()
    loaded = load_depth_model(
        depth_checkpoint,
        model_id=args.depth_model,
        device=args.device,
        trusted_pickle=args.trusted_depth_checkpoint,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    checkpoint_report = loaded.checkpoint
    resolved_device = str(loaded.device)
    resolved_model_id = loaded.model_id
    depth_started = time.perf_counter()
    try:
        prediction = predict_depth(
            loaded,
            color_bgr,
            raw_depth,
            depth_scale=args.depth_scale,
            max_depth_m=args.max_depth,
            input_size=args.input_size,
            resize_method=args.resize_method,
        )
    finally:
        del loaded
        _clear_model_cache()
    depth_ms = (time.perf_counter() - depth_started) * 1000.0

    prediction_path = run_dir / "pred_depth.npy"
    np.save(prediction_path, prediction)
    metadata_path = run_dir / "run_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "mode": "asdepth_depth_only",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": resolved_model_id,
        "depth_checkpoint": str(depth_checkpoint),
        "depth_checkpoint_source_key": checkpoint_report.source_key,
        "depth_checkpoint_tensor_count": checkpoint_report.tensor_count,
        "depth_checkpoint_stripped_prefixes": list(checkpoint_report.stripped_prefixes),
        "rgb_image": str(rgb_path),
        "raw_depth_image": str(depth_path),
        "prediction": str(prediction_path),
        "input_shape": list(color_bgr.shape[:2]),
        "prediction_shape": list(prediction.shape),
        "prediction_dtype": str(prediction.dtype),
        "prediction_unit": "meter",
        "device": resolved_device,
        "depth_scale": float(args.depth_scale),
        "max_depth_m": float(args.max_depth),
        "input_size": int(args.input_size),
        "resize_method": args.resize_method,
        "timings_ms": {
            "depth_model_load": load_ms,
            "depth_inference": depth_ms,
        },
    }
    _write_metadata(metadata_path, metadata)
    return {
        "run_dir": str(run_dir),
        "prediction": str(prediction_path),
        "metadata": str(metadata_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        result = run(args)
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"asdepth_depth_only: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
