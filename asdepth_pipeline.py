"""Orbbec/RealSense RGB-D → 深度模型 → AnyGrasp → 可选 Piper 的安全入口。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from camera_capture import CameraIntrinsics, CaptureResult

DEFAULT_GRASP_CHECKPOINT = "ckpts/checkpoint_detection.tar"

LEGACY_CAMERA_INTRINSICS = CameraIntrinsics(
    fx=616.22601724,
    fy=615.78839082,
    cx=315.33494299,
    cy=251.59150012,
    width=640,
    height=480,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用本地 RGB-D 深度模型的抓取流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--depth-checkpoint", required=True, help="深度模型 checkpoint 路径")
    parser.add_argument(
        "--depth-model",
        choices=["defm_vit_l14_depth", "defm_stackconv_depth"],
        required=True,
        help="checkpoint 对应的模型架构",
    )
    parser.add_argument(
        "--grasp-checkpoint",
        default=DEFAULT_GRASP_CHECKPOINT,
        help="AnyGrasp 2026 detection checkpoint 路径",
    )
    parser.add_argument("--rgb-image", help="离线 RGB 图像；必须与 --depth-image 同时提供")
    parser.add_argument("--depth-image", help="离线 raw depth 图像；必须与 --rgb-image 同时提供")
    parser.add_argument("--save-dir", default="debug/asdepth", help="运行产物根目录")
    parser.add_argument("--device", default="auto", help="深度模型推理设备，例如 cuda、cuda:0、cpu")
    parser.add_argument(
        "--camera-backend",
        choices=["orbbec", "realsense", "auto"],
        default="orbbec",
        help="在线采集后端；离线图片模式下忽略",
    )
    parser.add_argument("--camera-width", type=int, default=640, help="RealSense 彩色/深度宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="RealSense 彩色/深度高度")
    parser.add_argument("--camera-fps", type=int, default=30, help="RealSense 彩色/深度帧率")
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--camera-timeout", type=float, default=20.0, help="在线采集超时，单位秒")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="每米对应的 raw depth 单位数；在线模式默认读取相机，离线默认 1000",
    )
    parser.add_argument("--camera-fx", type=float, help="覆盖彩色相机内参 fx")
    parser.add_argument("--camera-fy", type=float, help="覆盖彩色相机内参 fy")
    parser.add_argument("--camera-cx", type=float, help="覆盖彩色相机内参 cx")
    parser.add_argument("--camera-cy", type=float, help="覆盖彩色相机内参 cy")
    parser.add_argument(
        "--max-depth", type=float, default=10.0, help="raw depth 有效上限，单位 meter"
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--resize-method",
        choices=["lower_bound", "upper_bound"],
        default="lower_bound",
    )
    parser.add_argument("--max-gripper-width", type=float, default=0.095)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument(
        "--top-down-grasp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--debug", action="store_true", help="开启 AnyGrasp Open3D 可视化")
    parser.add_argument(
        "--execute-arm",
        action="store_true",
        help="真实连接 Piper 并执行运动；还必须同时传 --confirm-arm-motion",
    )
    parser.add_argument(
        "--arm-dry-run",
        action="store_true",
        help="只生成并校验 Piper 运动计划，不连接 CAN 或发送运动命令",
    )
    parser.add_argument(
        "--confirm-arm-motion",
        action="store_true",
        help="第二重确认真实机械臂运动；只在与 --execute-arm 同时使用时有效",
    )
    parser.add_argument(
        "--arm-config",
        default="config/piper_device.json",
        help="Piper 双臂、CAN、手眼标定和安全参数 JSON",
    )
    parser.add_argument(
        "--arm-side",
        choices=["left", "right"],
        default=None,
        help="选择机械臂；默认使用配置文件中的 active_arm",
    )
    parser.add_argument(
        "--arm-can-interface",
        default=None,
        help="临时覆盖配置文件中的 SocketCAN 接口",
    )
    parser.add_argument(
        "--arm-speed-percent",
        type=int,
        default=None,
        help="临时覆盖 Piper 运动速度百分比",
    )
    parser.add_argument(
        "--arm-gripper-max-width",
        type=float,
        default=None,
        help="临时覆盖现场 Piper 夹爪最大行程，单位米",
    )
    parser.add_argument(
        "--arm-min-grasp-score",
        type=float,
        default=None,
        help="临时覆盖允许机械臂真实执行的最低 AnyGrasp 分数；默认读取 arm config",
    )
    parser.add_argument("--arm-tool-offset", type=float, default=None, help="临时覆盖工具偏移")
    parser.add_argument(
        "--arm-pregrasp-clearance",
        type=float,
        default=None,
        help="临时覆盖预抓取点沿工具轴后退距离",
    )
    parser.add_argument(
        "--arm-lift-distance", type=float, default=None, help="临时覆盖抓取后抬升距离"
    )
    parser.add_argument("--arm-max-reach", type=float, default=None, help="临时覆盖最大可达距离")
    parser.add_argument("--arm-min-z", type=float, default=None, help="临时覆盖目标最低基座 Z")
    parser.add_argument("--arm-max-z", type=float, default=None, help="临时覆盖目标最高基座 Z")
    parser.add_argument(
        "--arm-max-abs-x",
        type=float,
        default=None,
        help="临时覆盖基座坐标系 X 轴绝对值上限，单位米",
    )
    parser.add_argument(
        "--arm-max-abs-y",
        type=float,
        default=None,
        help="临时覆盖基座坐标系 Y 轴绝对值上限，单位米",
    )
    parser.add_argument("--arm-enable-timeout", type=float, default=None, help="临时覆盖使能超时，单位秒")
    parser.add_argument("--arm-move-timeout", type=float, default=None, help="临时覆盖单次移动超时，单位秒")
    parser.add_argument("--arm-gripper-timeout", type=float, default=None, help="临时覆盖夹爪动作超时，单位秒")
    parser.add_argument(
        "--arm-position-tolerance",
        type=float,
        default=None,
        help="临时覆盖移动到位位置容差，单位米",
    )
    parser.add_argument(
        "--arm-angle-tolerance",
        type=float,
        default=None,
        help="临时覆盖移动到位角度容差，单位度",
    )
    parser.add_argument(
        "--arm-gripper-tolerance",
        type=float,
        default=None,
        help="临时覆盖夹爪到位容差，单位米",
    )
    parser.add_argument(
        "--trusted-depth-checkpoint",
        action="store_true",
        help="允许使用 pickle 读取受信任的旧深度模型 checkpoint",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if bool(args.rgb_image) != bool(args.depth_image):
        parser.error("--rgb-image and --depth-image must be provided together")
    if args.depth_scale is not None and args.depth_scale <= 0:
        parser.error("--depth-scale must be positive")
    if args.max_depth <= 0 or args.input_size <= 0:
        parser.error("--max-depth and --input-size must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        parser.error("--camera-width, --camera-height and --camera-fps must be positive")
    if args.camera_warmup_frames < 0 or args.camera_timeout <= 0:
        parser.error("--camera-warmup-frames must be non-negative and --camera-timeout positive")
    intrinsic_values = [args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy]
    if any(value is not None for value in intrinsic_values) and not all(
        value is not None for value in intrinsic_values
    ):
        parser.error(
            "--camera-fx, --camera-fy, --camera-cx and --camera-cy must be provided together"
        )
    if args.camera_fx is not None and (args.camera_fx <= 0 or args.camera_fy <= 0):
        parser.error("--camera-fx and --camera-fy must be positive")
    if not 0.0 <= args.max_gripper_width <= 0.1:
        parser.error("--max-gripper-width must be between 0 and 0.1 meter")
    if args.gripper_height <= 0:
        parser.error("--gripper-height must be positive")
    if args.execute_arm and args.arm_dry_run:
        parser.error("--execute-arm and --arm-dry-run are mutually exclusive")
    if args.execute_arm and not args.confirm_arm_motion:
        parser.error("--execute-arm requires --confirm-arm-motion")
    if args.confirm_arm_motion and not args.execute_arm:
        parser.error("--confirm-arm-motion is only valid with --execute-arm")
    if args.arm_can_interface == "":
        parser.error("--arm-can-interface must not be empty")
    if args.arm_speed_percent is not None and not 1 <= args.arm_speed_percent <= 100:
        parser.error("--arm-speed-percent must be between 1 and 100")
    if args.arm_gripper_max_width is not None and not 0 < args.arm_gripper_max_width <= 0.1:
        parser.error("--arm-gripper-max-width must be in (0, 0.1] meter")
    if args.arm_min_grasp_score is not None and not 0 <= args.arm_min_grasp_score <= 1:
        parser.error("--arm-min-grasp-score must be between 0 and 1")
    distances = (
        args.arm_tool_offset,
        args.arm_pregrasp_clearance,
        args.arm_lift_distance,
    )
    if any(value is not None and value <= 0 for value in distances):
        parser.error("arm tool offset, pregrasp clearance and lift distance must be positive")
    if args.arm_max_reach is not None and args.arm_max_reach <= 0:
        parser.error("--arm-max-reach must be positive")
    if args.arm_min_z is not None and args.arm_min_z < 0:
        parser.error("--arm-min-z must be non-negative")
    if args.arm_max_z is not None and args.arm_max_z <= 0:
        parser.error("--arm-max-z must be positive")
    if (
        args.arm_min_z is not None
        and args.arm_max_z is not None
        and args.arm_min_z >= args.arm_max_z
    ):
        parser.error("Piper arm Z range is invalid")
    workspace_limits = (args.arm_max_abs_x, args.arm_max_abs_y)
    if any(value is not None and value <= 0 for value in workspace_limits):
        parser.error("--arm-max-abs-x and --arm-max-abs-y must be positive")
    timeouts = (
        args.arm_enable_timeout,
        args.arm_move_timeout,
        args.arm_gripper_timeout,
    )
    if any(value is not None and value <= 0 for value in timeouts):
        parser.error("arm operation timeouts must be positive")
    tolerances = (
        args.arm_position_tolerance,
        args.arm_angle_tolerance,
        args.arm_gripper_tolerance,
    )
    if any(value is not None and value <= 0 for value in tolerances):
        parser.error("arm position, angle and gripper tolerances must be positive")


def _resolve_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _new_file_run_dir(base_dir: str | Path) -> Path:
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
        raise ValueError(f"RGB/depth spatial mismatch: rgb={color.shape[:2]}, depth={depth.shape}")
    return (
        np.ascontiguousarray(color, dtype=np.uint8),
        np.ascontiguousarray(depth),
    )


def _load_anygrasp_function() -> Callable[..., Any]:
    from anygrasp_runtime import load_gsnet_module, validate_license_dir

    validate_license_dir()
    load_gsnet_module()
    from get_pose import run_anygrasp

    return run_anygrasp


def _capture_camera(args: argparse.Namespace) -> CaptureResult:
    from camera_capture import capture_one_frame

    return capture_one_frame(
        args.save_dir,
        backend=args.camera_backend,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        warmup_frames=args.camera_warmup_frames,
        timeout_s=args.camera_timeout,
    )


def _load_arm_runner() -> Callable[..., Any]:
    from grasp_piper import run_pipeline

    return run_pipeline


def _resolve_arm_min_grasp_score(args: argparse.Namespace) -> tuple[float, str]:
    from grasp_piper import resolve_arm_min_grasp_score

    return resolve_arm_min_grasp_score(
        args.arm_config,
        override=args.arm_min_grasp_score,
    )


def _arm_runner_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device_config_path": args.arm_config,
        "arm_side": args.arm_side,
        "can_name": args.arm_can_interface,
        "motion_speed_percent": args.arm_speed_percent,
        "gripper_max_width_m": args.arm_gripper_max_width,
        "tool_offset_m": args.arm_tool_offset,
        "pregrasp_clearance_m": args.arm_pregrasp_clearance,
        "lift_distance_m": args.arm_lift_distance,
        "max_reach_m": args.arm_max_reach,
        "min_z_m": args.arm_min_z,
        "max_z_m": args.arm_max_z,
        "max_abs_x_m": args.arm_max_abs_x,
        "max_abs_y_m": args.arm_max_abs_y,
        "enable_timeout_s": args.arm_enable_timeout,
        "move_timeout_s": args.arm_move_timeout,
        "gripper_timeout_s": args.arm_gripper_timeout,
        "position_tolerance_m": args.arm_position_tolerance,
        "angle_tolerance_deg": args.arm_angle_tolerance,
        "gripper_tolerance_m": args.arm_gripper_tolerance,
    }


def _clear_model_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _write_grasp_pose(
    path: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    width: float,
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("R_cam:\n")
        stream.write(np.array2string(rotation, precision=6, suppress_small=True))
        stream.write("\n\nt_cam:\n")
        stream.write(np.array2string(translation.reshape(-1), precision=6, suppress_small=True))
        stream.write(f"\n\nwidth: {width}\n")


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)


def _grasp_config(
    args: argparse.Namespace,
    checkpoint: Path,
    intrinsics: CameraIntrinsics,
) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=str(checkpoint),
        max_gripper_width=float(args.max_gripper_width),
        gripper_height=float(args.gripper_height),
        top_down_grasp=bool(args.top_down_grasp),
        debug=bool(args.debug),
        save_dir=str(args.save_dir),
        camera_intrinsics={
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
    )


def _resolve_intrinsics(
    args: argparse.Namespace,
    *,
    capture: CaptureResult | None,
    image_shape: tuple[int, int],
) -> CameraIntrinsics:
    height, width = image_shape
    if args.camera_fx is not None:
        return CameraIntrinsics(
            fx=float(args.camera_fx),
            fy=float(args.camera_fy),
            cx=float(args.camera_cx),
            cy=float(args.camera_cy),
            width=width,
            height=height,
        )
    if capture is not None:
        return capture.intrinsics
    if (height, width) == (LEGACY_CAMERA_INTRINSICS.height, LEGACY_CAMERA_INTRINSICS.width):
        return LEGACY_CAMERA_INTRINSICS
    scale_x = width / LEGACY_CAMERA_INTRINSICS.width
    scale_y = height / LEGACY_CAMERA_INTRINSICS.height
    return CameraIntrinsics(
        fx=LEGACY_CAMERA_INTRINSICS.fx * scale_x,
        fy=LEGACY_CAMERA_INTRINSICS.fy * scale_y,
        cx=LEGACY_CAMERA_INTRINSICS.cx * scale_x,
        cy=LEGACY_CAMERA_INTRINSICS.cy * scale_y,
        width=width,
        height=height,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    depth_checkpoint = _resolve_file(args.depth_checkpoint, label="depth model checkpoint")
    grasp_checkpoint = _resolve_file(args.grasp_checkpoint, label="AnyGrasp checkpoint")
    arm_requested = bool(args.execute_arm or args.arm_dry_run)
    if arm_requested:
        arm_min_grasp_score, arm_min_grasp_score_source = _resolve_arm_min_grasp_score(args)
    else:
        arm_min_grasp_score = (
            float(args.arm_min_grasp_score) if args.arm_min_grasp_score is not None else None
        )
        arm_min_grasp_score_source = (
            "command_line" if args.arm_min_grasp_score is not None else None
        )

    capture: CaptureResult | None = None
    if args.rgb_image:
        rgb_path = _resolve_file(args.rgb_image, label="RGB image")
        depth_path = _resolve_file(args.depth_image, label="raw depth image")
        run_dir = _new_file_run_dir(args.save_dir)
    else:
        capture = _capture_camera(args)
        run_dir = capture.run_dir
        rgb_path = capture.color_path
        depth_path = capture.depth_path
    color_bgr, raw_depth = _load_rgbd_files(rgb_path, depth_path)
    if args.depth_scale is not None:
        resolved_depth_scale = float(args.depth_scale)
    elif capture is not None:
        resolved_depth_scale = float(capture.raw_units_per_meter)
    else:
        resolved_depth_scale = 1000.0
    intrinsics = _resolve_intrinsics(
        args,
        capture=capture,
        image_shape=color_bgr.shape[:2],
    )
    run_anygrasp = _load_anygrasp_function()

    from asdepth_depth import load_depth_model, predict_depth, save_depth_visualizations

    load_started = time.perf_counter()
    loaded = load_depth_model(
        depth_checkpoint,
        model_id=args.depth_model,
        device=args.device,
        trusted_pickle=args.trusted_depth_checkpoint,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    depth_started = time.perf_counter()
    pred_depth = predict_depth(
        loaded,
        color_bgr,
        raw_depth,
        depth_scale=resolved_depth_scale,
        max_depth_m=args.max_depth,
        input_size=args.input_size,
        resize_method=args.resize_method,
    )
    depth_ms = (time.perf_counter() - depth_started) * 1000.0
    checkpoint_report = loaded.checkpoint
    resolved_device = str(loaded.device)
    resolved_model_id = loaded.model_id
    del loaded
    _clear_model_cache()

    prediction_path = run_dir / "pred_depth.npy"
    np.save(prediction_path, pred_depth)
    visualizations = save_depth_visualizations(
        run_dir,
        raw_depth,
        pred_depth,
        depth_scale=resolved_depth_scale,
        max_depth_m=args.max_depth,
    )

    grasp_started = time.perf_counter()
    grasp_config = _grasp_config(args, grasp_checkpoint, intrinsics)
    grasp = run_anygrasp(
        str(run_dir),
        grasp_config,
        rgb=color_bgr,
        depth=pred_depth,
    )
    grasp_ms = (time.perf_counter() - grasp_started) * 1000.0
    if grasp is None or len(grasp) != 3:
        raise RuntimeError("AnyGrasp did not return a valid grasp pose")
    rotation = np.asarray(grasp[0], dtype=np.float64)
    translation = np.asarray(grasp[1], dtype=np.float64).reshape(-1)
    width = float(grasp[2])
    grasp_score_value = getattr(grasp_config, "grasp_score", None)
    grasp_score = float(grasp_score_value) if grasp_score_value is not None else None
    grasp_candidate_count = int(getattr(grasp_config, "grasp_count", 0))
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"invalid AnyGrasp pose shapes: rotation={rotation.shape}, translation={translation.shape}"
        )
    pose_path = run_dir / "grasp_pose.txt"
    _write_grasp_pose(pose_path, rotation, translation, width)

    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_id": resolved_model_id,
        "depth_checkpoint": str(depth_checkpoint),
        "depth_checkpoint_source_key": checkpoint_report.source_key,
        "depth_checkpoint_tensor_count": checkpoint_report.tensor_count,
        "depth_checkpoint_stripped_prefixes": list(checkpoint_report.stripped_prefixes),
        "grasp_checkpoint": str(grasp_checkpoint),
        "grasp_score": grasp_score,
        "grasp_candidate_count": grasp_candidate_count,
        "grasp_width_m": width,
        "rgb_image": str(rgb_path.resolve()),
        "raw_depth_image": str(depth_path.resolve()),
        "raw_depth_visualization": str(visualizations.raw_depth_path.resolve()),
        "prediction": str(prediction_path.resolve()),
        "prediction_visualization": str(visualizations.prediction_path.resolve()),
        "grasp_pose": str(pose_path.resolve()),
        "input_shape": list(color_bgr.shape[:2]),
        "prediction_shape": list(pred_depth.shape),
        "prediction_dtype": str(pred_depth.dtype),
        "prediction_unit": "meter",
        "device": resolved_device,
        "depth_scale": resolved_depth_scale,
        "camera_backend": capture.backend if capture is not None else "offline",
        "camera_name": capture.camera_name if capture is not None else None,
        "camera_serial_number": capture.serial_number if capture is not None else None,
        "camera_metadata": str(capture.metadata_path) if capture is not None else None,
        "camera_intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "max_depth_m": float(args.max_depth),
        "depth_visualization": {
            "colormap": visualizations.colormap,
            "min_depth_m": visualizations.min_depth_m,
            "max_depth_m": visualizations.max_depth_m,
            "percentile_min": visualizations.percentile_min,
            "percentile_max": visualizations.percentile_max,
            "invalid_color": "black",
            "shared_scale": True,
        },
        "input_size": int(args.input_size),
        "resize_method": args.resize_method,
        "execute_arm_requested": bool(args.execute_arm),
        "arm_dry_run_requested": bool(args.arm_dry_run),
        "arm_min_grasp_score": arm_min_grasp_score,
        "arm_min_grasp_score_source": arm_min_grasp_score_source,
        "arm_score_gate_passed": (
            grasp_score is not None
            and arm_min_grasp_score is not None
            and grasp_score >= arm_min_grasp_score
        ),
        "arm_executed": False,
        "arm_execution_state": "not_requested",
        "arm_may_have_moved": False,
        "timings_ms": {
            "depth_model_load": load_ms,
            "depth_inference": depth_ms,
            "anygrasp": grasp_ms,
        },
    }
    metadata_path = run_dir / "run_metadata.json"
    _write_metadata(metadata_path, metadata)

    if arm_requested:
        if args.execute_arm and grasp_score is None:
            metadata["arm_execution_state"] = "rejected"
            metadata["arm_error"] = "AnyGrasp did not expose a score for safety validation"
            _write_metadata(metadata_path, metadata)
            raise RuntimeError(metadata["arm_error"])
        if args.execute_arm and grasp_score < arm_min_grasp_score:
            metadata["arm_execution_state"] = "rejected"
            metadata["arm_error"] = (
                f"AnyGrasp score {grasp_score:.6f} is below the arm threshold "
                f"{arm_min_grasp_score:.6f}"
            )
            _write_metadata(metadata_path, metadata)
            raise RuntimeError(metadata["arm_error"])

        arm_runner = _load_arm_runner()
        runner_kwargs = _arm_runner_kwargs(args)
        planning_started = time.perf_counter()
        try:
            preview = arm_runner(
                rotation,
                translation,
                width,
                execute=False,
                **runner_kwargs,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            metadata["arm_execution_state"] = "rejected"
            metadata["arm_error"] = str(exc)
            metadata["timings_ms"]["arm_planning"] = (
                time.perf_counter() - planning_started
            ) * 1000.0
            _write_metadata(metadata_path, metadata)
            raise

        metadata["arm_plan"] = preview["plan"]
        metadata["arm_safety_config"] = preview["safety"]
        if preview.get("device") is not None:
            metadata["arm_device"] = preview["device"]
        metadata["timings_ms"]["arm_planning"] = (time.perf_counter() - planning_started) * 1000.0
        if args.arm_dry_run:
            metadata["arm_execution_state"] = "dry_run_complete"
            _write_metadata(metadata_path, metadata)
        else:
            metadata["arm_execution_state"] = "starting"
            _write_metadata(metadata_path, metadata)
            arm_started = time.perf_counter()
            try:
                execution = arm_runner(
                    rotation,
                    translation,
                    width,
                    execute=True,
                    **runner_kwargs,
                )
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                metadata["arm_execution_state"] = "failed"
                metadata["arm_may_have_moved"] = True
                metadata["arm_error"] = str(exc)
                metadata["timings_ms"]["arm_execution"] = (
                    time.perf_counter() - arm_started
                ) * 1000.0
                _write_metadata(metadata_path, metadata)
                raise
            metadata["arm_executed"] = bool(execution["executed"])
            metadata["arm_execution_state"] = "complete"
            metadata["arm_may_have_moved"] = bool(execution["executed"])
            metadata["timings_ms"]["arm_execution"] = (time.perf_counter() - arm_started) * 1000.0
            _write_metadata(metadata_path, metadata)

    return {
        "run_dir": str(run_dir),
        "prediction": str(prediction_path),
        "raw_depth_visualization": str(visualizations.raw_depth_path),
        "prediction_visualization": str(visualizations.prediction_path),
        "grasp_pose": str(pose_path),
        "metadata": str(metadata_path),
        "arm_executed": metadata["arm_executed"],
        "arm_execution_state": metadata["arm_execution_state"],
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
        print(f"asdepth_pipeline: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
