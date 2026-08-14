"""RGB-D camera capture backends for Orbbec and Intel RealSense."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

CameraBackend = Literal["orbbec", "realsense", "auto"]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


@dataclass(frozen=True)
class CapturedFrame:
    color_bgr: np.ndarray
    raw_depth: np.ndarray
    raw_units_per_meter: float
    intrinsics: CameraIntrinsics
    backend: str
    camera_name: str
    serial_number: str


@dataclass(frozen=True)
class CaptureResult:
    run_dir: Path
    color_path: Path
    depth_path: Path
    metadata_path: Path
    raw_units_per_meter: float
    intrinsics: CameraIntrinsics
    backend: str
    camera_name: str
    serial_number: str


def _validate_frame(frame: CapturedFrame) -> CapturedFrame:
    color = np.asarray(frame.color_bgr)
    depth = np.asarray(frame.raw_depth)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"captured color frame must be HxWx3, got {color.shape}")
    if color.dtype != np.uint8:
        raise ValueError(f"captured color frame must be uint8, got {color.dtype}")
    if depth.ndim != 2:
        raise ValueError(f"captured depth frame must be HxW, got {depth.shape}")
    if not np.issubdtype(depth.dtype, np.unsignedinteger):
        raise ValueError(f"captured raw depth frame must be unsigned integer, got {depth.dtype}")
    if color.shape[:2] != depth.shape:
        raise ValueError(
            f"aligned RGB/depth spatial mismatch: rgb={color.shape[:2]}, depth={depth.shape}"
        )
    if frame.raw_units_per_meter <= 0:
        raise ValueError("camera depth scale must be positive")
    if (frame.intrinsics.height, frame.intrinsics.width) != depth.shape:
        raise ValueError(
            "camera intrinsics resolution does not match aligned frames: "
            f"intrinsics={(frame.intrinsics.height, frame.intrinsics.width)}, "
            f"frames={depth.shape}"
        )
    return CapturedFrame(
        color_bgr=np.ascontiguousarray(color),
        raw_depth=np.ascontiguousarray(depth),
        raw_units_per_meter=float(frame.raw_units_per_meter),
        intrinsics=frame.intrinsics,
        backend=frame.backend,
        camera_name=frame.camera_name,
        serial_number=frame.serial_number,
    )


def _write_capture(base_dir: str | Path, frame: CapturedFrame) -> CaptureResult:
    validated = _validate_frame(frame)
    root = Path(base_dir).expanduser().resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"capture_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    color_path = run_dir / "color.png"
    depth_path = run_dir / "depth.png"
    metadata_path = run_dir / "camera_metadata.json"
    if not cv2.imwrite(str(color_path), validated.color_bgr):
        raise OSError(f"failed to save captured color image: {color_path}")
    if not cv2.imwrite(str(depth_path), validated.raw_depth):
        raise OSError(f"failed to save captured depth image: {depth_path}")
    metadata = {
        "schema_version": "1.0.0",
        "backend": validated.backend,
        "camera_name": validated.camera_name,
        "serial_number": validated.serial_number,
        "raw_units_per_meter": validated.raw_units_per_meter,
        "intrinsics": asdict(validated.intrinsics),
        "color_path": str(color_path),
        "depth_path": str(depth_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return CaptureResult(
        run_dir=run_dir,
        color_path=color_path,
        depth_path=depth_path,
        metadata_path=metadata_path,
        raw_units_per_meter=validated.raw_units_per_meter,
        intrinsics=validated.intrinsics,
        backend=validated.backend,
        camera_name=validated.camera_name,
        serial_number=validated.serial_number,
    )


def _realsense_info(device: Any, info: Any, fallback: str) -> str:
    try:
        if device.supports(info):
            return str(device.get_info(info))
    except (AttributeError, RuntimeError):
        pass
    return fallback


def _capture_realsense(
    *,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    timeout_s: float,
) -> CapturedFrame:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "RealSense backend requires pyrealsense2; install requirements-realsense.txt"
        ) from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        align = rs.align(rs.stream.color)
        device = profile.get_device()
        depth_scale_m = float(device.first_depth_sensor().get_depth_scale())
        if depth_scale_m <= 0:
            raise RuntimeError(f"RealSense reported invalid depth scale: {depth_scale_m}")
        deadline = time.monotonic() + timeout_s
        valid_frames = 0
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(1000)
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            valid_frames += 1
            if valid_frames <= warmup_frames:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intrinsic = color_frame.profile.as_video_stream_profile().get_intrinsics()
            return CapturedFrame(
                color_bgr=np.asarray(color, dtype=np.uint8),
                raw_depth=np.asarray(depth),
                raw_units_per_meter=1.0 / depth_scale_m,
                intrinsics=CameraIntrinsics(
                    fx=float(intrinsic.fx),
                    fy=float(intrinsic.fy),
                    cx=float(intrinsic.ppx),
                    cy=float(intrinsic.ppy),
                    width=int(intrinsic.width),
                    height=int(intrinsic.height),
                ),
                backend="realsense",
                camera_name=_realsense_info(device, rs.camera_info.name, "Intel RealSense"),
                serial_number=_realsense_info(device, rs.camera_info.serial_number, "unknown"),
            )
        raise RuntimeError(
            f"RealSense did not produce aligned RGB-D frames within {timeout_s:.1f}s"
        )
    finally:
        if started:
            pipeline.stop()


def _orbbec_text(value: Any, method: str, fallback: str) -> str:
    try:
        return str(getattr(value, method)())
    except Exception:
        return fallback


def _orbbec_color_to_bgr(frame: Any, sdk: Any) -> np.ndarray:
    color_format = frame.get_format()
    width = int(frame.get_width())
    height = int(frame.get_height())
    frame_data = frame.get_data()
    if isinstance(frame_data, np.ndarray):
        data = np.asarray(frame_data, dtype=np.uint8).reshape(-1)
    else:
        data = np.frombuffer(frame_data, dtype=np.uint8)

    def is_format(name: str) -> bool:
        return color_format == getattr(sdk.OBFormat, name, None)

    if is_format("RGB"):
        rgb = data.reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if is_format("BGR"):
        return data.reshape(height, width, 3).copy()
    if is_format("YUYV") or is_format("YUY2"):
        yuyv = data.reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if is_format("UYVY"):
        uyvy = data.reshape(height, width, 2)
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if is_format("MJPG"):
        image = cv2.imdecode(data.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode Orbbec MJPEG color frame")
        return image
    if is_format("I420"):
        i420 = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    if is_format("NV12"):
        nv12 = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if is_format("NV21"):
        nv21 = data.reshape(height * 3 // 2, width)
        return cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    raise RuntimeError(f"unsupported Orbbec color format: {color_format}")


def _capture_orbbec_with_align_mode(
    sdk: Any,
    *,
    align_mode: Any,
    warmup_frames: int,
    timeout_s: float,
) -> CapturedFrame:
    pipeline = sdk.Pipeline()
    config = sdk.Config()
    color_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
    if color_profiles is None or depth_profiles is None:
        raise RuntimeError("Orbbec camera does not expose both color and depth sensors")
    color_profile = color_profiles.get_default_video_stream_profile()
    depth_profile = depth_profiles.get_default_video_stream_profile()
    if color_profile is None or depth_profile is None:
        raise RuntimeError("Orbbec camera has no usable default RGB-D stream profiles")
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(align_mode)
    with suppress(Exception):
        pipeline.enable_frame_sync()

    started = False
    try:
        pipeline.start(config)
        started = True
        device_info = pipeline.get_device().get_device_info()
        deadline = time.monotonic() + timeout_s
        valid_frames = 0
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue
            color = _orbbec_color_to_bgr(color_frame, sdk)
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                int(depth_frame.get_height()), int(depth_frame.get_width())
            )
            if color.shape[:2] != depth.shape:
                continue
            valid_frames += 1
            if valid_frames <= warmup_frames:
                continue
            scale_mm_per_unit = float(depth_frame.get_depth_scale())
            if scale_mm_per_unit <= 0:
                raise RuntimeError(f"Orbbec reported invalid depth scale: {scale_mm_per_unit}")
            intrinsic = color_profile.get_intrinsic()
            return CapturedFrame(
                color_bgr=np.asarray(color, dtype=np.uint8),
                raw_depth=depth.copy(),
                raw_units_per_meter=1000.0 / scale_mm_per_unit,
                intrinsics=CameraIntrinsics(
                    fx=float(intrinsic.fx),
                    fy=float(intrinsic.fy),
                    cx=float(intrinsic.cx),
                    cy=float(intrinsic.cy),
                    width=int(intrinsic.width),
                    height=int(intrinsic.height),
                ),
                backend="orbbec",
                camera_name=_orbbec_text(device_info, "get_name", "Orbbec DaBai"),
                serial_number=_orbbec_text(device_info, "get_serial_number", "unknown"),
            )
        raise RuntimeError(
            "Orbbec did not produce aligned RGB-D frames with matching resolution "
            f"within {timeout_s:.1f}s"
        )
    finally:
        if started:
            pipeline.stop()


def _capture_orbbec(*, warmup_frames: int, timeout_s: float) -> CapturedFrame:
    try:
        import pyorbbecsdk as sdk
    except ImportError as exc:
        raise ImportError(
            "Orbbec backend requires the legacy PyOrbbecSDK v1 module "
            "'pyorbbecsdk'; installing pyorbbecsdk2 alone is insufficient for "
            "a DaBai camera using legacy/OpenNI firmware"
        ) from exc

    failures: list[str] = []
    for mode_name in ("HW_MODE", "SW_MODE"):
        try:
            mode = getattr(sdk.OBAlignMode, mode_name)
            return _capture_orbbec_with_align_mode(
                sdk,
                align_mode=mode,
                warmup_frames=warmup_frames,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            failures.append(f"{mode_name}: {exc}")
    raise RuntimeError("Orbbec capture failed; " + "; ".join(failures))


def capture_one_frame(
    base_dir: str | Path,
    *,
    backend: CameraBackend = "orbbec",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 30,
    timeout_s: float = 20.0,
) -> CaptureResult:
    """Capture one aligned RGB-D frame and persist images plus calibration metadata."""

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("camera width, height and fps must be positive")
    if warmup_frames < 0 or timeout_s <= 0:
        raise ValueError("camera warmup frames must be non-negative and timeout positive")
    if backend not in {"orbbec", "realsense", "auto"}:
        raise ValueError(f"unsupported camera backend: {backend}")

    if backend == "orbbec":
        frame = _capture_orbbec(warmup_frames=warmup_frames, timeout_s=timeout_s)
        return _write_capture(base_dir, frame)
    if backend == "realsense":
        frame = _capture_realsense(
            width=width,
            height=height,
            fps=fps,
            warmup_frames=warmup_frames,
            timeout_s=timeout_s,
        )
        return _write_capture(base_dir, frame)

    errors: list[str] = []
    for candidate in ("orbbec", "realsense"):
        try:
            return capture_one_frame(
                base_dir,
                backend=candidate,
                width=width,
                height=height,
                fps=fps,
                warmup_frames=warmup_frames,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("no supported RGB-D camera could be captured; " + "; ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="采集一帧对齐后的 RGB-D 图像并保存相机标定元数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-backend",
        choices=["orbbec", "realsense", "auto"],
        default="orbbec",
    )
    parser.add_argument("--save-dir", default="debug/camera")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument("--camera-timeout", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = capture_one_frame(
            args.save_dir,
            backend=args.camera_backend,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
            timeout_s=args.camera_timeout,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"camera_capture: error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "backend": result.backend,
                "camera_name": result.camera_name,
                "serial_number": result.serial_number,
                "run_dir": str(result.run_dir),
                "color": str(result.color_path),
                "depth": str(result.depth_path),
                "metadata": str(result.metadata_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
