#!/usr/bin/env python3
"""Piper motion planning and guarded hardware execution."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from piper_control import enable_with_timeout as _enable_piper_with_timeout
from piper_control import request_emergency_stop

DEFAULT_DEVICE_CONFIG_PATH = "config/piper_device.json"
DEFAULT_ARM_MIN_GRASP_SCORE = 0.2
GRIPPER_WIDTH_EPSILON_M = 1e-6
DEFAULT_T_CAM_TO_BASE = np.array(
    [
        [-0.03188415, -0.64642446, 0.7623115, -0.02826907],
        [-0.99742629, -0.02842686, -0.06582336, -0.2205848],
        [0.06421995, -0.76244825, -0.64385438, 0.57842954],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ArmPose:
    x_m: float
    y_m: float
    z_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def command(self) -> tuple[int, int, int, int, int, int]:
        return (
            _meters_to_micrometers(self.x_m),
            _meters_to_micrometers(self.y_m),
            _meters_to_micrometers(self.z_m),
            _degrees_to_millidegrees(self.rx_deg),
            _degrees_to_millidegrees(self.ry_deg),
            _degrees_to_millidegrees(self.rz_deg),
        )


@dataclass(frozen=True)
class ArmSafetyConfig:
    can_name: str = "can1"
    motion_speed_percent: int = 10
    gripper_max_width_m: float = 0.095
    tool_offset_m: float = 0.07
    pregrasp_clearance_m: float = 0.15
    lift_distance_m: float = 0.10
    max_reach_m: float = 0.62
    min_z_m: float = 0.05
    max_z_m: float = 0.60
    max_abs_x_m: float = 0.50
    max_abs_y_m: float = 0.50
    enable_timeout_s: float = 5.0
    move_timeout_s: float = 12.0
    gripper_timeout_s: float = 8.0
    position_tolerance_m: float = 0.005
    angle_tolerance_deg: float = 5.0
    gripper_tolerance_m: float = 0.003

    def __post_init__(self) -> None:
        if not self.can_name:
            raise ValueError("Piper CAN interface name must not be empty")
        if not 1 <= self.motion_speed_percent <= 100:
            raise ValueError("Piper motion speed must be between 1 and 100 percent")
        if not 0 < self.gripper_max_width_m <= 0.10:
            raise ValueError("Piper gripper maximum width must be in (0, 0.10] meter")
        if self.tool_offset_m <= 0 or self.pregrasp_clearance_m <= 0:
            raise ValueError("Piper tool offset and pregrasp clearance must be positive")
        if self.lift_distance_m <= 0 or self.max_reach_m <= 0:
            raise ValueError("Piper lift distance and maximum reach must be positive")
        if not 0 <= self.min_z_m < self.max_z_m:
            raise ValueError("Piper Z safety range is invalid")
        if self.max_abs_x_m <= 0 or self.max_abs_y_m <= 0:
            raise ValueError("Piper X/Y safety limits must be positive")
        if min(self.enable_timeout_s, self.move_timeout_s, self.gripper_timeout_s) <= 0:
            raise ValueError("Piper operation timeouts must be positive")
        if min(
            self.position_tolerance_m,
            self.angle_tolerance_deg,
            self.gripper_tolerance_m,
        ) <= 0:
            raise ValueError("Piper position, angle and gripper tolerances must be positive")


@dataclass(frozen=True)
class ArmMotionPlan:
    detected_object_pose: ArmPose
    ready_pose: ArmPose
    pregrasp_pose: ArmPose
    grasp_pose: ArmPose
    lift_pose: ArmPose
    gripper_open_width_m: float
    gripper_grasp_width_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_device_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Piper device config does not exist: {resolved}")
    return resolved


def _load_device_config_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_device_config_path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Piper device config JSON: {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Piper device config must contain a JSON object: {config_path}")
    if data.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported Piper device config schema: {data.get('schema_version')}")
    return config_path, data


def resolve_arm_min_grasp_score(
    path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
    *,
    override: float | None = None,
) -> tuple[float, str]:
    """Resolve the arm execution score gate and report where it came from."""

    if override is not None:
        score = float(override)
        source = "command_line"
    else:
        config_path, data = _load_device_config_json(path)
        motion = data.get("motion")
        if not isinstance(motion, dict):
            raise ValueError("Piper device config is missing the motion object")
        if "arm_min_grasp_score" in motion:
            score = float(motion["arm_min_grasp_score"])
            source = f"{config_path}#motion.arm_min_grasp_score"
        else:
            score = DEFAULT_ARM_MIN_GRASP_SCORE
            source = "built_in_default"
    if not 0.0 <= score <= 1.0:
        raise ValueError("Piper arm minimum grasp score must be between 0 and 1")
    return score, source


def _validate_camera_to_base(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Piper camera_to_base must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("Piper camera_to_base has an invalid homogeneous last row")
    _validate_rotation(transform[:3, :3])
    return transform


def load_device_config(
    path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
    *,
    arm_side: str | None = None,
    can_name: str | None = None,
    motion_speed_percent: int | None = None,
    gripper_max_width_m: float | None = None,
    tool_offset_m: float | None = None,
    pregrasp_clearance_m: float | None = None,
    lift_distance_m: float | None = None,
    max_reach_m: float | None = None,
    min_z_m: float | None = None,
    max_z_m: float | None = None,
    max_abs_x_m: float | None = None,
    max_abs_y_m: float | None = None,
    enable_timeout_s: float | None = None,
    move_timeout_s: float | None = None,
    gripper_timeout_s: float | None = None,
    position_tolerance_m: float | None = None,
    angle_tolerance_deg: float | None = None,
    gripper_tolerance_m: float | None = None,
) -> tuple[np.ndarray, ArmSafetyConfig, dict[str, Any]]:
    """Load the selected arm, calibration and safety settings from JSON."""

    config_path, data = _load_device_config_json(path)

    selected_side = arm_side or data.get("active_arm")
    arms = data.get("arms")
    if not isinstance(arms, dict) or selected_side not in arms:
        raise ValueError(f"Piper arm '{selected_side}' is not defined in {config_path}")
    arm = arms[selected_side]
    if not isinstance(arm, dict):
        raise ValueError(f"Piper arm '{selected_side}' config must be an object")
    if not arm.get("enabled", False):
        reason = arm.get("disabled_reason", "arm is disabled")
        raise RuntimeError(f"Piper arm '{selected_side}' is disabled: {reason}")
    camera_to_base_value = arm.get("camera_to_base")
    if camera_to_base_value is None:
        raise ValueError(f"Piper arm '{selected_side}' has no camera_to_base calibration")
    camera_to_base = _validate_camera_to_base(camera_to_base_value)

    motion = data.get("motion")
    if not isinstance(motion, dict):
        raise ValueError("Piper device config is missing the motion object")

    def setting(name: str, override: Any) -> Any:
        if override is not None:
            return override
        if name not in motion:
            raise ValueError(f"Piper device config motion.{name} is missing")
        return motion[name]

    resolved_can_name = arm.get("can_interface") if can_name is None else can_name
    safety = ArmSafetyConfig(
        can_name=str(resolved_can_name or "").strip(),
        motion_speed_percent=int(setting("motion_speed_percent", motion_speed_percent)),
        gripper_max_width_m=float(setting("gripper_max_width_m", gripper_max_width_m)),
        tool_offset_m=float(setting("tool_offset_m", tool_offset_m)),
        pregrasp_clearance_m=float(setting("pregrasp_clearance_m", pregrasp_clearance_m)),
        lift_distance_m=float(setting("lift_distance_m", lift_distance_m)),
        max_reach_m=float(setting("max_reach_m", max_reach_m)),
        min_z_m=float(setting("min_z_m", min_z_m)),
        max_z_m=float(setting("max_z_m", max_z_m)),
        max_abs_x_m=float(setting("max_abs_x_m", max_abs_x_m)),
        max_abs_y_m=float(setting("max_abs_y_m", max_abs_y_m)),
        enable_timeout_s=float(setting("enable_timeout_s", enable_timeout_s)),
        move_timeout_s=float(setting("move_timeout_s", move_timeout_s)),
        gripper_timeout_s=float(setting("gripper_timeout_s", gripper_timeout_s)),
        position_tolerance_m=float(setting("position_tolerance_m", position_tolerance_m)),
        angle_tolerance_deg=float(setting("angle_tolerance_deg", angle_tolerance_deg)),
        gripper_tolerance_m=float(setting("gripper_tolerance_m", gripper_tolerance_m)),
    )
    metadata = {
        "config_path": str(config_path),
        "device_name": str(data.get("device_name", "unknown")),
        "arm_side": str(selected_side),
        "can_interface": safety.can_name,
        "camera_to_base": camera_to_base.tolist(),
    }
    return camera_to_base, safety, metadata


def _meters_to_micrometers(value: float) -> int:
    return int(round(value * 1e6))


def _degrees_to_millidegrees(value: float) -> int:
    return int(round(value * 1e3))


def _euler_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch and yaw in radians for ``R = Rz * Ry * Rx``."""

    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def _normalize_gripper_angle(angle_deg: float) -> float:
    """Map an equivalent parallel-gripper angle to the conservative [-90, 90] range."""

    angle = (angle_deg + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _validate_rotation(rotation: np.ndarray) -> None:
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"invalid AnyGrasp rotation matrix: shape={rotation.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("AnyGrasp rotation matrix is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=1e-3):
        raise ValueError(f"AnyGrasp rotation determinant must be 1, got {determinant:.6f}")


def _pose_from_transform(transform: np.ndarray, angles_deg: tuple[float, float, float]) -> ArmPose:
    translation = transform[:3, 3]
    return ArmPose(
        x_m=float(translation[0]),
        y_m=float(translation[1]),
        z_m=float(translation[2]),
        rx_deg=float(angles_deg[0]),
        ry_deg=float(angles_deg[1]),
        rz_deg=float(angles_deg[2]),
    )


def _normalize_gripper_width(width_m: float, maximum_m: float) -> float:
    if width_m > maximum_m + GRIPPER_WIDTH_EPSILON_M:
        raise ValueError(
            "AnyGrasp width exceeds the configured Piper gripper limit: "
            f"{width_m:.4f}m > {maximum_m:.4f}m"
        )
    return min(width_m, maximum_m)


def build_motion_plan(
    rotation_camera: np.ndarray,
    translation_camera: np.ndarray,
    grasp_width_m: float,
    *,
    safety: ArmSafetyConfig | None = None,
    camera_to_base: np.ndarray | None = None,
) -> ArmMotionPlan:
    """Create and validate a hardware-independent Piper motion plan."""

    config = safety or ArmSafetyConfig()
    base_from_camera = _validate_camera_to_base(
        DEFAULT_T_CAM_TO_BASE if camera_to_base is None else camera_to_base
    )
    rotation = np.asarray(rotation_camera, dtype=np.float64)
    translation = np.asarray(translation_camera, dtype=np.float64).reshape(-1)
    _validate_rotation(rotation)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(f"invalid AnyGrasp translation: shape={translation.shape}")
    if not math.isfinite(grasp_width_m) or grasp_width_m <= 0:
        raise ValueError(f"invalid AnyGrasp gripper width: {grasp_width_m}")
    command_gripper_width_m = _normalize_gripper_width(
        float(grasp_width_m), config.gripper_max_width_m
    )

    camera_to_object = np.eye(4, dtype=np.float64)
    camera_to_object[:3, :3] = rotation
    camera_to_object[:3, 3] = translation
    base_to_detected = base_from_camera @ camera_to_object

    original_angles = tuple(np.degrees(_euler_zyx(base_to_detected[:3, :3])))
    # Preserve the deployed tool-mount convention, but normalize after applying the
    # fixed 85-degree mount offset so an angle such as 146 degrees is never emitted.
    command_angles = (
        _normalize_gripper_angle(float(original_angles[0])),
        _normalize_gripper_angle(float(original_angles[1]) + 85.0),
        _normalize_gripper_angle(float(original_angles[2])),
    )

    tool_axis_mapping = np.eye(4, dtype=np.float64)
    tool_axis_mapping[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    camera_to_tool = camera_to_object @ tool_axis_mapping

    tool_offset = np.eye(4, dtype=np.float64)
    tool_offset[:3, 3] = [0.0, 0.0, -config.tool_offset_m]
    base_to_grasp = base_from_camera @ camera_to_tool @ tool_offset

    pregrasp_offset = np.eye(4, dtype=np.float64)
    pregrasp_offset[:3, 3] = [0.0, 0.0, -config.pregrasp_clearance_m]
    base_to_pregrasp = base_to_grasp @ pregrasp_offset

    base_to_lift = base_to_grasp.copy()
    base_to_lift[2, 3] += config.lift_distance_m

    plan = ArmMotionPlan(
        detected_object_pose=_pose_from_transform(base_to_detected, original_angles),
        ready_pose=ArmPose(0.015, 0.0, 0.275, 0.0, 85.0, 0.0),
        pregrasp_pose=_pose_from_transform(base_to_pregrasp, command_angles),
        grasp_pose=_pose_from_transform(base_to_grasp, command_angles),
        lift_pose=_pose_from_transform(base_to_lift, command_angles),
        gripper_open_width_m=config.gripper_max_width_m,
        gripper_grasp_width_m=command_gripper_width_m,
    )
    validate_motion_plan(plan, config)
    return plan


def validate_motion_plan(plan: ArmMotionPlan, safety: ArmSafetyConfig) -> None:
    _normalize_gripper_width(plan.gripper_grasp_width_m, safety.gripper_max_width_m)

    for name, pose in (
        ("ready", plan.ready_pose),
        ("pregrasp", plan.pregrasp_pose),
        ("grasp", plan.grasp_pose),
        ("lift", plan.lift_pose),
    ):
        values = asdict(pose).values()
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Piper {name} pose contains a non-finite value")
        radius = math.sqrt(pose.x_m**2 + pose.y_m**2 + pose.z_m**2)
        if radius > safety.max_reach_m:
            raise ValueError(
                f"Piper {name} pose exceeds maximum reach: {radius:.4f}m > "
                f"{safety.max_reach_m:.4f}m"
            )
        if abs(pose.x_m) > safety.max_abs_x_m or abs(pose.y_m) > safety.max_abs_y_m:
            raise ValueError(f"Piper {name} pose exceeds configured X/Y workspace limits")
        if not safety.min_z_m <= pose.z_m <= safety.max_z_m:
            raise ValueError(f"Piper {name} Z is outside the configured range: {pose.z_m:.4f}m")
        if max(abs(pose.rx_deg), abs(pose.ry_deg), abs(pose.rz_deg)) > 90.0:
            raise ValueError(f"Piper {name} orientation is outside [-90, 90] degrees")


def _angular_error_mdeg(actual: int, target: int) -> int:
    return abs((actual - target + 180_000) % 360_000 - 180_000)


def move_with_check(
    piper: Any,
    pose: ArmPose,
    *,
    timeout_s: float,
    position_tolerance_m: float,
    angle_tolerance_deg: float,
) -> None:
    """Send an end-pose target and fail unless all six feedback axes converge."""

    target = pose.command()
    position_tolerance = _meters_to_micrometers(position_tolerance_m)
    angle_tolerance = _degrees_to_millidegrees(angle_tolerance_deg)
    deadline = time.monotonic() + timeout_s
    last_actual: tuple[int, int, int, int, int, int] | None = None

    while time.monotonic() < deadline:
        piper.EndPoseCtrl(*target)
        feedback = piper.GetArmEndPoseMsgs().end_pose
        last_actual = (
            int(feedback.X_axis),
            int(feedback.Y_axis),
            int(feedback.Z_axis),
            int(feedback.RX_axis),
            int(feedback.RY_axis),
            int(feedback.RZ_axis),
        )
        position_ok = all(
            abs(actual - expected) <= position_tolerance
            for actual, expected in zip(last_actual[:3], target[:3], strict=True)
        )
        angle_ok = all(
            _angular_error_mdeg(actual, expected) <= angle_tolerance
            for actual, expected in zip(last_actual[3:], target[3:], strict=True)
        )
        if position_ok and angle_ok:
            return
        time.sleep(0.02)

    raise TimeoutError(
        f"Piper failed to reach pose within {timeout_s:.1f}s; "
        f"target={target}, last_feedback={last_actual}"
    )


def _enable_with_timeout(piper: Any, timeout_s: float) -> None:
    _enable_piper_with_timeout(piper, timeout_s)


def _set_gripper_with_check(
    piper: Any,
    width_m: float,
    *,
    timeout_s: float,
    tolerance_m: float,
) -> None:
    target = _meters_to_micrometers(width_m)
    tolerance = _meters_to_micrometers(tolerance_m)
    deadline = time.monotonic() + timeout_s
    last_actual: int | None = None

    while time.monotonic() < deadline:
        piper.GripperCtrl(target, 1000, 0x01, 0)
        last_actual = int(piper.GetArmGripperMsgs().gripper_state.grippers_angle)
        if abs(last_actual - target) <= tolerance:
            return
        time.sleep(0.02)

    raise TimeoutError(
        f"Piper gripper failed to reach {width_m:.4f}m within {timeout_s:.1f}s; "
        f"last_feedback={last_actual}"
    )


def _print_plan(plan: ArmMotionPlan) -> None:
    print("Piper detected object pose:", asdict(plan.detected_object_pose))
    print("Piper ready pose:", asdict(plan.ready_pose))
    print("Piper pregrasp pose:", asdict(plan.pregrasp_pose))
    print("Piper grasp pose:", asdict(plan.grasp_pose))
    print("Piper lift pose:", asdict(plan.lift_pose))
    print(
        "Piper gripper widths (open/grasp m):",
        plan.gripper_open_width_m,
        plan.gripper_grasp_width_m,
    )


def _execute_motion_plan(plan: ArmMotionPlan, safety: ArmSafetyConfig) -> None:
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as exc:
        raise ImportError("Piper execution requires piper_sdk") from exc

    piper: Any | None = None
    hardware_action_started = False
    try:
        piper = C_PiperInterface_V2(safety.can_name)
        piper.ConnectPort()
        _enable_with_timeout(piper, safety.enable_timeout_s)
        hardware_action_started = True

        piper.GripperCtrl(0, 1000, 0x02, 0)
        piper.GripperCtrl(0, 1000, 0x03, 0)
        _set_gripper_with_check(
            piper,
            plan.gripper_open_width_m,
            timeout_s=safety.gripper_timeout_s,
            tolerance_m=safety.gripper_tolerance_m,
        )

        piper.MotionCtrl_2(0x01, 0x00, safety.motion_speed_percent, 0x00)
        for name, pose in (
            ("ready", plan.ready_pose),
            ("pregrasp", plan.pregrasp_pose),
            ("grasp", plan.grasp_pose),
        ):
            move_with_check(
                piper,
                pose,
                timeout_s=safety.move_timeout_s,
                position_tolerance_m=safety.position_tolerance_m,
                angle_tolerance_deg=safety.angle_tolerance_deg,
            )
            print(f"Piper reached {name} pose")

        _set_gripper_with_check(
            piper,
            plan.gripper_grasp_width_m,
            timeout_s=safety.gripper_timeout_s,
            tolerance_m=safety.gripper_tolerance_m,
        )
        print("Piper gripper reached grasp width")

        move_with_check(
            piper,
            plan.lift_pose,
            timeout_s=safety.move_timeout_s,
            position_tolerance_m=safety.position_tolerance_m,
            angle_tolerance_deg=safety.angle_tolerance_deg,
        )
        print("Piper reached lift pose")
    except Exception as exc:
        emergency_requested = False
        if piper is not None and hardware_action_started:
            with suppress(Exception):
                request_emergency_stop(piper)
                emergency_requested = True
        suffix = "emergency stop requested" if emergency_requested else "no motion command sent"
        raise RuntimeError(f"Piper execution failed ({suffix}): {exc}") from exc
    finally:
        if piper is not None:
            with suppress(Exception):
                piper.DisconnectPort()


def run_pipeline(
    rotation_camera: np.ndarray,
    translation_camera: np.ndarray,
    grasp_width_m: float,
    *,
    execute: bool = False,
    device_config_path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
    arm_side: str | None = None,
    can_name: str | None = None,
    motion_speed_percent: int | None = None,
    gripper_max_width_m: float | None = None,
    tool_offset_m: float | None = None,
    pregrasp_clearance_m: float | None = None,
    lift_distance_m: float | None = None,
    max_reach_m: float | None = None,
    min_z_m: float | None = None,
    max_z_m: float | None = None,
    max_abs_x_m: float | None = None,
    max_abs_y_m: float | None = None,
    enable_timeout_s: float | None = None,
    move_timeout_s: float | None = None,
    gripper_timeout_s: float | None = None,
    position_tolerance_m: float | None = None,
    angle_tolerance_deg: float | None = None,
    gripper_tolerance_m: float | None = None,
) -> dict[str, Any]:
    """Plan a Piper grasp and execute it only when ``execute`` is explicitly true."""

    camera_to_base, safety, device = load_device_config(
        device_config_path,
        arm_side=arm_side,
        can_name=can_name,
        motion_speed_percent=motion_speed_percent,
        gripper_max_width_m=gripper_max_width_m,
        tool_offset_m=tool_offset_m,
        pregrasp_clearance_m=pregrasp_clearance_m,
        lift_distance_m=lift_distance_m,
        max_reach_m=max_reach_m,
        min_z_m=min_z_m,
        max_z_m=max_z_m,
        max_abs_x_m=max_abs_x_m,
        max_abs_y_m=max_abs_y_m,
        enable_timeout_s=enable_timeout_s,
        move_timeout_s=move_timeout_s,
        gripper_timeout_s=gripper_timeout_s,
        position_tolerance_m=position_tolerance_m,
        angle_tolerance_deg=angle_tolerance_deg,
        gripper_tolerance_m=gripper_tolerance_m,
    )
    plan = build_motion_plan(
        rotation_camera,
        translation_camera,
        grasp_width_m,
        safety=safety,
        camera_to_base=camera_to_base,
    )
    _print_plan(plan)
    if execute:
        _execute_motion_plan(plan, safety)
    else:
        print("Piper dry-run complete; no CAN interface was opened")
    return {
        "executed": bool(execute),
        "plan": plan.as_dict(),
        "safety": asdict(safety),
        "device": device,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture an RGB-D frame, plan a Piper grasp, and execute only with confirmation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        default="ckpts/checkpoint_detection.tar",
        help="AnyGrasp 2026 detection checkpoint path",
    )
    parser.add_argument("--max_gripper_width", type=float, default=0.095)
    parser.add_argument("--gripper_height", type=float, default=0.03)
    parser.add_argument("--top_down_grasp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_dir", default="debug/funny")
    parser.add_argument(
        "--camera_backend",
        choices=["orbbec", "realsense", "auto"],
        default="orbbec",
    )
    parser.add_argument("--execute-arm", action="store_true")
    parser.add_argument("--confirm-arm-motion", action="store_true")
    parser.add_argument("--arm-config", default=DEFAULT_DEVICE_CONFIG_PATH)
    parser.add_argument("--arm-side", choices=["left", "right"])
    parser.add_argument("--arm-can-interface")
    parser.add_argument("--arm-speed-percent", type=int)
    parser.add_argument("--arm-gripper-max-width", type=float)
    parser.add_argument(
        "--arm-min-grasp-score",
        type=float,
        default=None,
        help="minimum AnyGrasp score; defaults to motion.arm_min_grasp_score in --arm-config",
    )
    parser.add_argument("--arm-tool-offset", type=float)
    parser.add_argument("--arm-pregrasp-clearance", type=float)
    parser.add_argument("--arm-lift-distance", type=float)
    parser.add_argument("--arm-max-reach", type=float)
    parser.add_argument("--arm-min-z", type=float)
    parser.add_argument("--arm-max-z", type=float)
    parser.add_argument("--arm-max-abs-x", type=float)
    parser.add_argument("--arm-max-abs-y", type=float)
    parser.add_argument("--arm-enable-timeout", type=float)
    parser.add_argument("--arm-move-timeout", type=float)
    parser.add_argument("--arm-gripper-timeout", type=float)
    parser.add_argument("--arm-position-tolerance", type=float)
    parser.add_argument("--arm-angle-tolerance", type=float)
    parser.add_argument("--arm-gripper-tolerance", type=float)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.execute_arm and not args.confirm_arm_motion:
        raise SystemExit("--execute-arm requires --confirm-arm-motion")

    from camera_capture import capture_one_frame
    from get_pose import run_anygrasp

    args.max_gripper_width = max(0.0, min(0.095, args.max_gripper_width))
    capture = capture_one_frame(args.save_dir, backend=args.camera_backend)
    args.depth_scale = capture.raw_units_per_meter
    args.camera_intrinsics = {
        "fx": capture.intrinsics.fx,
        "fy": capture.intrinsics.fy,
        "cx": capture.intrinsics.cx,
        "cy": capture.intrinsics.cy,
    }
    run_dir = str(capture.run_dir)
    grasp = run_anygrasp(run_dir, args, data_dir=run_dir)
    if grasp is None:
        raise RuntimeError("AnyGrasp did not return a valid grasp")
    min_grasp_score, _ = resolve_arm_min_grasp_score(
        args.arm_config,
        override=args.arm_min_grasp_score,
    )
    grasp_score_value = getattr(args, "grasp_score", None)
    if args.execute_arm and grasp_score_value is None:
        raise RuntimeError("AnyGrasp did not expose a score for safety validation")
    if args.execute_arm and float(grasp_score_value) < min_grasp_score:
        raise RuntimeError(
            f"AnyGrasp score {float(grasp_score_value):.6f} is below the arm threshold "
            f"{min_grasp_score:.6f}"
        )
    run_pipeline(
        grasp[0],
        grasp[1],
        grasp[2],
        execute=args.execute_arm,
        device_config_path=args.arm_config,
        arm_side=args.arm_side,
        can_name=args.arm_can_interface,
        motion_speed_percent=args.arm_speed_percent,
        gripper_max_width_m=args.arm_gripper_max_width,
        tool_offset_m=args.arm_tool_offset,
        pregrasp_clearance_m=args.arm_pregrasp_clearance,
        lift_distance_m=args.arm_lift_distance,
        max_reach_m=args.arm_max_reach,
        min_z_m=args.arm_min_z,
        max_z_m=args.arm_max_z,
        max_abs_x_m=args.arm_max_abs_x,
        max_abs_y_m=args.arm_max_abs_y,
        enable_timeout_s=args.arm_enable_timeout,
        move_timeout_s=args.arm_move_timeout,
        gripper_timeout_s=args.arm_gripper_timeout,
        position_tolerance_m=args.arm_position_tolerance,
        angle_tolerance_deg=args.arm_angle_tolerance,
        gripper_tolerance_m=args.arm_gripper_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
