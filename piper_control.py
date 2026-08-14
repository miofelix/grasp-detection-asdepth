#!/usr/bin/env python3
"""Hardware-focused Piper status and guarded manual-control helpers.

This module deliberately has no camera, vision-model, or NumPy dependency so it can be
used on a robot host for diagnostics even when the perception environment is unavailable.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_DEVICE_CONFIG_PATH = "config/piper_device.json"

CTRL_MODE_LABELS = {
    0x00: "standby",
    0x01: "can_control",
    0x02: "teaching",
    0x03: "ethernet_control",
    0x04: "wifi_control",
    0x05: "remote_control",
    0x06: "linkage_teaching",
    0x07: "offline_trajectory",
}
ARM_STATUS_LABELS = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_solution",
    0x03: "singularity",
    0x04: "target_exceeds_limit",
    0x05: "joint_communication_error",
    0x06: "joint_brake_not_released",
    0x07: "collision",
    0x08: "teaching_overspeed",
    0x09: "joint_status_error",
    0x0A: "other_error",
    0x0B: "teaching_record",
    0x0C: "teaching_execution",
    0x0D: "teaching_pause",
    0x0E: "main_controller_over_temperature",
    0x0F: "release_resistor_over_temperature",
}
MOVE_MODE_LABELS = {
    0x00: "move_p",
    0x01: "move_j",
    0x02: "move_l",
    0x03: "move_c",
    0x04: "move_m",
    0x05: "move_cpv",
}
TEACH_STATUS_LABELS = {
    0x00: "disabled",
    0x01: "recording",
    0x02: "recording_stopped",
    0x03: "trajectory_execution",
    0x04: "execution_paused",
    0x05: "execution_resumed",
    0x06: "execution_terminated",
    0x07: "moving_to_start",
}
MOTION_STATUS_LABELS = {0x00: "reached", 0x01: "not_reached"}

GRIPPER_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)
MOTOR_FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "stall_status",
)


@dataclass(frozen=True)
class ManualPose:
    x_m: float
    y_m: float
    z_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def command(self) -> tuple[int, int, int, int, int, int]:
        return (
            meters_to_micrometers(self.x_m),
            meters_to_micrometers(self.y_m),
            meters_to_micrometers(self.z_m),
            degrees_to_millidegrees(self.rx_deg),
            degrees_to_millidegrees(self.ry_deg),
            degrees_to_millidegrees(self.rz_deg),
        )


@dataclass(frozen=True)
class ManualControlConfig:
    config_path: str
    device_name: str
    arm_side: str
    can_name: str
    motion_speed_percent: int
    gripper_max_width_m: float
    max_reach_m: float
    min_z_m: float
    max_z_m: float
    max_abs_x_m: float
    max_abs_y_m: float
    enable_timeout_s: float
    move_timeout_s: float
    gripper_timeout_s: float
    position_tolerance_m: float
    angle_tolerance_deg: float
    gripper_tolerance_m: float
    feedback_timeout_s: float
    watch_hz: float
    max_cartesian_step_m: float
    max_angular_step_deg: float
    max_joint_step_deg: float
    joint_limit_margin_deg: float
    self_test_z_step_m: float
    self_test_joint_6_step_deg: float
    self_test_gripper_step_m: float

    def __post_init__(self) -> None:
        if not self.can_name:
            raise ValueError("Piper CAN interface name must not be empty")
        if not 1 <= self.motion_speed_percent <= 100:
            raise ValueError("Piper motion speed must be between 1 and 100 percent")
        if not 0 < self.gripper_max_width_m <= 0.10:
            raise ValueError("Piper gripper maximum width must be in (0, 0.10] meter")
        if self.max_reach_m <= 0 or self.max_abs_x_m <= 0 or self.max_abs_y_m <= 0:
            raise ValueError("Piper workspace limits must be positive")
        if not 0 <= self.min_z_m < self.max_z_m:
            raise ValueError("Piper Z safety range is invalid")
        if min(
            self.enable_timeout_s,
            self.move_timeout_s,
            self.gripper_timeout_s,
            self.feedback_timeout_s,
        ) <= 0:
            raise ValueError("Piper operation and feedback timeouts must be positive")
        if min(
            self.position_tolerance_m,
            self.angle_tolerance_deg,
            self.gripper_tolerance_m,
            self.watch_hz,
            self.max_cartesian_step_m,
            self.max_angular_step_deg,
            self.max_joint_step_deg,
            self.joint_limit_margin_deg,
            self.self_test_z_step_m,
            self.self_test_joint_6_step_deg,
            self.self_test_gripper_step_m,
        ) <= 0:
            raise ValueError("Piper manual-control limits must be positive")
        if self.max_cartesian_step_m > 0.10:
            raise ValueError("Piper manual Cartesian step limit must not exceed 0.10 meter")
        if self.max_angular_step_deg > 30 or self.max_joint_step_deg > 30:
            raise ValueError("Piper manual angular step limits must not exceed 30 degrees")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parent / candidate
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Piper device config does not exist: {resolved}")
    return resolved


def _load_config_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_config_path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Piper device config JSON: {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Piper device config must contain a JSON object: {config_path}")
    if data.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported Piper device config schema: {data.get('schema_version')}")
    return config_path, data


def load_manual_control_config(
    path: str | Path = DEFAULT_DEVICE_CONFIG_PATH,
    *,
    arm_side: str | None = None,
    can_name: str | None = None,
    motion_speed_percent: int | None = None,
    feedback_timeout_s: float | None = None,
    allow_manual_disabled: bool = False,
) -> ManualControlConfig:
    """Load one arm for manual control without requiring camera calibration."""

    config_path, data = _load_config_json(path)
    selected_side = arm_side or data.get("active_arm")
    arms = data.get("arms")
    if not isinstance(arms, dict) or selected_side not in arms:
        raise ValueError(f"Piper arm '{selected_side}' is not defined in {config_path}")
    arm = arms[selected_side]
    if not isinstance(arm, dict):
        raise ValueError(f"Piper arm '{selected_side}' config must be an object")
    manual_enabled = arm.get("manual_control_enabled", arm.get("enabled", False))
    if not manual_enabled and not allow_manual_disabled:
        reason = arm.get("manual_control_disabled_reason", "manual control is disabled")
        raise RuntimeError(f"Piper arm '{selected_side}' manual control is disabled: {reason}")

    motion = data.get("motion")
    manual = data.get("manual_control", {})
    if not isinstance(motion, dict):
        raise ValueError("Piper device config is missing the motion object")
    if not isinstance(manual, dict):
        raise ValueError("Piper device config manual_control must be an object")

    def required_motion(name: str) -> Any:
        if name not in motion:
            raise ValueError(f"Piper device config motion.{name} is missing")
        return motion[name]

    def manual_value(name: str, default: Any) -> Any:
        return manual.get(name, default)

    resolved_can_name = arm.get("can_interface") if can_name is None else can_name
    return ManualControlConfig(
        config_path=str(config_path),
        device_name=str(data.get("device_name", "unknown")),
        arm_side=str(selected_side),
        can_name=str(resolved_can_name or "").strip(),
        motion_speed_percent=int(
            required_motion("motion_speed_percent")
            if motion_speed_percent is None
            else motion_speed_percent
        ),
        gripper_max_width_m=float(required_motion("gripper_max_width_m")),
        max_reach_m=float(required_motion("max_reach_m")),
        min_z_m=float(required_motion("min_z_m")),
        max_z_m=float(required_motion("max_z_m")),
        max_abs_x_m=float(required_motion("max_abs_x_m")),
        max_abs_y_m=float(required_motion("max_abs_y_m")),
        enable_timeout_s=float(required_motion("enable_timeout_s")),
        move_timeout_s=float(required_motion("move_timeout_s")),
        gripper_timeout_s=float(required_motion("gripper_timeout_s")),
        position_tolerance_m=float(required_motion("position_tolerance_m")),
        angle_tolerance_deg=float(required_motion("angle_tolerance_deg")),
        gripper_tolerance_m=float(required_motion("gripper_tolerance_m")),
        feedback_timeout_s=float(
            manual_value("feedback_timeout_s", 2.0)
            if feedback_timeout_s is None
            else feedback_timeout_s
        ),
        watch_hz=float(manual_value("watch_hz", 5.0)),
        max_cartesian_step_m=float(manual_value("max_cartesian_step_m", 0.02)),
        max_angular_step_deg=float(manual_value("max_angular_step_deg", 5.0)),
        max_joint_step_deg=float(manual_value("max_joint_step_deg", 5.0)),
        joint_limit_margin_deg=float(manual_value("joint_limit_margin_deg", 2.0)),
        self_test_z_step_m=float(manual_value("self_test_z_step_m", 0.01)),
        self_test_joint_6_step_deg=float(
            manual_value("self_test_joint_6_step_deg", 3.0)
        ),
        self_test_gripper_step_m=float(manual_value("self_test_gripper_step_m", 0.005)),
    )


def meters_to_micrometers(value: float) -> int:
    return int(round(value * 1e6))


def degrees_to_millidegrees(value: float) -> int:
    return int(round(value * 1e3))


def _enum_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raw = getattr(value, "value", value)
        return int(raw)


def _label(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"unknown_0x{value:02x}")


def _hz(message: Any) -> float:
    return float(getattr(message, "Hz", 0.0) or 0.0)


def _flag_dict(value: Any, names: tuple[str, ...]) -> dict[str, bool]:
    return {name: bool(getattr(value, name, False)) for name in names}


def _end_pose_from_message(message: Any) -> ManualPose:
    pose = message.end_pose
    return ManualPose(
        x_m=float(pose.X_axis) / 1e6,
        y_m=float(pose.Y_axis) / 1e6,
        z_m=float(pose.Z_axis) / 1e6,
        rx_deg=float(pose.RX_axis) / 1e3,
        ry_deg=float(pose.RY_axis) / 1e3,
        rz_deg=float(pose.RZ_axis) / 1e3,
    )


def _joint_degrees(message: Any) -> list[float]:
    state = message.joint_state
    return [float(getattr(state, f"joint_{index}")) / 1e3 for index in range(1, 7)]


def _motor_snapshot(index: int, high_message: Any, low_message: Any) -> dict[str, Any]:
    high = getattr(high_message, f"motor_{index}")
    low = getattr(low_message, f"motor_{index}")
    status = getattr(low, "foc_status", object())
    flags = _flag_dict(status, MOTOR_FAULT_FIELDS + ("driver_enable_status",))
    effort = getattr(high, "effort", None)
    return {
        "joint": index,
        "speed_rad_s": float(getattr(high, "motor_speed", 0)) / 1e3,
        "current_a": float(getattr(high, "current", 0)) / 1e3,
        "position_raw_rad": int(getattr(high, "pos", 0)),
        "effort_nm": None if effort is None else float(effort) / 1e3,
        "voltage_v": float(getattr(low, "vol", 0)) / 10.0,
        "driver_temperature_c": float(getattr(low, "foc_temp", 0)),
        "motor_temperature_c": float(getattr(low, "motor_temp", 0)),
        "bus_current_a": float(getattr(low, "bus_current", 0)) / 1e3,
        "status": flags,
    }


def _arm_error_flags(err_code: int) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for index in range(1, 7):
        result[f"joint_{index}_communication_error"] = bool(err_code & (1 << (index - 1)))
        result[f"joint_{index}_angle_limit"] = bool(err_code & (1 << (index + 7)))
    return result


def collect_snapshot(
    piper: Any,
    config: ManualControlConfig,
    *,
    require_full_feedback: bool = True,
) -> dict[str, Any]:
    """Collect one normalized, JSON-safe diagnostic snapshot."""

    status_message = piper.GetArmStatus()
    end_message = piper.GetArmEndPoseMsgs()
    joint_message = piper.GetArmJointMsgs()
    gripper_message = piper.GetArmGripperMsgs()
    high_message = piper.GetArmHighSpdInfoMsgs()
    low_message = piper.GetArmLowSpdInfoMsgs()

    arm_status = status_message.arm_status
    ctrl_mode = _enum_int(getattr(arm_status, "ctrl_mode", 0))
    state_code = _enum_int(getattr(arm_status, "arm_status", 0))
    mode_feed = _enum_int(getattr(arm_status, "mode_feed", 0))
    teach_status = _enum_int(getattr(arm_status, "teach_status", 0))
    motion_status = _enum_int(getattr(arm_status, "motion_status", 0))
    err_code = int(getattr(arm_status, "err_code", 0))
    pose = _end_pose_from_message(end_message)
    gripper = gripper_message.gripper_state
    gripper_status = _flag_dict(
        getattr(gripper, "foc_status", object()),
        GRIPPER_FAULT_FIELDS + ("driver_enable_status", "homing_status"),
    )
    motors = [_motor_snapshot(index, high_message, low_message) for index in range(1, 7)]
    enabled = [bool(motor["status"]["driver_enable_status"]) for motor in motors]
    feedback_hz = {
        "can": float(piper.GetCanFps()),
        "arm_status": _hz(status_message),
        "end_pose": _hz(end_message),
        "joints": _hz(joint_message),
        "gripper": _hz(gripper_message),
        "motor_high_speed": _hz(high_message),
        "motor_low_speed": _hz(low_message),
    }

    problems: list[str] = []
    required_streams = (
        "can",
        "arm_status",
        "end_pose",
        "joints",
        "gripper",
        "motor_high_speed",
        "motor_low_speed",
    )
    if not require_full_feedback:
        required_streams = ("can", "arm_status", "end_pose", "joints", "motor_low_speed")
    for name in required_streams:
        if feedback_hz[name] <= 0:
            problems.append(f"missing_{name}_feedback")
    if state_code != 0:
        problems.append(f"arm_status_{_label(ARM_STATUS_LABELS, state_code)}")
    if err_code:
        problems.append(f"arm_error_code_0x{err_code:04x}")
    for motor in motors:
        for name in MOTOR_FAULT_FIELDS:
            if motor["status"][name]:
                problems.append(f"joint_{motor['joint']}_{name}")
    for name in GRIPPER_FAULT_FIELDS:
        if gripper_status[name]:
            problems.append(f"gripper_{name}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": config.device_name,
            "arm_side": config.arm_side,
            "can_interface": config.can_name,
            "config_path": config.config_path,
        },
        "connection": {"feedback_hz": feedback_hz},
        "arm": {
            "ctrl_mode": {"code": ctrl_mode, "label": _label(CTRL_MODE_LABELS, ctrl_mode)},
            "status": {"code": state_code, "label": _label(ARM_STATUS_LABELS, state_code)},
            "move_mode": {"code": mode_feed, "label": _label(MOVE_MODE_LABELS, mode_feed)},
            "teach_status": {
                "code": teach_status,
                "label": _label(TEACH_STATUS_LABELS, teach_status),
            },
            "motion_status": {
                "code": motion_status,
                "label": _label(MOTION_STATUS_LABELS, motion_status),
            },
            "trajectory_num": int(getattr(arm_status, "trajectory_num", 0)),
            "error_code": err_code,
            "error_flags": _arm_error_flags(err_code),
            "enabled_joints": enabled,
            "all_enabled": all(enabled),
        },
        "end_pose": asdict(pose),
        "joints_deg": _joint_degrees(joint_message),
        "gripper": {
            "width_m": float(getattr(gripper, "grippers_angle", 0)) / 1e6,
            "effort_nm": float(getattr(gripper, "grippers_effort", 0)) / 1e3,
            "status": gripper_status,
        },
        "motors": motors,
        "healthy": not problems,
        "problems": problems,
    }


def wait_for_feedback(
    piper: Any,
    config: ManualControlConfig,
    *,
    require_full_feedback: bool,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.feedback_timeout_s
    last_snapshot: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_snapshot = collect_snapshot(
            piper,
            config,
            require_full_feedback=require_full_feedback,
        )
        missing = [item for item in last_snapshot["problems"] if item.startswith("missing_")]
        if not missing:
            return last_snapshot
        time.sleep(0.05)
    missing_text = "unknown"
    if last_snapshot is not None:
        missing_text = ", ".join(
            item for item in last_snapshot["problems"] if item.startswith("missing_")
        )
    raise TimeoutError(
        f"Piper feedback did not become ready within {config.feedback_timeout_s:.1f}s: "
        f"{missing_text}"
    )


class PiperSession:
    """One-arm CAN session with deterministic disconnect behavior."""

    def __init__(
        self,
        config: ManualControlConfig,
        *,
        piper_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self._factory = piper_factory
        self.piper: Any | None = None
        self.hardware_action_started = False

    def __enter__(self) -> PiperSession:
        factory = self._factory
        if factory is None:
            try:
                from piper_sdk import C_PiperInterface_V2
            except ImportError as exc:
                raise ImportError("Piper control requires piper_sdk>=0.2.17") from exc
            factory = C_PiperInterface_V2
        self.piper = factory(self.config.can_name)
        self.piper.ConnectPort()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.piper is not None:
            with suppress(Exception):
                self.piper.DisconnectPort()

    def require_piper(self) -> Any:
        if self.piper is None:
            raise RuntimeError("Piper CAN session is not connected")
        return self.piper

    def snapshot(self, *, full: bool = True, wait: bool = True) -> dict[str, Any]:
        piper = self.require_piper()
        if wait:
            return wait_for_feedback(
                piper,
                self.config,
                require_full_feedback=full,
            )
        return collect_snapshot(piper, self.config, require_full_feedback=full)

    def mark_action_started(self) -> None:
        self.hardware_action_started = True

    def emergency_stop(self) -> None:
        request_emergency_stop(self.require_piper())


def get_enable_status(piper: Any) -> list[bool]:
    if hasattr(piper, "GetArmEnableStatus"):
        return [bool(value) for value in piper.GetArmEnableStatus()]
    low = piper.GetArmLowSpdInfoMsgs()
    return [
        bool(getattr(low, f"motor_{index}").foc_status.driver_enable_status)
        for index in range(1, 7)
    ]


def enable_with_timeout(piper: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hasattr(piper, "EnablePiper"):
            if piper.EnablePiper():
                return
        else:
            piper.EnableArm(7, 0x02)
            if all(get_enable_status(piper)):
                return
        time.sleep(0.05)
    raise TimeoutError(f"Piper could not be enabled within {timeout_s:.1f}s")


def disable_with_timeout(piper: Any, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hasattr(piper, "DisablePiper"):
            piper.DisablePiper()
        else:
            piper.DisableArm(7, 0x01)
        if not any(get_enable_status(piper)):
            return
        time.sleep(0.05)
    raise TimeoutError(f"Piper could not be disabled within {timeout_s:.1f}s")


def request_emergency_stop(piper: Any, *, recover: bool = False) -> None:
    command = 0x02 if recover else 0x01
    if hasattr(piper, "EmergencyStop"):
        piper.EmergencyStop(command)
    else:
        piper.MotionCtrl_1(command, 0x00, 0x00)


def validate_pose(pose: ManualPose, config: ManualControlConfig) -> None:
    values = asdict(pose).values()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Piper target pose contains a non-finite value")
    radius = math.sqrt(pose.x_m**2 + pose.y_m**2 + pose.z_m**2)
    if radius > config.max_reach_m:
        raise ValueError(
            f"Piper target pose exceeds maximum reach: {radius:.4f}m > "
            f"{config.max_reach_m:.4f}m"
        )
    if abs(pose.x_m) > config.max_abs_x_m or abs(pose.y_m) > config.max_abs_y_m:
        raise ValueError("Piper target pose exceeds configured X/Y workspace limits")
    if not config.min_z_m <= pose.z_m <= config.max_z_m:
        raise ValueError(f"Piper target Z is outside the configured range: {pose.z_m:.4f}m")
    if max(abs(pose.rx_deg), abs(pose.ry_deg), abs(pose.rz_deg)) > 90.0:
        raise ValueError("Piper target orientation is outside [-90, 90] degrees")


def offset_pose(pose: ManualPose, field: str, delta: float) -> ManualPose:
    if field not in {"x_m", "y_m", "z_m", "rx_deg", "ry_deg", "rz_deg"}:
        raise ValueError(f"unsupported Piper pose field: {field}")
    return replace(pose, **{field: getattr(pose, field) + delta})


def _angular_error_mdeg(actual: int, target: int) -> int:
    return abs((actual - target + 180_000) % 360_000 - 180_000)


def move_pose_with_check(
    piper: Any,
    pose: ManualPose,
    config: ManualControlConfig,
) -> None:
    target = pose.command()
    position_tolerance = meters_to_micrometers(config.position_tolerance_m)
    angle_tolerance = degrees_to_millidegrees(config.angle_tolerance_deg)
    deadline = time.monotonic() + config.move_timeout_s
    last_actual: tuple[int, int, int, int, int, int] | None = None
    piper.MotionCtrl_2(0x01, 0x00, config.motion_speed_percent, 0x00)
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
        f"Piper failed to reach pose within {config.move_timeout_s:.1f}s; "
        f"target={target}, last_feedback={last_actual}"
    )


def current_joint_targets(piper: Any) -> tuple[int, int, int, int, int, int]:
    state = piper.GetArmJointMsgs().joint_state
    return tuple(int(getattr(state, f"joint_{index}")) for index in range(1, 7))


def get_joint_limits_deg(piper: Any) -> list[tuple[float, float]]:
    wrapper = piper.GetAllMotorAngleLimitMaxSpd()
    timestamp = float(getattr(wrapper, "time_stamp", 0.0) or 0.0)
    limits = wrapper.all_motor_angle_limit_max_spd.motor
    result: list[tuple[float, float]] = []
    for index in range(1, 7):
        item = limits[index]
        motor_num = int(getattr(item, "motor_num", 0))
        minimum = float(getattr(item, "min_angle_limit", 0)) / 10.0
        maximum = float(getattr(item, "max_angle_limit", 0)) / 10.0
        if motor_num != index or minimum >= maximum:
            raise RuntimeError("Piper joint-limit feedback is not ready")
        result.append((minimum, maximum))
    if timestamp <= 0:
        raise RuntimeError("Piper joint-limit feedback is not ready")
    return result


def wait_for_joint_limits(piper: Any, timeout_s: float) -> list[tuple[float, float]]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return get_joint_limits_deg(piper)
        except (AttributeError, KeyError, RuntimeError, TypeError) as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"Piper joint-limit feedback unavailable: {last_error}")


def validate_joint_targets(
    target_mdeg: tuple[int, int, int, int, int, int],
    limits_deg: list[tuple[float, float]],
    margin_deg: float,
) -> None:
    if len(limits_deg) != 6:
        raise ValueError("Piper joint limit list must contain six joints")
    for index, (target, limits) in enumerate(zip(target_mdeg, limits_deg, strict=True), start=1):
        target_deg = target / 1e3
        minimum, maximum = limits
        safe_min = minimum + margin_deg
        safe_max = maximum - margin_deg
        if not safe_min <= target_deg <= safe_max:
            raise ValueError(
                f"Piper J{index} target {target_deg:.3f}deg is outside the safe limit "
                f"[{safe_min:.3f}, {safe_max:.3f}]deg"
            )


def move_joints_with_check(
    piper: Any,
    target_mdeg: tuple[int, int, int, int, int, int],
    config: ManualControlConfig,
) -> None:
    tolerance = degrees_to_millidegrees(config.angle_tolerance_deg)
    deadline = time.monotonic() + config.move_timeout_s
    last_actual: tuple[int, int, int, int, int, int] | None = None
    piper.MotionCtrl_2(0x01, 0x01, config.motion_speed_percent, 0x00)
    while time.monotonic() < deadline:
        piper.JointCtrl(*target_mdeg)
        last_actual = current_joint_targets(piper)
        if all(
            _angular_error_mdeg(actual, target) <= tolerance
            for actual, target in zip(last_actual, target_mdeg, strict=True)
        ):
            return
        time.sleep(0.02)
    raise TimeoutError(
        f"Piper failed to reach joint target within {config.move_timeout_s:.1f}s; "
        f"target={target_mdeg}, last_feedback={last_actual}"
    )


def set_gripper_with_check(
    piper: Any,
    width_m: float,
    config: ManualControlConfig,
) -> None:
    if not math.isfinite(width_m) or not 0 <= width_m <= config.gripper_max_width_m:
        raise ValueError(
            f"Piper gripper width must be between 0 and {config.gripper_max_width_m:.4f}m"
        )
    target = meters_to_micrometers(width_m)
    tolerance = meters_to_micrometers(config.gripper_tolerance_m)
    deadline = time.monotonic() + config.gripper_timeout_s
    last_actual: int | None = None
    while time.monotonic() < deadline:
        piper.GripperCtrl(target, 1000, 0x01, 0)
        last_actual = int(piper.GetArmGripperMsgs().gripper_state.grippers_angle)
        if abs(last_actual - target) <= tolerance:
            return
        time.sleep(0.02)
    raise TimeoutError(
        f"Piper gripper failed to reach {width_m:.4f}m within "
        f"{config.gripper_timeout_s:.1f}s; last_feedback={last_actual}"
    )


def require_motion_ready(snapshot: dict[str, Any]) -> None:
    missing = [item for item in snapshot["problems"] if item.startswith("missing_")]
    if missing:
        raise RuntimeError(f"Piper feedback is incomplete: {', '.join(missing)}")
    if snapshot["arm"]["status"]["code"] != 0:
        raise RuntimeError(f"Piper arm status is {snapshot['arm']['status']['label']}")
    if snapshot["arm"]["error_code"] != 0:
        raise RuntimeError(
            f"Piper arm error code is 0x{snapshot['arm']['error_code']:04x}"
        )
    if not snapshot["arm"]["all_enabled"]:
        raise RuntimeError("Piper motion requires all six joints to be enabled")
    motor_faults = [
        problem
        for problem in snapshot["problems"]
        if problem.startswith("joint_") and not problem.endswith("driver_enable_status")
    ]
    if motor_faults:
        raise RuntimeError(f"Piper motor fault detected: {', '.join(motor_faults)}")


def pose_from_snapshot(snapshot: dict[str, Any]) -> ManualPose:
    return ManualPose(**snapshot["end_pose"])


def choose_signed_delta(
    positive_value: float,
    *,
    positive_valid: Callable[[float], bool],
    negative_valid: Callable[[float], bool],
) -> float:
    if positive_valid(positive_value):
        return positive_value
    if negative_valid(-positive_value):
        return -positive_value
    raise ValueError("Piper self-test has no safe positive or negative direction from this state")
