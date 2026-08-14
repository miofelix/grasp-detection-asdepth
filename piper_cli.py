#!/usr/bin/env python3
"""CLI and interactive console for Piper status and guarded motion testing."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from dataclasses import asdict
from typing import Any, Sequence

from piper_control import (
    DEFAULT_DEVICE_CONFIG_PATH,
    ManualControlConfig,
    PiperSession,
    choose_signed_delta,
    current_joint_targets,
    disable_with_timeout,
    enable_with_timeout,
    load_manual_control_config,
    move_joints_with_check,
    move_pose_with_check,
    offset_pose,
    pose_from_snapshot,
    request_emergency_stop,
    require_motion_ready,
    set_gripper_with_check,
    validate_joint_targets,
    validate_pose,
    wait_for_joint_limits,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_UNHEALTHY = 2


class PiperArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _add_commands(subparsers: Any, *, include_shell: bool) -> None:
    subparsers.add_parser("status", help="print one complete arm diagnostic snapshot")

    watch = subparsers.add_parser("watch", help="continuously refresh complete arm status")
    watch.add_argument("--hz", type=float, default=None, help="display refresh rate")

    if include_shell:
        subparsers.add_parser("shell", help="open an interactive console using one CAN session")

    subparsers.add_parser("enable", help="enable all six arm joints")
    subparsers.add_parser("disable", help="disable all joints; the arm may fall under gravity")
    subparsers.add_parser("stop", help="request an immediate emergency stop")
    subparsers.add_parser("recover", help="request recovery from emergency-stop state")

    cartesian = subparsers.add_parser(
        "jog-cartesian",
        help="move one Cartesian axis relative to the current feedback pose",
    )
    axis_group = cartesian.add_mutually_exclusive_group(required=True)
    axis_group.add_argument("--x-mm", type=float)
    axis_group.add_argument("--y-mm", type=float)
    axis_group.add_argument("--z-mm", type=float)
    axis_group.add_argument("--rx-deg", type=float)
    axis_group.add_argument("--ry-deg", type=float)
    axis_group.add_argument("--rz-deg", type=float)

    joint = subparsers.add_parser(
        "jog-joint",
        help="move one joint relative to its current feedback angle",
    )
    joint.add_argument("--joint", type=int, choices=range(1, 7), required=True)
    joint.add_argument("--delta-deg", type=float, required=True)

    gripper = subparsers.add_parser("gripper", help="move the gripper to an absolute width")
    gripper.add_argument("--width-mm", type=float, required=True)

    subparsers.add_parser(
        "self-test",
        help="run guarded Z, J6 and gripper out-and-back checks",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = PiperArgumentParser(
        description="Inspect and safely test a Piper arm without loading any vision stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm-config", default=DEFAULT_DEVICE_CONFIG_PATH)
    parser.add_argument("--arm-side", choices=["left", "right"])
    parser.add_argument("--arm-can-interface")
    parser.add_argument("--speed-percent", type=int)
    parser.add_argument("--feedback-timeout", type=float)
    parser.add_argument(
        "--execute-arm",
        action="store_true",
        help="unlock commands that can energize or move hardware",
    )
    parser.add_argument(
        "--confirm-arm-motion",
        action="store_true",
        help="second startup-level confirmation for hardware actions",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_commands(subparsers, include_shell=True)
    return parser


def build_shell_parser() -> argparse.ArgumentParser:
    parser = PiperArgumentParser(prog="piper", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_commands(subparsers, include_shell=False)
    return parser


def _validate_global_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.execute_arm != args.confirm_arm_motion:
        parser.error("--execute-arm and --confirm-arm-motion must be supplied together")
    if args.arm_can_interface == "":
        parser.error("--arm-can-interface must not be empty")
    if args.speed_percent is not None and not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be between 1 and 100")
    if args.feedback_timeout is not None and args.feedback_timeout <= 0:
        parser.error("--feedback-timeout must be positive")
    if args.command == "watch" and args.hz is not None and args.hz <= 0:
        parser.error("watch --hz must be positive")


def _hardware_unlocked(args: argparse.Namespace) -> bool:
    return bool(args.execute_arm and args.confirm_arm_motion)


def _require_unlocked(args: argparse.Namespace) -> None:
    if not _hardware_unlocked(args):
        raise PermissionError(
            "hardware action is locked; add --execute-arm --confirm-arm-motion before the subcommand"
        )


def confirm_action(message: str) -> bool:
    if not sys.stdin.isatty():
        raise PermissionError("hardware actions require an interactive terminal for y/N confirmation")
    print(message)
    answer = input("Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def print_snapshot(snapshot: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        _print_json(snapshot)
        return

    device = snapshot["device"]
    arm = snapshot["arm"]
    pose = snapshot["end_pose"]
    gripper = snapshot["gripper"]
    print(
        f"Piper {device['arm_side']} arm on {device['can_interface']} | "
        f"healthy={_format_bool(snapshot['healthy'])}"
    )
    print(
        "Status: "
        f"{arm['status']['label']} | ctrl={arm['ctrl_mode']['label']} | "
        f"mode={arm['move_mode']['label']} | motion={arm['motion_status']['label']} | "
        f"error=0x{arm['error_code']:04x}"
    )
    enabled = " ".join(
        f"J{index}={'on' if value else 'off'}"
        for index, value in enumerate(arm["enabled_joints"], start=1)
    )
    print(f"Enable: {enabled}")
    print(
        "End pose: "
        f"x={pose['x_m']:.4f}m y={pose['y_m']:.4f}m z={pose['z_m']:.4f}m "
        f"rx={pose['rx_deg']:.3f}° ry={pose['ry_deg']:.3f}° rz={pose['rz_deg']:.3f}°"
    )
    print(
        "Joints: "
        + " ".join(
            f"J{index}={angle:.3f}°"
            for index, angle in enumerate(snapshot["joints_deg"], start=1)
        )
    )
    print(
        f"Gripper: width={gripper['width_m'] * 1000:.3f}mm "
        f"effort={gripper['effort_nm']:.3f}N·m "
        f"enabled={_format_bool(gripper['status']['driver_enable_status'])} "
        f"homed={_format_bool(gripper['status']['homing_status'])}"
    )
    print("Motors:")
    print("  J   speed(rad/s) current(A) bus(A) voltage(V) driver°C motor°C enabled flags")
    for motor in snapshot["motors"]:
        flags = [
            name
            for name, active in motor["status"].items()
            if active and name != "driver_enable_status"
        ]
        flag_text = ",".join(flags) if flags else "-"
        print(
            f"  {motor['joint']:<2} {motor['speed_rad_s']:>12.3f} "
            f"{motor['current_a']:>10.3f} {motor['bus_current_a']:>6.3f} "
            f"{motor['voltage_v']:>10.1f} {motor['driver_temperature_c']:>7.1f} "
            f"{motor['motor_temperature_c']:>6.1f} "
            f"{_format_bool(motor['status']['driver_enable_status']):>7} {flag_text}"
        )
    rates = snapshot["connection"]["feedback_hz"]
    print("Feedback Hz: " + " ".join(f"{name}={value:.1f}" for name, value in rates.items()))
    if snapshot["problems"]:
        print("Problems: " + ", ".join(snapshot["problems"]))


def _action_result(
    action: str,
    *,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    target: Any = None,
    executed: bool = True,
) -> dict[str, Any]:
    return {
        "action": action,
        "executed": executed,
        "target": target,
        "before": before,
        "after": after,
        "may_have_moved": bool(executed),
    }


def _emit_action_result(result: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        _print_json(result)
        return
    state = "complete" if result["executed"] else "cancelled"
    print(f"Piper action {result['action']}: {state}")
    if result.get("after") is not None:
        print_snapshot(result["after"], as_json=False)


def _cancelled_result(action: str, before: dict[str, Any], target: Any) -> dict[str, Any]:
    return _action_result(action, before=before, target=target, executed=False)


def _run_status(session: PiperSession, args: argparse.Namespace) -> int:
    snapshot = session.snapshot(full=True, wait=True)
    print_snapshot(snapshot, as_json=args.json)
    return EXIT_OK if snapshot["healthy"] else EXIT_UNHEALTHY


def _run_watch(session: PiperSession, args: argparse.Namespace) -> int:
    rate = session.config.watch_hz if args.hz is None else float(args.hz)
    if rate <= 0:
        raise ValueError("watch refresh rate must be positive")
    session.snapshot(full=True, wait=True)
    interval = 1.0 / rate
    try:
        while True:
            snapshot = session.snapshot(full=True, wait=False)
            if args.json:
                print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), flush=True)
            else:
                if sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                print_snapshot(snapshot, as_json=False)
                print("Press Ctrl-C to stop watching.")
            time.sleep(interval)
    except KeyboardInterrupt:
        return EXIT_OK


def _run_enable(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    target = {"enabled_joints": [True] * 6}
    if not confirm_action(
        f"Enable all six joints on {session.config.arm_side}/{session.config.can_name}?"
    ):
        _emit_action_result(_cancelled_result("enable", before, target), as_json=args.json)
        return EXIT_OK
    session.mark_action_started()
    enable_with_timeout(session.require_piper(), session.config.enable_timeout_s)
    after = session.snapshot(full=False, wait=False)
    if not after["arm"]["all_enabled"]:
        raise RuntimeError("Piper reported that one or more joints are still disabled")
    _emit_action_result(
        _action_result("enable", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


def _run_disable(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    target = {"enabled_joints": [False] * 6}
    warning = (
        f"Disable all joints on {session.config.arm_side}/{session.config.can_name}? "
        "WARNING: the arm may lose support and fall under gravity."
    )
    if not confirm_action(warning):
        _emit_action_result(_cancelled_result("disable", before, target), as_json=args.json)
        return EXIT_OK
    session.mark_action_started()
    disable_with_timeout(session.require_piper(), session.config.enable_timeout_s)
    after = session.snapshot(full=False, wait=False)
    if any(after["arm"]["enabled_joints"]):
        raise RuntimeError("Piper reported that one or more joints are still enabled")
    _emit_action_result(
        _action_result("disable", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


def _run_stop(session: PiperSession, args: argparse.Namespace) -> int:
    request_emergency_stop(session.require_piper())
    result = {
        "action": "stop",
        "executed": True,
        "may_have_moved": True,
        "message": "emergency stop requested",
    }
    if args.json:
        _print_json(result)
    else:
        print("Piper emergency stop requested.")
    return EXIT_OK


def _run_recover(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    target = {"emergency_stop": "recover", "auto_enable": False}
    if not confirm_action(
        f"Recover emergency-stop state on {session.config.arm_side}/{session.config.can_name}? "
        "The arm will not be enabled automatically."
    ):
        _emit_action_result(_cancelled_result("recover", before, target), as_json=args.json)
        return EXIT_OK
    session.mark_action_started()
    request_emergency_stop(session.require_piper(), recover=True)
    time.sleep(0.1)
    after = session.snapshot(full=False, wait=False)
    _emit_action_result(
        _action_result("recover", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


def _cartesian_request(args: argparse.Namespace) -> tuple[str, float, str]:
    choices = (
        ("x_mm", "x_m", 1e-3, "mm"),
        ("y_mm", "y_m", 1e-3, "mm"),
        ("z_mm", "z_m", 1e-3, "mm"),
        ("rx_deg", "rx_deg", 1.0, "deg"),
        ("ry_deg", "ry_deg", 1.0, "deg"),
        ("rz_deg", "rz_deg", 1.0, "deg"),
    )
    for argument, field, factor, unit in choices:
        value = getattr(args, argument, None)
        if value is not None:
            return field, float(value) * factor, unit
    raise ValueError("one Cartesian jog axis must be supplied")


def _run_jog_cartesian(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    require_motion_ready(before)
    field, delta, unit = _cartesian_request(args)
    step_limit = (
        session.config.max_cartesian_step_m
        if field.endswith("_m")
        else session.config.max_angular_step_deg
    )
    if not 0 < abs(delta) <= step_limit:
        display_limit = step_limit * 1000 if unit == "mm" else step_limit
        raise ValueError(f"Cartesian jog must be non-zero and no larger than {display_limit:g}{unit}")
    start = pose_from_snapshot(before)
    target_pose = offset_pose(start, field, delta)
    validate_pose(target_pose, session.config)
    target = asdict(target_pose)
    if not confirm_action(
        f"Cartesian jog on {session.config.arm_side}/{session.config.can_name}:\n"
        f"  current={asdict(start)}\n  target={target}"
    ):
        _emit_action_result(
            _cancelled_result("jog-cartesian", before, target),
            as_json=args.json,
        )
        return EXIT_OK
    session.mark_action_started()
    move_pose_with_check(session.require_piper(), target_pose, session.config)
    after = session.snapshot(full=False, wait=False)
    _emit_action_result(
        _action_result("jog-cartesian", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


def _run_jog_joint(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    require_motion_ready(before)
    delta = float(args.delta_deg)
    if not 0 < abs(delta) <= session.config.max_joint_step_deg:
        raise ValueError(
            "joint jog must be non-zero and no larger than "
            f"{session.config.max_joint_step_deg:g}deg"
        )
    piper = session.require_piper()
    limits = wait_for_joint_limits(piper, session.config.feedback_timeout_s)
    start = current_joint_targets(piper)
    target_list = list(start)
    target_list[args.joint - 1] += int(round(delta * 1000))
    target = tuple(target_list)
    validate_joint_targets(target, limits, session.config.joint_limit_margin_deg)
    target_deg = [value / 1000.0 for value in target]
    if not confirm_action(
        f"Joint jog on {session.config.arm_side}/{session.config.can_name}:\n"
        f"  J{args.joint} delta={delta:.3f}deg\n"
        f"  current={[value / 1000.0 for value in start]}\n  target={target_deg}"
    ):
        _emit_action_result(
            _cancelled_result("jog-joint", before, {"joints_deg": target_deg}),
            as_json=args.json,
        )
        return EXIT_OK
    session.mark_action_started()
    move_joints_with_check(piper, target, session.config)
    after = session.snapshot(full=False, wait=False)
    _emit_action_result(
        _action_result(
            "jog-joint",
            before=before,
            after=after,
            target={"joints_deg": target_deg},
        ),
        as_json=args.json,
    )
    return EXIT_OK


def _run_gripper(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=False, wait=True)
    require_motion_ready(before)
    _require_gripper_ready(before)
    width_m = float(args.width_mm) / 1000.0
    if not 0 <= width_m <= session.config.gripper_max_width_m:
        raise ValueError(
            "gripper width must be between 0 and "
            f"{session.config.gripper_max_width_m * 1000:.1f}mm"
        )
    target = {"width_m": width_m}
    if not confirm_action(
        f"Move gripper on {session.config.arm_side}/{session.config.can_name}: "
        f"{before['gripper']['width_m'] * 1000:.3f}mm -> {args.width_mm:.3f}mm?"
    ):
        _emit_action_result(_cancelled_result("gripper", before, target), as_json=args.json)
        return EXIT_OK
    session.mark_action_started()
    set_gripper_with_check(session.require_piper(), width_m, session.config)
    after = session.snapshot(full=False, wait=False)
    _emit_action_result(
        _action_result("gripper", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


def _require_gripper_ready(snapshot: dict[str, Any]) -> None:
    gripper_faults = [
        name
        for name, active in snapshot["gripper"]["status"].items()
        if active and name not in {"driver_enable_status", "homing_status"}
    ]
    if gripper_faults:
        raise RuntimeError(f"Piper gripper fault detected: {', '.join(gripper_faults)}")


def _pose_delta_is_valid(start: Any, delta: float, config: ManualControlConfig) -> bool:
    try:
        validate_pose(offset_pose(start, "z_m", delta), config)
    except ValueError:
        return False
    return True


def _joint_delta_is_valid(
    start: tuple[int, int, int, int, int, int],
    delta_deg: float,
    limits: list[tuple[float, float]],
    config: ManualControlConfig,
) -> bool:
    target = list(start)
    target[5] += int(round(delta_deg * 1000))
    try:
        validate_joint_targets(tuple(target), limits, config.joint_limit_margin_deg)
    except ValueError:
        return False
    return True


def _run_self_test(session: PiperSession, args: argparse.Namespace) -> int:
    _require_unlocked(args)
    before = session.snapshot(full=True, wait=True)
    require_motion_ready(before)
    _require_gripper_ready(before)
    piper = session.require_piper()
    config = session.config
    start_pose = pose_from_snapshot(before)
    start_joints = current_joint_targets(piper)
    limits = wait_for_joint_limits(piper, config.feedback_timeout_s)
    start_gripper = float(before["gripper"]["width_m"])

    z_delta = choose_signed_delta(
        config.self_test_z_step_m,
        positive_valid=lambda value: _pose_delta_is_valid(start_pose, value, config),
        negative_valid=lambda value: _pose_delta_is_valid(start_pose, value, config),
    )
    joint_delta = choose_signed_delta(
        config.self_test_joint_6_step_deg,
        positive_valid=lambda value: _joint_delta_is_valid(start_joints, value, limits, config),
        negative_valid=lambda value: _joint_delta_is_valid(start_joints, value, limits, config),
    )
    gripper_delta = choose_signed_delta(
        config.self_test_gripper_step_m,
        positive_valid=lambda value: start_gripper + value <= config.gripper_max_width_m,
        negative_valid=lambda value: start_gripper + value >= 0,
    )

    z_target = offset_pose(start_pose, "z_m", z_delta)
    joint_target_list = list(start_joints)
    joint_target_list[5] += int(round(joint_delta * 1000))
    joint_target = tuple(joint_target_list)
    gripper_target = start_gripper + gripper_delta
    validate_pose(z_target, config)
    validate_joint_targets(joint_target, limits, config.joint_limit_margin_deg)

    target = {
        "z": {"start_m": start_pose.z_m, "test_m": z_target.z_m},
        "joint_6": {
            "start_deg": start_joints[5] / 1000.0,
            "test_deg": joint_target[5] / 1000.0,
        },
        "gripper": {"start_m": start_gripper, "test_m": gripper_target},
        "sequence": [
            "z_test",
            "z_return",
            "joint_6_test",
            "joint_6_return",
            "gripper_test",
            "gripper_return",
        ],
    }
    if not confirm_action(
        f"Run Piper self-test on {config.arm_side}/{config.can_name}?\n"
        f"  Z: {start_pose.z_m:.4f}m -> {z_target.z_m:.4f}m -> start\n"
        f"  J6: {start_joints[5] / 1000:.3f}deg -> "
        f"{joint_target[5] / 1000:.3f}deg -> start\n"
        f"  gripper: {start_gripper * 1000:.3f}mm -> "
        f"{gripper_target * 1000:.3f}mm -> start"
    ):
        _emit_action_result(_cancelled_result("self-test", before, target), as_json=args.json)
        return EXIT_OK

    session.mark_action_started()
    move_pose_with_check(piper, z_target, config)
    move_pose_with_check(piper, start_pose, config)
    move_joints_with_check(piper, joint_target, config)
    move_joints_with_check(piper, start_joints, config)
    set_gripper_with_check(piper, gripper_target, config)
    set_gripper_with_check(piper, start_gripper, config)
    after = session.snapshot(full=True, wait=False)
    _emit_action_result(
        _action_result("self-test", before=before, after=after, target=target),
        as_json=args.json,
    )
    return EXIT_OK


ACTION_HANDLERS = {
    "status": _run_status,
    "watch": _run_watch,
    "enable": _run_enable,
    "disable": _run_disable,
    "stop": _run_stop,
    "recover": _run_recover,
    "jog-cartesian": _run_jog_cartesian,
    "jog-joint": _run_jog_joint,
    "gripper": _run_gripper,
    "self-test": _run_self_test,
}


def _handle_action_failure(session: PiperSession, command: str) -> None:
    if not session.hardware_action_started:
        return
    if command in {"stop", "recover", "disable"}:
        return
    try:
        session.emergency_stop()
    except Exception as stop_error:
        print(f"WARNING: emergency-stop request also failed: {stop_error}", file=sys.stderr)


def _run_action(session: PiperSession, args: argparse.Namespace) -> int:
    handler = ACTION_HANDLERS.get(args.command)
    if handler is None:
        raise ValueError(f"unsupported Piper command: {args.command}")
    session.hardware_action_started = False
    try:
        return handler(session, args)
    except KeyboardInterrupt:
        _handle_action_failure(session, args.command)
        print("Piper action interrupted; emergency stop requested when motion had started.", file=sys.stderr)
        return EXIT_UNHEALTHY
    except Exception:
        _handle_action_failure(session, args.command)
        raise


def _shell_namespace(global_args: argparse.Namespace, command_args: argparse.Namespace) -> argparse.Namespace:
    values = vars(command_args).copy()
    values.update(
        {
            "execute_arm": global_args.execute_arm,
            "confirm_arm_motion": global_args.confirm_arm_motion,
            "json": global_args.json,
        }
    )
    return argparse.Namespace(**values)


def _run_shell(session: PiperSession, args: argparse.Namespace) -> int:
    parser = build_shell_parser()
    lock_state = "unlocked" if _hardware_unlocked(args) else "read-only"
    print(
        f"Connected to Piper {session.config.arm_side}/{session.config.can_name}; "
        f"shell is {lock_state}. Type 'help', 'quit', or a command."
    )
    session.snapshot(full=False, wait=True)
    while True:
        try:
            line = input("piper> ").strip()
        except EOFError:
            print()
            return EXIT_OK
        except KeyboardInterrupt:
            print("\nUse 'stop' for an emergency stop or 'quit' to exit.")
            continue
        if not line:
            continue
        if line in {"quit", "exit"}:
            return EXIT_OK
        if line == "help":
            parser.print_help()
            continue
        try:
            command_args = parser.parse_args(shlex.split(line))
        except SystemExit:
            continue
        effective_args = _shell_namespace(args, command_args)
        try:
            _run_action(session, effective_args)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)


def _load_config(args: argparse.Namespace) -> ManualControlConfig:
    return load_manual_control_config(
        args.arm_config,
        arm_side=args.arm_side,
        can_name=args.arm_can_interface,
        motion_speed_percent=args.speed_percent,
        feedback_timeout_s=args.feedback_timeout,
        allow_manual_disabled=args.command == "stop",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_global_args(parser, args)
    try:
        config = _load_config(args)
        with PiperSession(config) as session:
            if args.command == "shell":
                return _run_shell(session, args)
            return _run_action(session, args)
    except KeyboardInterrupt:
        print("Piper CLI interrupted.", file=sys.stderr)
        return EXIT_UNHEALTHY
    except PermissionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_UNHEALTHY


if __name__ == "__main__":
    raise SystemExit(main())
