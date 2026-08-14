from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import piper_cli
import piper_control


def ns(**values: object) -> SimpleNamespace:
    return SimpleNamespace(**values)


class FakePiper:
    def __init__(self, can_name: str = "can1") -> None:
        self.can_name = can_name
        self.connected = False
        self.disconnected = False
        self.pose = [15_000, 0, 275_000, 0, 85_000, 0]
        self.joints = [0, 90_000, -90_000, 0, 0, 0]
        self.gripper_width = 40_000
        self.enabled = [True] * 6
        self.arm_status_code = 0
        self.err_code = 0
        self.feedback_hz = 10.0
        self.motion_commands: list[tuple[int, int, int, int]] = []
        self.stop_commands: list[tuple[int, int, int]] = []
        self.pose_commands: list[tuple[int, ...]] = []
        self.joint_commands: list[tuple[int, ...]] = []
        self.gripper_commands: list[tuple[int, int, int, int]] = []

    def ConnectPort(self) -> None:
        self.connected = True

    def DisconnectPort(self) -> None:
        self.disconnected = True

    def GetCanFps(self) -> float:
        return self.feedback_hz

    def GetArmStatus(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            arm_status=ns(
                ctrl_mode=1,
                arm_status=self.arm_status_code,
                mode_feed=0,
                teach_status=0,
                motion_status=0,
                trajectory_num=0,
                err_code=self.err_code,
            ),
        )

    def GetArmEndPoseMsgs(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            end_pose=ns(
                X_axis=self.pose[0],
                Y_axis=self.pose[1],
                Z_axis=self.pose[2],
                RX_axis=self.pose[3],
                RY_axis=self.pose[4],
                RZ_axis=self.pose[5],
            ),
        )

    def GetArmJointMsgs(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            joint_state=ns(**{f"joint_{index}": value for index, value in enumerate(self.joints, 1)}),
        )

    def GetArmGripperMsgs(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            gripper_state=ns(
                grippers_angle=self.gripper_width,
                grippers_effort=500,
                foc_status=ns(
                    voltage_too_low=False,
                    motor_overheating=False,
                    driver_overcurrent=False,
                    driver_overheating=False,
                    sensor_status=False,
                    driver_error_status=False,
                    driver_enable_status=True,
                    homing_status=True,
                ),
            ),
        )

    def _high_motor(self, index: int) -> object:
        return ns(
            can_id=0x250 + index,
            motor_speed=index * 100,
            current=index * 200,
            pos=index,
        )

    def _low_motor(self, index: int) -> object:
        return ns(
            can_id=0x260 + index,
            vol=240,
            foc_temp=30 + index,
            motor_temp=35 + index,
            bus_current=index * 100,
            foc_status=ns(
                voltage_too_low=False,
                motor_overheating=False,
                driver_overcurrent=False,
                driver_overheating=False,
                collision_status=False,
                driver_error_status=False,
                driver_enable_status=self.enabled[index - 1],
                stall_status=False,
            ),
        )

    def GetArmHighSpdInfoMsgs(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            **{f"motor_{index}": self._high_motor(index) for index in range(1, 7)},
        )

    def GetArmLowSpdInfoMsgs(self) -> object:
        return ns(
            Hz=self.feedback_hz,
            **{f"motor_{index}": self._low_motor(index) for index in range(1, 7)},
        )

    def GetAllMotorAngleLimitMaxSpd(self) -> object:
        limits = {
            1: (-1500, 1500),
            2: (0, 1800),
            3: (-1700, 0),
            4: (-1000, 1000),
            5: (-700, 700),
            6: (-1200, 1200),
        }
        motors = {
            index: ns(
                motor_num=index,
                min_angle_limit=limits[index][0],
                max_angle_limit=limits[index][1],
                max_joint_spd=1000,
            )
            for index in range(1, 7)
        }
        return ns(time_stamp=1.0, all_motor_angle_limit_max_spd=ns(motor=motors))

    def MotionCtrl_1(self, first: int, second: int, third: int) -> None:
        self.stop_commands.append((first, second, third))

    def MotionCtrl_2(self, *values: int) -> None:
        self.motion_commands.append(tuple(values))

    def EnableArm(self, _motor: int, _flag: int) -> None:
        self.enabled = [True] * 6

    def DisableArm(self, _motor: int, _flag: int) -> None:
        self.enabled = [False] * 6

    def EndPoseCtrl(self, *target: int) -> None:
        self.pose_commands.append(tuple(target))
        self.pose = list(target)

    def JointCtrl(self, *target: int) -> None:
        self.joint_commands.append(tuple(target))
        self.joints = list(target)

    def GripperCtrl(self, width: int, effort: int, code: int, set_zero: int) -> None:
        self.gripper_commands.append((width, effort, code, set_zero))
        self.gripper_width = width


def cli_args(command: str, **values: object) -> SimpleNamespace:
    defaults = {
        "command": command,
        "execute_arm": True,
        "confirm_arm_motion": True,
        "json": False,
        "hz": None,
    }
    defaults.update(values)
    return ns(**defaults)


class PiperControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            piper_control.load_manual_control_config(),
            feedback_timeout_s=0.1,
            enable_timeout_s=0.1,
            move_timeout_s=0.1,
            gripper_timeout_s=0.1,
        )
        self.piper = FakePiper(self.config.can_name)

    def test_manual_config_allows_right_arm_without_camera_calibration(self) -> None:
        config = piper_control.load_manual_control_config(arm_side="right")

        self.assertEqual(config.arm_side, "right")
        self.assertEqual(config.can_name, "can2")
        self.assertEqual(config.max_cartesian_step_m, 0.02)

    def test_snapshot_contains_normalized_complete_diagnostics(self) -> None:
        snapshot = piper_control.collect_snapshot(self.piper, self.config)

        self.assertTrue(snapshot["healthy"])
        self.assertEqual(snapshot["arm"]["ctrl_mode"]["label"], "can_control")
        self.assertEqual(snapshot["end_pose"]["z_m"], 0.275)
        self.assertEqual(snapshot["joints_deg"][1], 90.0)
        self.assertEqual(snapshot["gripper"]["width_m"], 0.04)
        self.assertEqual(snapshot["motors"][0]["voltage_v"], 24.0)
        self.assertEqual(snapshot["motors"][0]["current_a"], 0.2)
        json.dumps(snapshot)

    def test_raw_error_code_is_decoded_without_sdk_err_status_bug(self) -> None:
        self.piper.err_code = (1 << 0) | (1 << 8)

        snapshot = piper_control.collect_snapshot(self.piper, self.config)

        self.assertFalse(snapshot["healthy"])
        self.assertTrue(snapshot["arm"]["error_flags"]["joint_1_communication_error"])
        self.assertTrue(snapshot["arm"]["error_flags"]["joint_1_angle_limit"])

    def test_feedback_timeout_reports_missing_streams(self) -> None:
        self.piper.feedback_hz = 0.0
        config = replace(self.config, feedback_timeout_s=0.01)

        with self.assertRaisesRegex(TimeoutError, "feedback did not become ready"):
            piper_control.wait_for_feedback(
                self.piper,
                config,
                require_full_feedback=True,
            )

    def test_legacy_enable_disable_and_stop_apis_are_supported(self) -> None:
        self.piper.enabled = [False] * 6

        piper_control.enable_with_timeout(self.piper, 0.1)
        self.assertTrue(all(self.piper.enabled))
        piper_control.disable_with_timeout(self.piper, 0.1)
        self.assertFalse(any(self.piper.enabled))
        piper_control.request_emergency_stop(self.piper)
        piper_control.request_emergency_stop(self.piper, recover=True)

        self.assertEqual(self.piper.stop_commands, [(1, 0, 0), (2, 0, 0)])

    def test_pose_validation_rejects_workspace_violation(self) -> None:
        pose = piper_control.ManualPose(0.0, 0.0, 0.7, 0.0, 0.0, 0.0)

        with self.assertRaisesRegex(ValueError, "maximum reach|Z is outside"):
            piper_control.validate_pose(pose, self.config)

    def test_joint_target_respects_reported_limit_margin(self) -> None:
        limits = piper_control.get_joint_limits_deg(self.piper)
        target = tuple(self.piper.joints[:-1] + [119_000])

        with self.assertRaisesRegex(ValueError, "J6 target"):
            piper_control.validate_joint_targets(target, limits, margin_deg=2.0)

    def test_session_always_disconnects(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "boom"), piper_control.PiperSession(
            self.config,
            piper_factory=lambda _name: self.piper,
        ):
            raise RuntimeError("boom")

        self.assertTrue(self.piper.connected)
        self.assertTrue(self.piper.disconnected)


class PiperCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            piper_control.load_manual_control_config(),
            feedback_timeout_s=0.1,
            enable_timeout_s=0.1,
            move_timeout_s=0.1,
            gripper_timeout_s=0.1,
        )
        self.piper = FakePiper(self.config.can_name)
        self.session = piper_control.PiperSession(
            self.config,
            piper_factory=lambda _name: self.piper,
        )
        self.session.__enter__()

    def tearDown(self) -> None:
        self.session.__exit__(None, None, None)

    def test_import_does_not_load_visual_modules(self) -> None:
        self.assertNotIn("camera_capture", sys.modules)
        self.assertNotIn("get_pose", sys.modules)
        self.assertNotIn("open3d", sys.modules)

    def test_argument_errors_exit_with_usage_code_one(self) -> None:
        parser = piper_cli.build_parser()
        args = parser.parse_args(["--execute-arm", "status"])

        with self.assertRaises(SystemExit) as raised:
            piper_cli._validate_global_args(parser, args)

        self.assertEqual(raised.exception.code, piper_cli.EXIT_USAGE)

    def test_motion_is_rejected_without_startup_unlock(self) -> None:
        args = cli_args(
            "jog-cartesian",
            execute_arm=False,
            confirm_arm_motion=False,
            x_mm=1.0,
            y_mm=None,
            z_mm=None,
            rx_deg=None,
            ry_deg=None,
            rz_deg=None,
        )

        with self.assertRaisesRegex(PermissionError, "locked"):
            piper_cli._run_action(self.session, args)

        self.assertFalse(self.piper.pose_commands)

    def test_cartesian_jog_reaches_feedback_target(self) -> None:
        args = cli_args(
            "jog-cartesian",
            x_mm=None,
            y_mm=None,
            z_mm=10.0,
            rx_deg=None,
            ry_deg=None,
            rz_deg=None,
        )

        with mock.patch.object(piper_cli, "confirm_action", return_value=True):
            result = piper_cli._run_action(self.session, args)

        self.assertEqual(result, piper_cli.EXIT_OK)
        self.assertEqual(self.piper.pose[2], 285_000)
        self.assertEqual(self.piper.motion_commands[-1][1], 0x00)

    def test_cartesian_step_limit_blocks_command_before_confirmation(self) -> None:
        args = cli_args(
            "jog-cartesian",
            x_mm=21.0,
            y_mm=None,
            z_mm=None,
            rx_deg=None,
            ry_deg=None,
            rz_deg=None,
        )

        with self.assertRaisesRegex(ValueError, "no larger than 20mm"):
            piper_cli._run_action(self.session, args)

        self.assertFalse(self.piper.pose_commands)

    def test_joint_jog_uses_feedback_and_reported_limits(self) -> None:
        args = cli_args("jog-joint", joint=6, delta_deg=3.0)

        with mock.patch.object(piper_cli, "confirm_action", return_value=True):
            piper_cli._run_action(self.session, args)

        self.assertEqual(self.piper.joints[5], 3_000)
        self.assertEqual(self.piper.motion_commands[-1][1], 0x01)

    def test_stop_needs_no_unlock_or_prompt(self) -> None:
        args = cli_args(
            "stop",
            execute_arm=False,
            confirm_arm_motion=False,
        )

        with mock.patch.object(piper_cli, "confirm_action") as confirmation:
            piper_cli._run_action(self.session, args)

        confirmation.assert_not_called()
        self.assertEqual(self.piper.stop_commands[-1], (1, 0, 0))

    def test_self_test_returns_pose_joint_and_gripper_to_start(self) -> None:
        start_pose = list(self.piper.pose)
        start_joints = list(self.piper.joints)
        start_gripper = self.piper.gripper_width
        args = cli_args("self-test")

        with mock.patch.object(piper_cli, "confirm_action", return_value=True):
            piper_cli._run_action(self.session, args)

        self.assertEqual(self.piper.pose, start_pose)
        self.assertEqual(self.piper.joints, start_joints)
        self.assertEqual(self.piper.gripper_width, start_gripper)
        self.assertGreaterEqual(len(self.piper.pose_commands), 2)
        self.assertGreaterEqual(len(self.piper.joint_commands), 2)
        self.assertGreaterEqual(len(self.piper.gripper_commands), 2)

    def test_motion_failure_requests_emergency_stop(self) -> None:
        args = cli_args(
            "jog-cartesian",
            x_mm=None,
            y_mm=None,
            z_mm=5.0,
            rx_deg=None,
            ry_deg=None,
            rz_deg=None,
        )

        with (
            mock.patch.object(piper_cli, "confirm_action", return_value=True),
            mock.patch.object(
                piper_cli,
                "move_pose_with_check",
                side_effect=TimeoutError("stuck"),
            ),
            self.assertRaisesRegex(TimeoutError, "stuck"),
        ):
            piper_cli._run_action(self.session, args)

        self.assertEqual(self.piper.stop_commands[-1], (1, 0, 0))

    def test_confirmation_rejects_noninteractive_stdin(self) -> None:
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=False),
            self.assertRaisesRegex(PermissionError, "interactive terminal"),
        ):
            piper_cli.confirm_action("move?")


if __name__ == "__main__":
    unittest.main()
