from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import grasp_piper

OBSERVED_ROTATION = np.array(
    [
        [-0.11492483, -0.95416409, 0.27633876],
        [0.36639848, -0.29928386, -0.88101155],
        [0.92333335, 0.0, 0.38399944],
    ],
    dtype=np.float64,
)
OBSERVED_TRANSLATION = np.array([0.01056872, -0.01207399, 0.30448267])


class FakePosePiper:
    def __init__(self) -> None:
        self.target = (0, 0, 0, 0, 0, 0)

    def EndPoseCtrl(self, *target: int) -> None:
        self.target = target

    def GetArmEndPoseMsgs(self) -> object:
        return SimpleNamespace(
            end_pose=SimpleNamespace(
                X_axis=self.target[0],
                Y_axis=self.target[1],
                Z_axis=self.target[2],
                RX_axis=self.target[3],
                RY_axis=self.target[4],
                RZ_axis=self.target[5],
            )
        )


class FakeStuckPiper(FakePosePiper):
    def GetArmEndPoseMsgs(self) -> object:
        return SimpleNamespace(
            end_pose=SimpleNamespace(
                X_axis=0,
                Y_axis=0,
                Z_axis=0,
                RX_axis=0,
                RY_axis=0,
                RZ_axis=0,
            )
        )


class FakeGripperPiper:
    def __init__(self) -> None:
        self.target = 0

    def GripperCtrl(self, target: int, _effort: int, _code: int, _zero: int) -> None:
        self.target = target

    def GetArmGripperMsgs(self) -> object:
        return SimpleNamespace(gripper_state=SimpleNamespace(grippers_angle=self.target))


class PiperSafetyTests(unittest.TestCase):
    def test_device_config_defaults_to_left_arm_can2(self) -> None:
        camera_to_base, safety, device = grasp_piper.load_device_config()

        self.assertEqual(camera_to_base.shape, (4, 4))
        self.assertEqual(safety.can_name, "can2")
        self.assertEqual(device["arm_side"], "left")
        self.assertEqual(device["can_interface"], "can2")
        self.assertEqual(
            Path(device["config_path"]),
            Path(grasp_piper.__file__).resolve().parent / "config/piper_device.json",
        )

    def test_right_arm_is_declared_but_rejected_until_calibrated(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "right.*disabled"):
            grasp_piper.load_device_config(arm_side="right")

    def test_can_interface_override_is_reported(self) -> None:
        _, safety, device = grasp_piper.load_device_config(can_name="can9")

        self.assertEqual(safety.can_name, "can9")
        self.assertEqual(device["can_interface"], "can9")

    def test_observed_pose_normalizes_final_mount_angle(self) -> None:
        safety = grasp_piper.ArmSafetyConfig(
            gripper_max_width_m=0.10,
            pregrasp_clearance_m=0.10,
            max_reach_m=0.70,
            max_z_m=0.70,
        )

        plan = grasp_piper.build_motion_plan(
            OBSERVED_ROTATION,
            OBSERVED_TRANSLATION,
            0.10,
            safety=safety,
        )

        self.assertAlmostEqual(plan.grasp_pose.ry_deg, -33.2085933, places=5)
        self.assertLessEqual(abs(plan.grasp_pose.ry_deg), 90.0)
        self.assertEqual(plan.gripper_grasp_width_m, 0.10)

    def test_default_small_gripper_rejects_ten_centimeter_grasp(self) -> None:
        with self.assertRaisesRegex(ValueError, "gripper limit"):
            grasp_piper.build_motion_plan(
                OBSERVED_ROTATION,
                OBSERVED_TRANSLATION,
                0.10,
            )

    def test_pregrasp_outside_reach_is_rejected_before_hardware(self) -> None:
        safety = grasp_piper.ArmSafetyConfig(gripper_max_width_m=0.10)

        with self.assertRaisesRegex(ValueError, "pregrasp pose exceeds maximum reach"):
            grasp_piper.build_motion_plan(
                OBSERVED_ROTATION,
                OBSERVED_TRANSLATION,
                0.10,
                safety=safety,
            )

    def test_move_check_uses_documented_feedback_fields(self) -> None:
        piper = FakePosePiper()
        pose = grasp_piper.ArmPose(0.20, -0.10, 0.30, 10.0, -20.0, 30.0)

        grasp_piper.move_with_check(
            piper,
            pose,
            timeout_s=0.1,
            position_tolerance_m=0.001,
            angle_tolerance_deg=1.0,
        )

        self.assertEqual(piper.target, pose.command())

    def test_move_check_raises_instead_of_continuing_after_timeout(self) -> None:
        piper = FakeStuckPiper()
        pose = grasp_piper.ArmPose(0.20, -0.10, 0.30, 10.0, -20.0, 30.0)

        with (
            mock.patch.object(grasp_piper.time, "monotonic", side_effect=[0.0, 0.0, 0.2]),
            mock.patch.object(grasp_piper.time, "sleep"),
            self.assertRaises(TimeoutError),
        ):
            grasp_piper.move_with_check(
                piper,
                pose,
                timeout_s=0.1,
                position_tolerance_m=0.001,
                angle_tolerance_deg=1.0,
            )

    def test_gripper_uses_full_anygrasp_width(self) -> None:
        piper = FakeGripperPiper()

        grasp_piper._set_gripper_with_check(
            piper,
            0.04,
            timeout_s=0.1,
            tolerance_m=0.001,
        )

        self.assertEqual(piper.target, 40_000)

    def test_run_pipeline_defaults_to_dry_run(self) -> None:
        result = grasp_piper.run_pipeline(
            OBSERVED_ROTATION,
            OBSERVED_TRANSLATION,
            0.10,
            gripper_max_width_m=0.10,
            pregrasp_clearance_m=0.10,
            max_reach_m=0.70,
            max_z_m=0.70,
        )

        self.assertFalse(result["executed"])
        self.assertAlmostEqual(result["plan"]["gripper_grasp_width_m"], 0.10)
        self.assertEqual(result["device"]["arm_side"], "left")
        self.assertEqual(result["device"]["can_interface"], "can2")


if __name__ == "__main__":
    unittest.main()
