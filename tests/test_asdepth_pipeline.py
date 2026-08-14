from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

import asdepth_depth
import asdepth_pipeline
from asdepth_depth import CheckpointLoadReport
from camera_capture import CameraIntrinsics, CaptureResult


class PipelineSafetyTests(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[Path, Path, Path, Path]:
        rgb = directory / "color.png"
        depth = directory / "depth.png"
        depth_checkpoint = directory / "asdepth.ckpt"
        grasp_checkpoint = directory / "anygrasp.tar"
        cv2.imwrite(str(rgb), np.zeros((4, 6, 3), dtype=np.uint8))
        cv2.imwrite(str(depth), np.full((4, 6), 1000, dtype=np.uint16))
        depth_checkpoint.write_bytes(b"depth")
        grasp_checkpoint.write_bytes(b"grasp")
        return rgb, depth, depth_checkpoint, grasp_checkpoint

    def _args(
        self,
        directory: Path,
        *,
        execute_arm: bool,
    ) -> tuple[object, Path, Path]:
        rgb, depth, depth_checkpoint, grasp_checkpoint = self._fixture(directory)
        values = [
            "--depth-checkpoint",
            str(depth_checkpoint),
            "--depth-model",
            "defm_vit_l14_depth",
            "--grasp-checkpoint",
            str(grasp_checkpoint),
            "--rgb-image",
            str(rgb),
            "--depth-image",
            str(depth),
            "--save-dir",
            str(directory / "runs"),
        ]
        if execute_arm:
            values.append("--execute-arm")
        return (
            asdepth_pipeline.build_parser().parse_args(values),
            depth_checkpoint,
            grasp_checkpoint,
        )

    def _loaded(self, checkpoint: Path) -> object:
        return SimpleNamespace(
            checkpoint=CheckpointLoadReport(checkpoint, "state_dict", 790, ()),
            device=torch.device("cpu"),
            model_id="defm_vit_l14_depth",
        )

    def test_default_camera_backend_is_orbbec(self) -> None:
        args = asdepth_pipeline.build_parser().parse_args(
            [
                "--depth-checkpoint",
                "asdepth.ckpt",
                "--depth-model",
                "defm_stackconv_depth",
            ]
        )

        self.assertEqual(args.camera_backend, "orbbec")

    def test_default_run_does_not_import_or_execute_piper(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            args, depth_checkpoint, _ = self._args(directory, execute_arm=False)
            grasp_calls: list[tuple[object, object]] = []

            def fake_grasp(_save_dir, _cfg, *, rgb, depth):
                grasp_calls.append((rgb, depth))
                return np.eye(3), np.array([0.1, 0.2, 0.3]), 0.04

            with (
                mock.patch.object(
                    asdepth_pipeline,
                    "_load_anygrasp_function",
                    return_value=fake_grasp,
                ),
                mock.patch.object(
                    asdepth_pipeline,
                    "_load_arm_runner",
                    side_effect=AssertionError("Piper must not be imported"),
                ),
                mock.patch.object(
                    asdepth_depth,
                    "load_depth_model",
                    return_value=self._loaded(depth_checkpoint),
                ),
                mock.patch.object(
                    asdepth_depth,
                    "predict_depth",
                    return_value=np.full((4, 6), 1.5, dtype=np.float32),
                ),
            ):
                result = asdepth_pipeline.run(args)

            self.assertEqual(len(grasp_calls), 1)
            self.assertFalse(result["arm_executed"])
            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            self.assertFalse(metadata["arm_executed"])
            self.assertEqual(metadata["model_id"], "defm_vit_l14_depth")
            self.assertEqual(metadata["prediction_unit"], "meter")
            self.assertEqual(np.load(result["prediction"]).shape, (4, 6))

    def test_execute_arm_requires_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            args, depth_checkpoint, _ = self._args(directory, execute_arm=True)
            arm_runner = mock.Mock()

            with (
                mock.patch.object(
                    asdepth_pipeline,
                    "_load_anygrasp_function",
                    return_value=mock.Mock(
                        return_value=(np.eye(3), np.array([0.1, 0.2, 0.3]), 0.04)
                    ),
                ),
                mock.patch.object(asdepth_pipeline, "_load_arm_runner", return_value=arm_runner),
                mock.patch.object(
                    asdepth_depth,
                    "load_depth_model",
                    return_value=self._loaded(depth_checkpoint),
                ),
                mock.patch.object(
                    asdepth_depth,
                    "predict_depth",
                    return_value=np.full((4, 6), 1.5, dtype=np.float32),
                ),
            ):
                result = asdepth_pipeline.run(args)

            arm_runner.assert_called_once()
            self.assertTrue(result["arm_executed"])

    def test_online_capture_uses_camera_depth_scale_and_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            _, _, depth_checkpoint, grasp_checkpoint = self._fixture(directory)
            run_dir = directory / "capture"
            run_dir.mkdir()
            color_path = run_dir / "color.png"
            depth_path = run_dir / "depth.png"
            metadata_path = run_dir / "camera_metadata.json"
            cv2.imwrite(str(color_path), np.zeros((4, 6, 3), dtype=np.uint8))
            cv2.imwrite(str(depth_path), np.full((4, 6), 2000, dtype=np.uint16))
            metadata_path.write_text("{}", encoding="utf-8")
            intrinsics = CameraIntrinsics(500.0, 501.0, 3.0, 2.0, 6, 4)
            capture = CaptureResult(
                run_dir=run_dir,
                color_path=color_path,
                depth_path=depth_path,
                metadata_path=metadata_path,
                raw_units_per_meter=2000.0,
                intrinsics=intrinsics,
                backend="orbbec",
                camera_name="DaBai DC1",
                serial_number="test-serial",
            )
            args = asdepth_pipeline.build_parser().parse_args(
                [
                    "--depth-checkpoint",
                    str(depth_checkpoint),
                    "--depth-model",
                    "defm_stackconv_depth",
                    "--grasp-checkpoint",
                    str(grasp_checkpoint),
                    "--save-dir",
                    str(directory / "runs"),
                ]
            )
            observed: dict[str, object] = {}

            def fake_predict(_loaded, _color, _depth, **kwargs):
                observed["depth_scale"] = kwargs["depth_scale"]
                return np.full((4, 6), 1.0, dtype=np.float32)

            def fake_grasp(_save_dir, cfg, *, rgb, depth):
                observed["intrinsics"] = cfg.camera_intrinsics
                return np.eye(3), np.array([0.1, 0.2, 0.3]), 0.04

            with (
                mock.patch.object(asdepth_pipeline, "_capture_camera", return_value=capture),
                mock.patch.object(
                    asdepth_pipeline,
                    "_load_anygrasp_function",
                    return_value=fake_grasp,
                ),
                mock.patch.object(
                    asdepth_depth,
                    "load_depth_model",
                    return_value=self._loaded(depth_checkpoint),
                ),
                mock.patch.object(asdepth_depth, "predict_depth", side_effect=fake_predict),
            ):
                result = asdepth_pipeline.run(args)

            self.assertEqual(observed["depth_scale"], 2000.0)
            self.assertEqual(
                observed["intrinsics"],
                {"fx": 500.0, "fy": 501.0, "cx": 3.0, "cy": 2.0},
            )
            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["camera_backend"], "orbbec")
            self.assertEqual(metadata["camera_serial_number"], "test-serial")

    def test_main_converts_expected_type_error_to_exit_code_two(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(asdepth_pipeline, "run", side_effect=TypeError("invalid checkpoint")),
            mock.patch("sys.stderr", stderr),
        ):
            exit_code = asdepth_pipeline.main(
                [
                    "--depth-checkpoint",
                    "model.ckpt",
                    "--depth-model",
                    "defm_vit_l14_depth",
                ]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("invalid checkpoint", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
