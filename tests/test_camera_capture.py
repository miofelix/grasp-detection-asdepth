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

import camera_capture
from camera_capture import CameraIntrinsics, CapturedFrame


class CameraCaptureTests(unittest.TestCase):
    def _frame(self) -> CapturedFrame:
        return CapturedFrame(
            color_bgr=np.zeros((4, 6, 3), dtype=np.uint8),
            raw_depth=np.full((4, 6), 1000, dtype=np.uint16),
            raw_units_per_meter=1000.0,
            intrinsics=CameraIntrinsics(500.0, 501.0, 3.0, 2.0, 6, 4),
            backend="orbbec",
            camera_name="DaBai DC1",
            serial_number="serial",
        )

    def test_write_capture_persists_images_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            result = camera_capture._write_capture(Path(value), self._frame())

            self.assertEqual(cv2.imread(str(result.color_path)).shape, (4, 6, 3))
            self.assertEqual(
                cv2.imread(str(result.depth_path), cv2.IMREAD_UNCHANGED).dtype,
                np.uint16,
            )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["backend"], "orbbec")
            self.assertEqual(metadata["raw_units_per_meter"], 1000.0)
            self.assertEqual(metadata["intrinsics"]["fx"], 500.0)

    def test_camera_cli_reports_capture_error(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                camera_capture,
                "capture_one_frame",
                side_effect=RuntimeError("camera unavailable"),
            ),
            mock.patch("sys.stderr", stderr),
        ):
            exit_code = camera_capture.main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("camera unavailable", stderr.getvalue())

    def test_validate_frame_rejects_unaligned_depth(self) -> None:
        frame = self._frame()
        mismatched = CapturedFrame(
            color_bgr=frame.color_bgr,
            raw_depth=np.zeros((3, 6), dtype=np.uint16),
            raw_units_per_meter=frame.raw_units_per_meter,
            intrinsics=frame.intrinsics,
            backend=frame.backend,
            camera_name=frame.camera_name,
            serial_number=frame.serial_number,
        )

        with self.assertRaisesRegex(ValueError, "spatial mismatch"):
            camera_capture._validate_frame(mismatched)

    def test_orbbec_bgr_frames_keep_channel_order(self) -> None:
        sdk = SimpleNamespace(
            OBFormat=SimpleNamespace(
                RGB=1,
                BGR=2,
                YUYV=3,
                YUY2=4,
                UYVY=5,
                MJPG=6,
                I420=7,
                NV12=8,
                NV21=9,
            )
        )
        pixels = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        frame = SimpleNamespace(
            get_format=lambda: sdk.OBFormat.BGR,
            get_width=lambda: 2,
            get_height=lambda: 1,
            get_data=lambda: pixels,
        )

        converted = camera_capture._orbbec_color_to_bgr(frame, sdk)

        np.testing.assert_array_equal(converted, pixels)

    def test_orbbec_capture_converts_depth_scale_and_uses_rgb_intrinsics(self) -> None:
        color = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        depth = np.array([[1000, 2000]], dtype=np.uint16)
        color_frame = SimpleNamespace(
            get_format=lambda: 2,
            get_width=lambda: 2,
            get_height=lambda: 1,
            get_data=lambda: color,
        )
        depth_frame = SimpleNamespace(
            get_width=lambda: 2,
            get_height=lambda: 1,
            get_data=lambda: depth,
            get_depth_scale=lambda: 0.5,
        )
        frames = SimpleNamespace(
            get_color_frame=lambda: color_frame,
            get_depth_frame=lambda: depth_frame,
        )
        intrinsic = SimpleNamespace(
            fx=500.0,
            fy=501.0,
            cx=1.0,
            cy=0.5,
            width=2,
            height=1,
        )
        profile = SimpleNamespace(get_intrinsic=lambda: intrinsic)
        profile_list = SimpleNamespace(get_default_video_stream_profile=lambda: profile)
        device_info = SimpleNamespace(
            get_name=lambda: "DaBai DC1",
            get_serial_number=lambda: "dc1-serial",
        )

        class FakePipeline:
            def get_stream_profile_list(self, _sensor_type):
                return profile_list

            def enable_frame_sync(self):
                return None

            def start(self, _config):
                return None

            def get_device(self):
                return SimpleNamespace(get_device_info=lambda: device_info)

            def wait_for_frames(self, _timeout_ms):
                return frames

            def stop(self):
                return None

        class FakeConfig:
            def enable_stream(self, _profile):
                return None

            def set_align_mode(self, _mode):
                return None

        sdk = SimpleNamespace(
            Pipeline=FakePipeline,
            Config=FakeConfig,
            OBSensorType=SimpleNamespace(COLOR_SENSOR=1, DEPTH_SENSOR=2),
            OBFormat=SimpleNamespace(
                RGB=1,
                BGR=2,
                YUYV=3,
                YUY2=4,
                UYVY=5,
                MJPG=6,
                I420=7,
                NV12=8,
                NV21=9,
            ),
        )

        captured = camera_capture._capture_orbbec_with_align_mode(
            sdk,
            align_mode=1,
            warmup_frames=0,
            timeout_s=1.0,
        )

        self.assertEqual(captured.raw_units_per_meter, 2000.0)
        self.assertEqual(captured.intrinsics.fx, 500.0)
        self.assertEqual(captured.camera_name, "DaBai DC1")
        np.testing.assert_array_equal(captured.raw_depth, depth)


if __name__ == "__main__":
    unittest.main()
