from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import anygrasp_runtime


class FakeGraspGroup:
    def __init__(self, size: int = 1) -> None:
        self.items = list(range(size))
        self.nms_called = False
        self.sort_called = False

    def __len__(self) -> int:
        return len(self.items)

    def nms(self) -> FakeGraspGroup:
        self.nms_called = True
        return self

    def sort_by_score(self) -> FakeGraspGroup:
        self.sort_called = True
        return self


class AnyGraspRuntimeTests(unittest.TestCase):
    def test_matching_binary_uses_current_extension_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            versions = root / "gsnet_versions"
            versions.mkdir()
            expected = versions / "gsnet.cpython-312-x86_64-linux-gnu.so"
            expected.write_bytes(b"binary")

            with (
                mock.patch.object(anygrasp_runtime.platform, "system", return_value="Linux"),
                mock.patch.object(anygrasp_runtime.platform, "machine", return_value="x86_64"),
                mock.patch.object(
                    anygrasp_runtime.sysconfig,
                    "get_config_var",
                    return_value=".cpython-312-x86_64-linux-gnu.so",
                ),
            ):
                actual = anygrasp_runtime.matching_gsnet_path(root)

            self.assertEqual(actual, expected.resolve())

    def test_matching_binary_rejects_non_linux_platform(self) -> None:
        with (
            mock.patch.object(anygrasp_runtime.platform, "system", return_value="Darwin"),
            self.assertRaisesRegex(RuntimeError, "Linux x86-64"),
        ):
            anygrasp_runtime.matching_gsnet_path()

    def test_create_detector_uses_project_license_and_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            license_dir = root / "license"
            license_dir.mkdir()
            (license_dir / "licenseCfg.json").write_text("{}", encoding="utf-8")
            calls: list[tuple[Path, object]] = []

            class FakeModule:
                @staticmethod
                def create_detector(config: object) -> object:
                    calls.append((Path.cwd(), config))
                    return object()

            config = SimpleNamespace(checkpoint_path="model.tar")
            with mock.patch.object(
                anygrasp_runtime,
                "load_gsnet_module",
                return_value=FakeModule(),
            ):
                detector = anygrasp_runtime.create_detector(config, root)

            self.assertIsNotNone(detector)
            self.assertEqual(calls, [(root.resolve(), config)])

    def test_create_detector_requires_new_license_layout(self) -> None:
        with (
            tempfile.TemporaryDirectory() as value,
            self.assertRaisesRegex(FileNotFoundError, "licenseCfg.json"),
        ):
            anygrasp_runtime.create_detector(SimpleNamespace(), value)

    def test_predict_grasps_maps_workspace_and_top_down_options(self) -> None:
        group = FakeGraspGroup()
        calls: list[tuple[np.ndarray, dict[str, object]]] = []

        class FakeDetector:
            @staticmethod
            def get_grasp(
                points: np.ndarray,
                options: dict[str, object],
            ) -> FakeGraspGroup:
                calls.append((points, options))
                return group

        points = np.array(
            [[0.0, 0.0, 0.5], [0.3, 0.0, 0.5]],
            dtype=np.float32,
        )
        result = anygrasp_runtime.predict_grasps(
            FakeDetector(),
            points,
            [-0.1, 0.1, -0.1, 0.1, 0.2, 1.0],
            top_down_grasp=True,
        )

        self.assertIs(result, group)
        self.assertTrue(group.nms_called)
        self.assertTrue(group.sort_called)
        options = calls[0][1]
        np.testing.assert_array_equal(options["region_steering"], [True, False])
        self.assertEqual(options["approach_steering"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(float(options["approach_thresh"]), np.pi / 6)

    def test_predict_grasps_returns_none_for_empty_sdk_result(self) -> None:
        detector = SimpleNamespace(get_grasp=mock.Mock(return_value=None))
        points = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)

        result = anygrasp_runtime.predict_grasps(
            detector,
            points,
            [-1.0, 1.0, -1.0, 1.0, 0.0, 1.0],
            top_down_grasp=False,
        )

        self.assertIsNone(result)
        self.assertIsNone(detector.get_grasp.call_args.args[1]["approach_steering"])


if __name__ == "__main__":
    unittest.main()
