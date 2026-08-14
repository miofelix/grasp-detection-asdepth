from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    def setUp(self) -> None:
        self.previous_gsnet = anygrasp_runtime.sys.modules.pop("gsnet", None)
        anygrasp_runtime._GSNET_MODULE = None
        anygrasp_runtime._GSNET_BINARY = None

    def tearDown(self) -> None:
        anygrasp_runtime.sys.modules.pop("gsnet", None)
        if self.previous_gsnet is not None:
            anygrasp_runtime.sys.modules["gsnet"] = self.previous_gsnet
        anygrasp_runtime._GSNET_MODULE = None
        anygrasp_runtime._GSNET_BINARY = None

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

    def test_load_gsnet_module_reuses_module_after_sdk_rewrites_file(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            versions = root / "gsnet_versions"
            versions.mkdir()
            binary = versions / "gsnet.cpython-310-x86_64-linux-gnu.so"
            binary.write_bytes(b"binary")
            module = ModuleType("gsnet")
            module.__file__ = str(binary)

            class FakeLoader:
                @staticmethod
                def exec_module(loaded: ModuleType) -> None:
                    loaded.__file__ = str(versions / "gsnet" / "__init__.py")

            spec = SimpleNamespace(loader=FakeLoader())
            with (
                mock.patch.object(
                    anygrasp_runtime,
                    "matching_gsnet_path",
                    return_value=binary,
                ),
                mock.patch.object(
                    anygrasp_runtime.importlib.util,
                    "spec_from_file_location",
                    return_value=spec,
                ),
                mock.patch.object(
                    anygrasp_runtime.importlib.util,
                    "module_from_spec",
                    return_value=module,
                ),
            ):
                first = anygrasp_runtime.load_gsnet_module(root)
                second = anygrasp_runtime.load_gsnet_module(root)

            self.assertIs(first, module)
            self.assertIs(second, module)
            self.assertEqual(
                module.__anygrasp_binary_path__,
                str(binary.resolve()),
            )

    def test_load_gsnet_module_still_rejects_external_name_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            binary = root / "gsnet.cpython-310-x86_64-linux-gnu.so"
            binary.write_bytes(b"binary")
            conflicting = ModuleType("gsnet")
            conflicting.__file__ = "/tmp/unrelated/gsnet/__init__.py"
            anygrasp_runtime.sys.modules["gsnet"] = conflicting

            with (
                mock.patch.object(
                    anygrasp_runtime,
                    "matching_gsnet_path",
                    return_value=binary,
                ),
                self.assertRaisesRegex(RuntimeError, "already loaded"),
            ):
                anygrasp_runtime.load_gsnet_module(root)

    def test_create_detector_uses_project_license_and_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "gsnet_versions").mkdir()
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
            self.assertEqual(
                (root / "gsnet_versions" / "license").resolve(),
                license_dir.resolve(),
            )

    def test_create_detector_requires_new_license_layout(self) -> None:
        with (
            tempfile.TemporaryDirectory() as value,
            self.assertRaisesRegex(FileNotFoundError, "licenseCfg.json"),
        ):
            anygrasp_runtime.create_detector(SimpleNamespace(), value)

    def test_prepare_binary_license_dir_rejects_conflicting_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            versions = root / "gsnet_versions"
            versions.mkdir()
            license_dir = root / "license"
            license_dir.mkdir()
            (license_dir / "licenseCfg.json").write_text("{}", encoding="utf-8")
            other = root / "other-license"
            other.mkdir()
            (versions / "license").symlink_to(other, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "points to"):
                anygrasp_runtime.prepare_binary_license_dir(root)

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
