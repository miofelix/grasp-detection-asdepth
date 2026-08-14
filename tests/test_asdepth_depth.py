from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

import asdepth_depth.api as depth_api
from asdepth_depth import (
    CheckpointLoadReport,
    LoadedDepthModel,
    colorize_metric_depth,
    depth_visualization_range,
    load_checkpoint,
    predict_depth,
    prepare_rgbd_input,
    save_depth_visualizations,
)
from asdepth_depth.models import DeFMRGBDDepth, DeFMStackConvRGBDDepth
from asdepth_depth.preprocess import metric_depth_from_raw, output_size


class _ToyDepthModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 3] + 0.25


class PreprocessTests(unittest.TestCase):
    def test_d435_shape_uses_aspect_ratio_and_patch_alignment(self) -> None:
        self.assertEqual(
            output_size(640, 480, input_size=518, resize_method="lower_bound"),
            (686, 518),
        )

    def test_metric_depth_filters_invalid_values(self) -> None:
        raw = np.array([[0.0, 1000.0, 9999.0, 10000.0, np.nan, -1.0]], dtype=np.float32)
        metric = metric_depth_from_raw(raw, depth_scale=1000.0, max_depth_m=10.0)
        np.testing.assert_allclose(metric, [[0.0, 1.0, 9.999, 0.0, 0.0, 0.0]])

    def test_prepare_rgbd_input_returns_four_channel_float_tensor(self) -> None:
        color = np.zeros((4, 6, 3), dtype=np.uint8)
        color[..., 2] = 255
        raw = np.full((4, 6), 1000, dtype=np.uint16)
        raw[0, 0] = 0

        inputs = prepare_rgbd_input(color, raw, input_size=14)

        self.assertEqual(tuple(inputs.shape), (1, 4, 14, 28))
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(float(inputs[:, 3].min()), 0.0)
        self.assertEqual(float(inputs[:, 3].max()), 1.0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_extracts_state_dict_and_strips_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save({"state_dict": {"module.layer.weight": torch.ones(2, 3)}}, path)

            state_dict, report = load_checkpoint(path)

        self.assertEqual(tuple(state_dict), ("layer.weight",))
        self.assertEqual(report.source_key, "state_dict")
        self.assertEqual(report.tensor_count, 1)
        self.assertEqual(report.stripped_prefixes, ("module.",))

    def test_migrated_model_keeps_expected_state_dict_contract(self) -> None:
        original_linspace = torch.linspace

        def cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return original_linspace(*args, **kwargs)

        with (
            mock.patch("torch.linspace", side_effect=cpu_linspace),
            torch.device("meta"),
        ):
            model = DeFMStackConvRGBDDepth()
        state_dict = model.state_dict()

        self.assertEqual(len(state_dict), 804)
        self.assertEqual(
            tuple(state_dict["pretrained.patch_embed.proj.weight"].shape),
            (1024, 3, 14, 14),
        )
        self.assertEqual(
            tuple(state_dict["depth_pretrained.patch_embed.proj.weight"].shape),
            (1024, 3, 14, 14),
        )
        self.assertIn("depth_pretrained.blocks.0.0.mlp.w12.weight", state_dict)
        self.assertIn("depth_head_rgbd.output_blocks.4.weight", state_dict)

    def test_defm_vit_l14_model_keeps_expected_state_dict_contract(self) -> None:
        original_linspace = torch.linspace

        def cpu_linspace(*args, **kwargs):
            kwargs.setdefault("device", "cpu")
            return original_linspace(*args, **kwargs)

        with (
            mock.patch("torch.linspace", side_effect=cpu_linspace),
            torch.device("meta"),
        ):
            model = DeFMRGBDDepth()
        state_dict = model.state_dict()

        self.assertEqual(len(state_dict), 790)
        self.assertEqual(
            tuple(state_dict["depth_head_rgbd.projects.0.weight"].shape),
            (256, 2048, 1, 1),
        )
        self.assertEqual(
            tuple(state_dict["depth_head_rgbd.scratch.output_conv2.2.weight"].shape),
            (1, 32, 1, 1),
        )
        self.assertNotIn("neck.stem_0.weight", state_dict)

    def test_load_depth_model_uses_explicit_model_id(self) -> None:
        report = CheckpointLoadReport(Path("model.ckpt"), "state_dict", 1, ())
        toy_model = torch.nn.Linear(1, 1, bias=False)
        with (
            mock.patch.object(
                depth_api,
                "load_checkpoint",
                return_value=({"weight": torch.ones(1, 1)}, report),
            ),
            mock.patch.object(depth_api, "_create_model", return_value=toy_model) as create_model,
        ):
            loaded = depth_api.load_depth_model(
                "model.ckpt",
                model_id="defm_vit_l14_depth",
                device="cpu",
            )

        create_model.assert_called_once_with("defm_vit_l14_depth")
        self.assertEqual(loaded.model_id, "defm_vit_l14_depth")

    def test_load_depth_model_rejects_automatic_model_selection(self) -> None:
        with (
            mock.patch.object(
                depth_api,
                "load_checkpoint",
                side_effect=AssertionError("checkpoint must not be inspected"),
            ),
            self.assertRaisesRegex(ValueError, "unsupported depth model"),
        ):
            depth_api.load_depth_model("model.ckpt", model_id="auto", device="cpu")


class VisualizationTests(unittest.TestCase):
    def test_shared_range_uses_joint_valid_depth_values(self) -> None:
        raw = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
        prediction = np.array([[1.5, 2.5], [20.0, -1.0]], dtype=np.float32)

        minimum, maximum = depth_visualization_range(
            raw,
            prediction,
            max_depth_m=10.0,
            percentile_min=0.0,
            percentile_max=100.0,
        )

        self.assertEqual(minimum, 1.0)
        self.assertEqual(maximum, 2.5)

    def test_colorize_metric_depth_keeps_invalid_pixels_black(self) -> None:
        depth = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)

        visualization = colorize_metric_depth(depth, min_depth_m=1.0, max_depth_m=2.0)

        self.assertEqual(visualization.shape, (2, 2, 3))
        np.testing.assert_array_equal(visualization[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(visualization[1, 1], [0, 0, 0])
        self.assertTrue(np.any(visualization[0, 1] != 0))

    def test_save_depth_visualizations_writes_raw_and_prediction_pngs(self) -> None:
        raw = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
        prediction = np.array([[np.nan, 1.5], [2.5, 3.5]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_depth_visualizations(
                directory,
                raw,
                prediction,
                depth_scale=1000.0,
                max_depth_m=10.0,
            )
            raw_visualization = cv2.imread(str(artifacts.raw_depth_path), cv2.IMREAD_COLOR)
            prediction_visualization = cv2.imread(str(artifacts.prediction_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(raw_visualization)
        self.assertIsNotNone(prediction_visualization)
        self.assertEqual(raw_visualization.shape, (2, 2, 3))
        self.assertEqual(prediction_visualization.shape, (2, 2, 3))
        np.testing.assert_array_equal(raw_visualization[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(prediction_visualization[0, 0], [0, 0, 0])


class InferenceTests(unittest.TestCase):
    def test_predict_depth_restores_camera_shape(self) -> None:
        report = CheckpointLoadReport(Path("model.ckpt"), "root", 1, ())
        loaded = LoadedDepthModel(
            _ToyDepthModel(),
            torch.device("cpu"),
            report,
            "defm_vit_l14_depth",
        )
        color = np.zeros((4, 6, 3), dtype=np.uint8)
        raw = np.full((4, 6), 1000, dtype=np.uint16)

        prediction = predict_depth(loaded, color, raw, input_size=14)

        self.assertEqual(prediction.shape, (4, 6))
        self.assertEqual(prediction.dtype, np.float32)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
