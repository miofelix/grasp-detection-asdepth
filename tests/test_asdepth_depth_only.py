from __future__ import annotations

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
import asdepth_depth_only
from asdepth_depth import CheckpointLoadReport


class DepthOnlyPipelineTests(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[Path, Path, Path]:
        rgb = directory / "color.png"
        depth = directory / "depth.png"
        checkpoint = directory / "asdepth.ckpt"
        cv2.imwrite(str(rgb), np.zeros((4, 6, 3), dtype=np.uint8))
        cv2.imwrite(str(depth), np.full((4, 6), 1000, dtype=np.uint16))
        checkpoint.write_bytes(b"depth")
        return rgb, depth, checkpoint

    def test_offline_depth_only_writes_prediction_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            rgb, depth, checkpoint = self._fixture(directory)
            args = asdepth_depth_only.build_parser().parse_args(
                [
                    "--depth-checkpoint",
                    str(checkpoint),
                    "--depth-model",
                    "defm_vit_l14_depth",
                    "--rgb-image",
                    str(rgb),
                    "--depth-image",
                    str(depth),
                    "--save-dir",
                    str(directory / "runs"),
                    "--device",
                    "cpu",
                ]
            )
            loaded = SimpleNamespace(
                checkpoint=CheckpointLoadReport(checkpoint, "state_dict", 790, ()),
                device=torch.device("cpu"),
                model_id="defm_vit_l14_depth",
            )
            with (
                mock.patch.object(asdepth_depth, "load_depth_model", return_value=loaded),
                mock.patch.object(
                    asdepth_depth,
                    "predict_depth",
                    return_value=np.full((4, 6), 1.5, dtype=np.float32),
                ),
            ):
                result = asdepth_depth_only.run(args)

            prediction = np.load(result["prediction"])
            self.assertEqual(prediction.shape, (4, 6))
            self.assertTrue(np.all(prediction == 1.5))
            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["mode"], "asdepth_depth_only")
            self.assertEqual(metadata["model_id"], "defm_vit_l14_depth")
            self.assertEqual(metadata["prediction_unit"], "meter")
            self.assertEqual(metadata["depth_checkpoint_tensor_count"], 790)

    def test_depth_only_main_reports_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            rgb = directory / "color.png"
            depth = directory / "depth.png"
            cv2.imwrite(str(rgb), np.zeros((4, 6, 3), dtype=np.uint8))
            cv2.imwrite(str(depth), np.full((4, 6), 1000, dtype=np.uint16))
            exit_code = asdepth_depth_only.main(
                [
                    "--depth-checkpoint",
                    str(directory / "missing.ckpt"),
                    "--depth-model",
                    "defm_vit_l14_depth",
                    "--rgb-image",
                    str(rgb),
                    "--depth-image",
                    str(depth),
                ]
            )
            self.assertEqual(exit_code, 2)

    def test_clear_model_cache_skips_unavailable_mps_backend(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            mock.patch.object(torch.mps, "empty_cache") as empty_cache,
        ):
            asdepth_depth_only._clear_model_cache()

        empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
