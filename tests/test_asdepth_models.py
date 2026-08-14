from __future__ import annotations

import io
import json
import unittest
from unittest import mock

import asdepth_models


class ModelCatalogCliTests(unittest.TestCase):
    def test_json_output_contains_catalog_models(self) -> None:
        stdout = io.StringIO()
        with mock.patch("sys.stdout", stdout):
            exit_code = asdepth_models.main(["--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(payload), 20)
        self.assertIn("defm_stackconv_depth", {item["model_id"] for item in payload})


if __name__ == "__main__":
    unittest.main()
