"""Tests for the deterministic ONNX export contract."""

import tempfile
import unittest
from pathlib import Path

import onnx

from inference_bench.onnx_export import (
    DEFAULT_OPSET_VERSION,
    INPUT_NAME,
    OUTPUT_NAME,
    export_onnx,
)


class OnnxExportTests(unittest.TestCase):
    def test_resnet50_export_is_structurally_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "resnet50.onnx"
            result = export_onnx("resnet50", output_path)

            exported_model = onnx.load(str(output_path))
            onnx.checker.check_model(exported_model)

        self.assertEqual(result.model_name, "resnet50")
        self.assertEqual(result.input_shape, (1, 3, 224, 224))
        self.assertEqual(result.opset_version, DEFAULT_OPSET_VERSION)
        self.assertGreater(result.artifact_size_bytes, 0)
        self.assertEqual(exported_model.graph.input[0].name, INPUT_NAME)
        self.assertEqual(exported_model.graph.output[0].name, OUTPUT_NAME)


if __name__ == "__main__":
    unittest.main()
