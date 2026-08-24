"""Tests that CLI artifact defaults follow the requested model."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_bench import benchmark, input_artifact, onnx_export, onnx_runner, openvino_runner


class ModelSpecificCliDefaultsTests(unittest.TestCase):
    def test_onnx_commands_default_to_the_requested_model_artifact(self) -> None:
        with patch.object(sys, "argv", ["onnx_export", "--model", "mobilenet_v3_large"]):
            self.assertEqual(
                onnx_export._parse_arguments().output,
                Path("artifacts/mobilenet_v3_large.onnx"),
            )
        with patch.object(sys, "argv", ["onnx_runner", "--model", "mobilenet_v3_large"]):
            self.assertEqual(
                onnx_runner._parse_arguments().model_path,
                Path("artifacts/mobilenet_v3_large.onnx"),
            )
        with patch.object(sys, "argv", ["openvino_runner", "--model", "mobilenet_v3_large"]):
            self.assertEqual(
                openvino_runner._parse_arguments().model_path,
                Path("artifacts/mobilenet_v3_large.onnx"),
            )
        with patch.object(
            sys,
            "argv",
            [
                "benchmark",
                "--engine",
                "onnxruntime",
                "--model",
                "mobilenet_v3_large",
                "--device",
                "cuda:0",
            ],
        ):
            arguments = benchmark._parse_arguments()
            self.assertEqual(
                arguments.model_path,
                Path("artifacts/mobilenet_v3_large.onnx"),
            )
            self.assertEqual(
                arguments.output_dir,
                Path("results/mobilenet_v3_large/cuda_0"),
            )

    def test_native_input_defaults_include_model_and_seeds(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "input_artifact",
                "--model",
                "mobilenet_v3_large",
                "--input-seed",
                "11",
                "--model-seed",
                "12",
            ],
        ):
            arguments = input_artifact._parse_arguments()

        self.assertEqual(
            arguments.output,
            Path("artifacts/inputs/mobilenet_v3_large_seed11_f32_nchw.bin"),
        )
        self.assertEqual(
            arguments.reference_output,
            Path("artifacts/reference_outputs/mobilenet_v3_large_seed12_input11_f32_logits.bin"),
        )


if __name__ == "__main__":
    unittest.main()
