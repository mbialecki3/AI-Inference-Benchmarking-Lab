"""Tests that CLI artifact defaults follow the requested model."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_bench import benchmark, input_artifact, onnx_export, onnx_runner, openvino_runner, yolo_benchmark


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

    def test_efficientnet_b0_defaults_to_its_own_artifacts(self) -> None:
        with patch.object(sys, "argv", ["onnx_export", "--model", "efficientnet_b0"]):
            self.assertEqual(
                onnx_export._parse_arguments().output,
                Path("artifacts/efficientnet_b0.onnx"),
            )
        with patch.object(sys, "argv", ["input_artifact", "--model", "efficientnet_b0"]):
            arguments = input_artifact._parse_arguments()

        self.assertEqual(
            arguments.output,
            Path("artifacts/inputs/efficientnet_b0_seed69420_f32_nchw.bin"),
        )
        self.assertEqual(
            arguments.reference_output,
            Path("artifacts/reference_outputs/efficientnet_b0_seed67_input69420_f32_logits.bin"),
        )

    def test_yolo_openvino_uses_detection_artifact_and_cpu_result_scope(self) -> None:
        with patch.object(sys, "argv", ["yolo_benchmark", "--engine", "openvino"]):
            arguments = yolo_benchmark._parse_arguments()

        self.assertEqual(arguments.model_path, Path("artifacts/yolo11n.onnx"))
        self.assertEqual(arguments.output_dir, Path("results/yolo11n/cpu"))

    def test_yolo_native_artifact_defaults_use_raw_reference_filename(self) -> None:
        with patch.object(sys, "argv", ["input_artifact", "--model", "yolo11n"]):
            arguments = input_artifact._parse_arguments()

        self.assertEqual(arguments.output, Path("artifacts/inputs/yolo11n_seed69420_f32_nchw.bin"))
        self.assertEqual(
            arguments.reference_output,
            Path("artifacts/reference_outputs/yolo11n_input69420_f32_raw.bin"),
        )


if __name__ == "__main__":
    unittest.main()
