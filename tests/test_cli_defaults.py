"""Tests that CLI artifact defaults follow the requested model."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_bench import (
    benchmark, detection_benchmark, detection_export, input_artifact, onnx_export,
    onnx_runner, openvino_runner, segmentation_benchmark, segmentation_export,
)


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

    def test_detection_openvino_uses_registered_artifact_and_cpu_result_scope(self) -> None:
        with patch.object(sys, "argv", ["detection_benchmark", "--engine", "openvino"]):
            arguments = detection_benchmark._parse_arguments()

        self.assertEqual(arguments.model_path, Path("artifacts/yolo11n.onnx"))
        self.assertEqual(arguments.output_dir, Path("results/yolo11n/cpu"))

    def test_detection_export_uses_registered_model_defaults(self) -> None:
        with patch.object(sys, "argv", ["detection_export"]):
            arguments = detection_export._parse_arguments()

        self.assertEqual(arguments.model, "yolo11n")
        self.assertEqual(arguments.weights, Path("artifacts/yolo11n.pt"))
        self.assertEqual(arguments.output, Path("artifacts/yolo11n.onnx"))

    def test_second_detector_uses_its_own_registered_artifact_defaults(self) -> None:
        with patch.object(sys, "argv", ["detection_export", "--model", "yolo11s"]):
            export_arguments = detection_export._parse_arguments()
        with patch.object(sys, "argv", ["detection_benchmark", "--model", "yolo11s", "--engine", "onnxruntime"]):
            benchmark_arguments = detection_benchmark._parse_arguments()

        self.assertEqual(export_arguments.weights, Path("artifacts/yolo11s.pt"))
        self.assertEqual(export_arguments.output, Path("artifacts/yolo11s.onnx"))
        self.assertEqual(benchmark_arguments.model_path, Path("artifacts/yolo11s.onnx"))
        self.assertEqual(benchmark_arguments.output_dir, Path("results/yolo11s/cpu"))

    def test_yolo_native_artifact_defaults_use_raw_reference_filename(self) -> None:
        with patch.object(sys, "argv", ["input_artifact", "--model", "yolo11n"]):
            arguments = input_artifact._parse_arguments()

        self.assertEqual(arguments.output, Path("artifacts/inputs/yolo11n_seed69420_f32_nchw.bin"))
        self.assertEqual(
            arguments.reference_output,
            Path("artifacts/reference_outputs/yolo11n_input69420_f32_raw.bin"),
        )

    def test_segmentation_commands_use_the_registered_model_scope(self) -> None:
        with patch.object(sys, "argv", ["segmentation_export"]):
            export_arguments = segmentation_export._parse_arguments()
        with patch.object(sys, "argv", ["segmentation_benchmark", "--engine", "onnxruntime", "--device", "cuda:0"]):
            benchmark_arguments = segmentation_benchmark._parse_arguments()

        self.assertEqual(export_arguments.output, Path("artifacts/deeplabv3_resnet50.onnx"))
        self.assertEqual(benchmark_arguments.model_path, Path("artifacts/deeplabv3_resnet50.onnx"))
        self.assertEqual(benchmark_arguments.output_dir, Path("results/deeplabv3_resnet50/cuda_0"))

    def test_native_segmentation_artifacts_use_logits_reference_defaults(self) -> None:
        with patch.object(sys, "argv", ["input_artifact", "--model", "deeplabv3_resnet50"]):
            arguments = input_artifact._parse_arguments()

        self.assertEqual(
            arguments.output,
            Path("artifacts/inputs/deeplabv3_resnet50_seed69420_f32_nchw.bin"),
        )
        self.assertEqual(
            arguments.reference_output,
            Path("artifacts/reference_outputs/deeplabv3_resnet50_seed67_input69420_f32_logits.bin"),
        )


if __name__ == "__main__":
    unittest.main()
