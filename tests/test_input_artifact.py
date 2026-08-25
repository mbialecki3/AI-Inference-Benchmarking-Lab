"""Tests for the shared binary input handoff used by native runners."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference_bench.detection import YOLO11N_INPUT_SHAPE, YOLO11N_OUTPUT_SHAPE, make_detection_input
from inference_bench.input_artifact import (
    export_input_artifact,
    export_reference_output_artifact,
)
from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.pytorch_runner import DEFAULT_MODEL_SEED, run_pytorch
from inference_bench.segmentation import (
    DEEPLABV3_RESNET50_INPUT_SHAPE,
    DEEPLABV3_RESNET50_OUTPUT_SHAPE,
    make_segmentation_input,
)


class InputArtifactTests(unittest.TestCase):
    def test_artifact_preserves_the_seeded_float32_nchw_input_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resnet50_input.bin"
            artifact = export_input_artifact("resnet50", path)

            expected = make_input("resnet50", seed=DEFAULT_INPUT_SEED).numpy()
            actual = np.fromfile(path, dtype="<f4").reshape(expected.shape)

            self.assertEqual(artifact.input_shape, (1, 3, 224, 224))
            self.assertEqual(artifact.input_seed, DEFAULT_INPUT_SEED)
            self.assertEqual(artifact.dtype, "float32")
            self.assertEqual(artifact.size_bytes, expected.nbytes)
            np.testing.assert_array_equal(actual, expected)

    def test_reference_artifact_preserves_seeded_pytorch_logits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resnet50_logits.bin"
            artifact = export_reference_output_artifact("resnet50", path)

            expected = run_pytorch(
                "resnet50",
                warmup_iterations=0,
                timed_iterations=1,
            ).output.numpy()
            actual = np.fromfile(path, dtype="<f4").reshape(expected.shape)

            self.assertEqual(artifact.output_shape, (1, 1000))
            self.assertEqual(artifact.input_seed, DEFAULT_INPUT_SEED)
            self.assertEqual(artifact.model_seed, DEFAULT_MODEL_SEED)
            self.assertEqual(artifact.dtype, "float32")
            self.assertEqual(artifact.size_bytes, expected.nbytes)
            np.testing.assert_array_equal(actual, expected)

    def test_mobilenet_input_and_reference_artifacts_preserve_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_artifact = export_input_artifact(
                "mobilenet_v3_large", directory / "mobilenet_input.bin"
            )
            output_artifact = export_reference_output_artifact(
                "mobilenet_v3_large", directory / "mobilenet_logits.bin"
            )

        self.assertEqual(input_artifact.input_shape, (1, 3, 224, 224))
        self.assertEqual(output_artifact.output_shape, (1, 1000))
        self.assertEqual(input_artifact.dtype, "float32")
        self.assertEqual(output_artifact.dtype, "float32")

    def test_efficientnet_input_and_reference_artifacts_preserve_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_artifact = export_input_artifact(
                "efficientnet_b0", directory / "efficientnet_input.bin"
            )
            output_artifact = export_reference_output_artifact(
                "efficientnet_b0", directory / "efficientnet_logits.bin"
            )

        self.assertEqual(input_artifact.input_shape, (1, 3, 224, 224))
        self.assertEqual(output_artifact.output_shape, (1, 1000))
        self.assertEqual(input_artifact.dtype, "float32")
        self.assertEqual(output_artifact.dtype, "float32")

    def test_yolo_artifacts_preserve_raw_input_and_reference_output_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "yolo_input.bin"
            output_path = directory / "yolo_raw.bin"
            input_artifact = export_input_artifact("yolo11n", input_path)
            expected_output = np.arange(np.prod(YOLO11N_OUTPUT_SHAPE), dtype=np.float32).reshape(YOLO11N_OUTPUT_SHAPE)
            with patch(
                "inference_bench.input_artifact.run_detection_pytorch",
                return_value=SimpleNamespace(output=expected_output),
            ):
                output_artifact = export_reference_output_artifact(
                    "yolo11n", output_path, weights="yolo11n.pt"
                )

            actual_input = np.fromfile(input_path, dtype="<f4").reshape(YOLO11N_INPUT_SHAPE)
            actual_output = np.fromfile(output_path, dtype="<f4").reshape(YOLO11N_OUTPUT_SHAPE)

        np.testing.assert_array_equal(actual_input, make_detection_input().numpy())
        np.testing.assert_array_equal(actual_output, expected_output)
        self.assertEqual(input_artifact.input_shape, YOLO11N_INPUT_SHAPE)
        self.assertEqual(output_artifact.output_shape, YOLO11N_OUTPUT_SHAPE)
        self.assertIsNone(output_artifact.model_seed)

    def test_segmentation_artifacts_preserve_rank_four_input_and_output_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "deeplab_input.bin"
            output_path = directory / "deeplab_logits.bin"
            expected_output = np.arange(
                np.prod(DEEPLABV3_RESNET50_OUTPUT_SHAPE), dtype=np.float32
            ).reshape(DEEPLABV3_RESNET50_OUTPUT_SHAPE)
            input_artifact = export_input_artifact("deeplabv3_resnet50", input_path)
            with patch(
                "inference_bench.input_artifact.run_segmentation_pytorch",
                return_value=SimpleNamespace(output=expected_output),
            ):
                output_artifact = export_reference_output_artifact("deeplabv3_resnet50", output_path)

            actual_input = np.fromfile(input_path, dtype="<f4").reshape(DEEPLABV3_RESNET50_INPUT_SHAPE)
            actual_output = np.fromfile(output_path, dtype="<f4").reshape(DEEPLABV3_RESNET50_OUTPUT_SHAPE)

        np.testing.assert_array_equal(actual_input, make_segmentation_input("deeplabv3_resnet50").numpy())
        np.testing.assert_array_equal(actual_output, expected_output)
        self.assertEqual(input_artifact.input_shape, DEEPLABV3_RESNET50_INPUT_SHAPE)
        self.assertEqual(output_artifact.output_shape, DEEPLABV3_RESNET50_OUTPUT_SHAPE)
        self.assertEqual(output_artifact.model_seed, DEFAULT_MODEL_SEED)


if __name__ == "__main__":
    unittest.main()
