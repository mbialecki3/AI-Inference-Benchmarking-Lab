"""Tests for the OpenVINO raw-output YOLO11n CPU runner."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from inference_bench.detection import YOLO11N_INPUT_SHAPE, YOLO11N_OUTPUT_SHAPE
from inference_bench.openvino_runner import CPU_DEVICE
from inference_bench.detection_openvino_runner import run_detection_openvino


class YoloOpenVinoRunnerTests(unittest.TestCase):
    def test_cpu_run_preserves_raw_detection_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "yolo11n.onnx"
            _write_static_raw_detection_model(model_path)

            result = run_detection_openvino(
                "yolo11n", model_path,
                warmup_iterations=0,
                timed_iterations=1,
            )

        self.assertEqual(result.engine, "openvino")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.input_shape, YOLO11N_INPUT_SHAPE)
        self.assertEqual(result.active_providers, (CPU_DEVICE,))
        self.assertEqual(result.output.shape, YOLO11N_OUTPUT_SHAPE)
        self.assertEqual(result.output.dtype, np.float32)
        self.assertTrue(np.array_equal(result.output, np.zeros(YOLO11N_OUTPUT_SHAPE, dtype=np.float32)))
        self.assertEqual(len(result.latencies_ms), 1)
        self.assertGreater(result.latencies_ms[0], 0)

    def test_non_cpu_device_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "yolo11n.onnx"
            _write_static_raw_detection_model(model_path)
            with self.assertRaisesRegex(ValueError, "only cpu"):
                run_detection_openvino("yolo11n", model_path, device="cuda:0", warmup_iterations=0, timed_iterations=1)

    def test_incorrect_raw_output_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "wrong_output.onnx"
            _write_static_raw_detection_model(model_path, output_shape=(1, 85, 8400))
            with self.assertRaisesRegex(ValueError, "raw output shape"):
                run_detection_openvino("yolo11n", model_path, warmup_iterations=0, timed_iterations=1)


def _write_static_raw_detection_model(
    path: Path,
    *,
    output_shape: tuple[int, int, int] = YOLO11N_OUTPUT_SHAPE,
) -> None:
    """Create a lightweight valid ONNX graph with YOLO-compatible ports."""

    input_value = helper.make_tensor_value_info("images", TensorProto.FLOAT, YOLO11N_INPUT_SHAPE)
    output_value = helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_shape)
    shape = helper.make_tensor("raw_output_shape", TensorProto.INT64, [3], output_shape)
    value = helper.make_tensor("raw_output_value", TensorProto.FLOAT, [1], [0.0])
    graph = helper.make_graph(
        [helper.make_node("ConstantOfShape", ["raw_output_shape"], ["output0"], value=value)],
        "static_raw_yolo_output",
        [input_value],
        [output_value],
        initializer=[shape],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        path,
    )
