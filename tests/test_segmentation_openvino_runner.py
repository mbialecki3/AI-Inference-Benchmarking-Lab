"""Tests for the OpenVINO static segmentation CPU runner."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from inference_bench.openvino_runner import CPU_DEVICE
from inference_bench.segmentation import DEEPLABV3_RESNET50_INPUT_SHAPE, DEEPLABV3_RESNET50_OUTPUT_SHAPE
from inference_bench.segmentation_openvino_runner import run_segmentation_openvino


class SegmentationOpenVinoRunnerTests(unittest.TestCase):
    def test_cpu_run_preserves_static_raw_logit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "deeplabv3_resnet50.onnx"
            _write_static_segmentation_model(model_path)
            result = run_segmentation_openvino("deeplabv3_resnet50", model_path, warmup_iterations=0, timed_iterations=1)

        self.assertEqual(result.engine, "openvino")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.input_shape, DEEPLABV3_RESNET50_INPUT_SHAPE)
        self.assertEqual(result.active_providers, (CPU_DEVICE,))
        self.assertEqual(result.output.shape, DEEPLABV3_RESNET50_OUTPUT_SHAPE)
        self.assertEqual(result.output.dtype, np.float32)
        self.assertEqual(len(result.latencies_ms), 1)

    def test_non_cpu_device_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "deeplabv3_resnet50.onnx"
            _write_static_segmentation_model(model_path)
            with self.assertRaisesRegex(ValueError, "only cpu"):
                run_segmentation_openvino("deeplabv3_resnet50", model_path, device="cuda:0", warmup_iterations=0, timed_iterations=1)


def _write_static_segmentation_model(path: Path) -> None:
    input_value = helper.make_tensor_value_info("images", TensorProto.FLOAT, DEEPLABV3_RESNET50_INPUT_SHAPE)
    output_value = helper.make_tensor_value_info("logits", TensorProto.FLOAT, DEEPLABV3_RESNET50_OUTPUT_SHAPE)
    shape = helper.make_tensor("logits_shape", TensorProto.INT64, [4], DEEPLABV3_RESNET50_OUTPUT_SHAPE)
    value = helper.make_tensor("logits_value", TensorProto.FLOAT, [1], [0.0])
    graph = helper.make_graph(
        [helper.make_node("ConstantOfShape", ["logits_shape"], ["logits"], value=value)],
        "static_segmentation_logits", [input_value], [output_value], initializer=[shape],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path)
