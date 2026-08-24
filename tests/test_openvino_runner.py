"""Tests for the OpenVINO CPU runner and PyTorch parity contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from inference_bench.openvino_runner import CPU_DEVICE, run_openvino
from inference_bench.pytorch_runner import run_pytorch
from _support import export_onnx_quietly


class OpenVinoRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temporary_directory.name) / "resnet50.onnx"
        export_onnx_quietly("resnet50", cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_cpu_run_returns_resnet50_logits(self) -> None:
        result = run_openvino(
            "resnet50",
            self.model_path,
            warmup_iterations=0,
            timed_iterations=1,
        )

        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.input_shape, (1, 3, 224, 224))
        self.assertEqual(result.active_devices, (CPU_DEVICE,))
        self.assertEqual(result.inference_precision, "f32")
        self.assertEqual(result.output.shape, (1, 1000))
        self.assertEqual(result.output.dtype, np.float32)
        self.assertEqual(len(result.latencies_ms), 1)
        self.assertGreater(result.latencies_ms[0], 0)

    def test_cpu_output_matches_pytorch_reference(self) -> None:
        openvino_result = run_openvino(
            "resnet50",
            self.model_path,
            warmup_iterations=0,
            timed_iterations=1,
        )
        pytorch_result = run_pytorch(
            "resnet50",
            device="cpu",
            warmup_iterations=0,
            timed_iterations=1,
        )

        expected_output = pytorch_result.output.numpy()
        np.testing.assert_allclose(
            openvino_result.output,
            expected_output,
            rtol=1e-3,
            atol=1e-4,
        )
        np.testing.assert_array_equal(
            np.argmax(openvino_result.output, axis=1),
            np.argmax(expected_output, axis=1),
        )

    def test_mobilenet_v3_large_cpu_output_matches_pytorch_reference(self) -> None:
        model_path = Path(self.temporary_directory.name) / "mobilenet_v3_large.onnx"
        export_onnx_quietly("mobilenet_v3_large", model_path)
        openvino_result = run_openvino(
            "mobilenet_v3_large",
            model_path,
            warmup_iterations=0,
            timed_iterations=1,
        )
        pytorch_result = run_pytorch(
            "mobilenet_v3_large",
            device="cpu",
            warmup_iterations=0,
            timed_iterations=1,
        )

        self.assertEqual(openvino_result.output.shape, (1, 1000))
        np.testing.assert_allclose(
            openvino_result.output,
            pytorch_result.output.numpy(),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_non_cpu_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only cpu"):
            run_openvino(
                "resnet50",
                self.model_path,
                device="cuda:0",
                warmup_iterations=0,
                timed_iterations=1,
            )

    def test_missing_model_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_openvino(
                "resnet50",
                self.model_path.with_name("missing.onnx"),
                warmup_iterations=0,
                timed_iterations=1,
            )

    def test_unexpected_model_interface_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "unexpected_interface.onnx"
            _write_unexpected_interface_model(model_path)
            with self.assertRaisesRegex(ValueError, "Expected OpenVINO input"):
                run_openvino(
                    "resnet50",
                    model_path,
                    warmup_iterations=0,
                    timed_iterations=1,
                )


def _write_unexpected_interface_model(path: Path) -> None:
    """Create a valid ONNX artifact whose port names violate our contract."""

    input_value = helper.make_tensor_value_info(
        "unexpected_input", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_value = helper.make_tensor_value_info(
        "unexpected_output", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["unexpected_input"], ["unexpected_output"])],
        "unexpected_interface",
        [input_value],
        [output_value],
    )
    onnx.save(helper.make_model(graph), path)


if __name__ == "__main__":
    unittest.main()
