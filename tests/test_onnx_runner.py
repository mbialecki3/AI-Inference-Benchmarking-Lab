"""Tests for the ONNX Runtime CPU runner and PyTorch parity contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_bench.onnx_export import export_onnx
from inference_bench.onnx_runner import CPU_EXECUTION_PROVIDER, run_onnx
from inference_bench.pytorch_runner import run_pytorch


class OnnxRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temporary_directory.name) / "resnet50.onnx"
        export_onnx("resnet50", cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_cpu_run_returns_resnet50_logits(self) -> None:
        result = run_onnx(
            "resnet50",
            self.model_path,
            warmup_iterations=0,
            timed_iterations=1,
        )

        self.assertEqual(result.input_shape, (1, 3, 224, 224))
        self.assertEqual(result.output.shape, (1, 1000))
        self.assertEqual(result.output.dtype, np.float32)
        self.assertEqual(result.active_providers, (CPU_EXECUTION_PROVIDER,))
        self.assertEqual(len(result.latencies_ms), 1)
        self.assertGreater(result.latencies_ms[0], 0)

    def test_cpu_output_matches_pytorch_reference(self) -> None:
        onnx_result = run_onnx(
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
            onnx_result.output,
            expected_output,
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            np.argmax(onnx_result.output, axis=1),
            np.argmax(expected_output, axis=1),
        )


if __name__ == "__main__":
    unittest.main()
