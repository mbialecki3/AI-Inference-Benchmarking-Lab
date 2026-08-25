"""Tests for benchmark orchestration without repeating expensive model execution."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from inference_bench.benchmark import benchmark_onnx, benchmark_openvino, benchmark_pytorch
from inference_bench.onnx_runner import OnnxRunResult
from inference_bench.openvino_runner import OpenVinoRunResult
from inference_bench.pytorch_runner import PyTorchRunResult


class BenchmarkTests(unittest.TestCase):
    @patch("inference_bench.benchmark.sample_gpu_telemetry", return_value={"status": "unavailable"})
    @patch("inference_bench.benchmark.process_rss_bytes", return_value={"status": "available", "value": 1, "unit": "bytes"})
    @patch("inference_bench.benchmark.collect_environment", return_value={"git_revision": "test"})
    @patch("inference_bench.benchmark.run_pytorch")
    def test_pytorch_record_wraps_existing_runner_output(
        self, run_pytorch_mock: object, _environment: object, _rss: object, _telemetry: object
    ) -> None:
        run_pytorch_mock.return_value = PyTorchRunResult(
            model_name="resnet50",
            device="cpu",
            input_shape=(1, 3, 224, 224),
            input_seed=7,
            model_seed=8,
            warmup_iterations=1,
            timed_iterations=2,
            output=torch.zeros((1, 1000)),
            latencies_ms=(2.0, 4.0),
        )

        result = benchmark_pytorch("resnet50", warmup_iterations=1, timed_iterations=2)

        record = result.summary()
        self.assertEqual(record["runner"]["engine"], "pytorch_eager")
        self.assertEqual(record["model"]["model_seed"], 8)
        self.assertEqual(record["measurement"]["latency_ms"]["p99"], 3.98)
        self.assertIsNone(record["model"]["artifact_path"])

    @patch("inference_bench.benchmark.sample_gpu_telemetry", return_value={"status": "unavailable"})
    @patch("inference_bench.benchmark.process_rss_bytes", return_value={"status": "available", "value": 1, "unit": "bytes"})
    @patch("inference_bench.benchmark.collect_environment", return_value={"git_revision": "test"})
    @patch("inference_bench.benchmark.run_pytorch")
    @patch("inference_bench.benchmark.run_onnx")
    def test_onnx_record_can_include_reference_parity(
        self,
        run_onnx_mock: object,
        run_pytorch_mock: object,
        _environment: object,
        _rss: object,
        _telemetry: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model.onnx"
            artifact.write_bytes(b"onnx")
            output = np.array([[0.1, 0.9]], dtype=np.float32)
            run_onnx_mock.return_value = OnnxRunResult(
                model_name="resnet50",
                model_path=artifact,
                device="cpu",
                input_shape=(1, 3, 224, 224),
                input_seed=7,
                warmup_iterations=1,
                timed_iterations=2,
                active_providers=("CPUExecutionProvider",),
                output=output,
                latencies_ms=(2.0, 4.0),
            )
            run_pytorch_mock.return_value = PyTorchRunResult(
                model_name="resnet50",
                device="cpu",
                input_shape=(1, 3, 224, 224),
                input_seed=7,
                model_seed=67,
                warmup_iterations=0,
                timed_iterations=1,
                output=torch.from_numpy(output.copy()),
                latencies_ms=(1.0,),
            )

            result = benchmark_onnx(
                "resnet50", artifact, warmup_iterations=1, timed_iterations=2, verify_parity=True
            )

        record = result.summary()
        self.assertEqual(record["runner"]["active_providers"], ["CPUExecutionProvider"])
        self.assertEqual(record["model"]["artifact_size_bytes"], 4)
        self.assertEqual(record["correctness"]["parity"]["prediction_agreement"], 1.0)

    @patch("inference_bench.benchmark.sample_gpu_telemetry", return_value={"status": "unavailable"})
    @patch("inference_bench.benchmark.process_rss_bytes", return_value={"status": "available", "value": 1, "unit": "bytes"})
    @patch("inference_bench.benchmark.collect_environment", return_value={"git_revision": "test"})
    @patch("inference_bench.benchmark.run_pytorch")
    @patch("inference_bench.benchmark.run_openvino")
    def test_openvino_record_can_include_reference_parity(
        self,
        run_openvino_mock: object,
        run_pytorch_mock: object,
        _environment: object,
        _rss: object,
        _telemetry: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "model.onnx"
            artifact.write_bytes(b"onnx")
            output = np.array([[0.1, 0.9]], dtype=np.float32)
            run_openvino_mock.return_value = OpenVinoRunResult(
                model_name="resnet50",
                model_path=artifact,
                device="cpu",
                input_shape=(1, 3, 224, 224),
                input_seed=7,
                warmup_iterations=1,
                timed_iterations=2,
                active_devices=("CPU",),
                inference_precision="f32",
                output=output,
                latencies_ms=(2.0, 4.0),
            )
            run_pytorch_mock.return_value = PyTorchRunResult(
                model_name="resnet50",
                device="cpu",
                input_shape=(1, 3, 224, 224),
                input_seed=7,
                model_seed=67,
                warmup_iterations=0,
                timed_iterations=1,
                output=torch.from_numpy(output.copy()),
                latencies_ms=(1.0,),
            )

            result = benchmark_openvino(
                "resnet50", artifact, warmup_iterations=1, timed_iterations=2, verify_parity=True
            )

        record = result.summary()
        self.assertEqual(record["runner"]["engine"], "openvino")
        self.assertEqual(record["runner"]["active_providers"], ["CPU"])
        self.assertEqual(
            record["runner"]["configuration"],
            {"inference_precision": "f32", "performance_hint": "latency"},
        )
        self.assertEqual(record["model"]["artifact_size_bytes"], 4)
        self.assertEqual(record["correctness"]["parity"]["prediction_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
