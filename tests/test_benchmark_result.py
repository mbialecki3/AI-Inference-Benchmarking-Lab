"""Tests for the stable result schema and JSON persistence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_bench.benchmark_result import SCHEMA_VERSION, BenchmarkResult, compare_outputs
from inference_bench.results import save_result


class BenchmarkResultTests(unittest.TestCase):
    def test_output_parity_reports_numerical_and_prediction_agreement(self) -> None:
        parity = compare_outputs(
            np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32),
            np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32),
        )

        self.assertAlmostEqual(parity.max_absolute_error, 0.1)
        self.assertAlmostEqual(parity.prediction_agreement, 1.0)
        self.assertGreater(parity.max_relative_error, 0)

    def test_result_serializes_and_persists_all_required_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            artifact = directory / "model.onnx"
            artifact.write_bytes(b"onnx")
            result = BenchmarkResult.create(
                engine="onnxruntime",
                model_name="resnet50",
                device="cpu",
                input_shape=(2, 3, 224, 224),
                input_seed=69420,
                model_seed=None,
                warmup_iterations=5,
                timed_iterations=2,
                active_providers=("CPUExecutionProvider",),
                artifact_path=artifact,
                latency_samples_ms=(2.0, 4.0),
                process_rss={"status": "unavailable", "reason": "test"},
                environment={"git_revision": "abc123"},
                gpu_telemetry_before={"status": "unavailable"},
                gpu_telemetry_after={"status": "unavailable"},
                cold_start_model_load_ms=12.5,
                device_latency_samples_ms=(1.0, 2.0),
            )

            record = result.summary()
            self.assertEqual(record["schema_version"], SCHEMA_VERSION)
            self.assertEqual(record["model"]["artifact_size_bytes"], 4)
            self.assertAlmostEqual(
                record["measurement"]["throughput_samples_per_second"], 666.6666666666666
            )
            self.assertEqual(record["measurement"]["latency_ms"]["p99"], 3.98)
            self.assertEqual(record["measurement"]["cold_start_model_load_ms"], 12.5)
            self.assertEqual(record["measurement"]["device_latency_ms"]["samples"], [1.0, 2.0])
            self.assertIsNone(record["correctness"]["parity"])

            path = save_result(result, directory / "results")
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)

    def test_mismatched_outputs_cannot_be_compared(self) -> None:
        with self.assertRaises(ValueError):
            compare_outputs(np.zeros((1, 2)), np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
