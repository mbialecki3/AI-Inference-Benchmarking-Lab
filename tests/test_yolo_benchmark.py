"""Tests for schema-v1 OpenVINO records in the YOLO11n benchmark path."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference_bench.yolo_benchmark import benchmark_yolo_openvino
from inference_bench.yolo_runner import YoloRun


class YoloBenchmarkTests(unittest.TestCase):
    @patch("inference_bench.yolo_benchmark.sample_gpu_telemetry", return_value={"status": "unavailable"})
    @patch("inference_bench.yolo_benchmark.process_rss_bytes", return_value={"status": "available", "value": 1, "unit": "bytes"})
    @patch("inference_bench.yolo_benchmark.collect_environment", return_value={"git_revision": "test"})
    @patch("inference_bench.yolo_benchmark.run_yolo_pytorch")
    @patch("inference_bench.yolo_benchmark.run_yolo_openvino")
    def test_openvino_record_includes_raw_output_contract_and_parity(
        self,
        run_openvino_mock: object,
        run_pytorch_mock: object,
        _environment: object,
        _rss: object,
        _telemetry: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "yolo11n.onnx"
            artifact.write_bytes(b"onnx")
            output = np.zeros((1, 84, 2), dtype=np.float32)
            output[:, 4, :] = 1.0
            run_openvino_mock.return_value = YoloRun(
                "openvino", artifact, "cpu", (1, 3, 640, 640), 7, 1, 2,
                output, (2.0, 4.0), ("CPU",),
            )
            run_pytorch_mock.return_value = YoloRun(
                "pytorch_eager", Path("weights.pt"), "cpu", (1, 3, 640, 640), 7, 0, 1,
                output.copy(), (1.0,), (),
            )

            result = benchmark_yolo_openvino(
                artifact, "weights.pt", warmup_iterations=1, timed_iterations=2, verify_parity=True
            )

        record = result.summary()
        self.assertEqual(record["runner"]["engine"], "openvino")
        self.assertEqual(record["runner"]["active_providers"], ["CPU"])
        self.assertEqual(
            record["runner"]["configuration"],
            {
                "task": "detection",
                "output": "raw_pre_nms",
                "box_coordinate_channels": 4,
                "class_channel_axis": 1,
                "candidate_axis": 2,
                "class_channel_start": 4,
                "class_count": 80,
                "inference_precision": "f32",
            },
        )
        self.assertEqual(record["model"]["artifact_size_bytes"], 4)
        self.assertEqual(record["correctness"]["parity"]["prediction_agreement"], 1.0)
