"""Tests for schema-v1 semantic-segmentation benchmark records."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference_bench.segmentation_benchmark import benchmark_segmentation_openvino
from inference_bench.segmentation_runner import SegmentationRun


class SegmentationBenchmarkTests(unittest.TestCase):
    @patch("inference_bench.segmentation_benchmark.sample_gpu_telemetry", return_value={"status": "unavailable"})
    @patch("inference_bench.segmentation_benchmark.process_rss_bytes", return_value={"status": "available", "value": 1, "unit": "bytes"})
    @patch("inference_bench.segmentation_benchmark.collect_environment", return_value={"git_revision": "test"})
    @patch("inference_bench.segmentation_benchmark.run_segmentation_pytorch")
    @patch("inference_bench.segmentation_benchmark.run_segmentation_openvino")
    def test_openvino_record_includes_raw_logit_contract_and_parity(
        self, run_openvino_mock: object, run_pytorch_mock: object, _environment: object,
        _rss: object, _telemetry: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory) / "deeplabv3_resnet50.onnx"
            artifact.write_bytes(b"onnx")
            output = np.zeros((1, 21, 224, 224), dtype=np.float32)
            output[:, 3, :, :] = 1.0
            run_openvino_mock.return_value = SegmentationRun(
                "openvino", artifact, "cpu", (1, 3, 224, 224), 7, 67, 1, 2,
                output, (2.0, 4.0), ("CPU",),
            )
            run_pytorch_mock.return_value = SegmentationRun(
                "pytorch_eager", None, "cpu", (1, 3, 224, 224), 7, 67, 0, 1,
                output.copy(), (1.0,), (),
            )
            result = benchmark_segmentation_openvino(
                "deeplabv3_resnet50", artifact, warmup_iterations=1, timed_iterations=2, verify_parity=True,
            )

        record = result.summary()
        self.assertEqual(record["runner"]["engine"], "openvino")
        self.assertEqual(record["runner"]["configuration"], {
            "task": "semantic_segmentation", "output": "raw_logits", "class_channel_axis": 1,
            "class_count": 21, "inference_precision": "f32", "performance_hint": "latency",
        })
        self.assertEqual(record["model"]["model_seed"], 67)
        self.assertEqual(record["correctness"]["parity"]["prediction_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
