"""Regression tests for schema-v1 comparison reporting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference_bench.reporting import latest_comparable_records, load_records, write_report


def _record(engine: str, mean_ms: float, created_at: str, *, device: str = "cpu") -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at_utc": created_at,
        "runner": {"engine": engine, "device": device, "active_providers": [], "configuration": {}},
        "model": {"name": "resnet50", "input_shape": [1, 3, 224, 224], "input_seed": 69420, "model_seed": 67},
        "configuration": {"warmup_iterations": 5, "timed_iterations": 20},
        "measurement": {
            "latency_ms": {"mean": mean_ms, "p50": mean_ms, "p95": mean_ms * 1.1, "p99": mean_ms * 1.2},
            "throughput_samples_per_second": 1000 / mean_ms,
            "process_rss": {"status": "available", "value": 1024 * 1024, "unit": "bytes"},
        },
        "correctness": {"parity": {"prediction_agreement": 1.0}},
    }


class ReportingTests(unittest.TestCase):
    def test_selects_latest_record_per_engine_in_fair_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for name, record in {
                "old.json": _record("openvino", 10.0, "2026-08-20T00:00:00+00:00"),
                "new.json": _record("openvino", 8.0, "2026-08-21T00:00:00+00:00"),
                "torch.json": _record("pytorch_eager", 12.0, "2026-08-21T00:00:00+00:00"),
                "cuda.json": _record("onnxruntime", 2.0, "2026-08-22T00:00:00+00:00", device="cuda:0"),
            }.items():
                (directory / name).write_text(json.dumps(record), encoding="utf-8")

            selected = latest_comparable_records(load_records([directory]))

        self.assertEqual([record.engine for record in selected], ["openvino", "pytorch_eager"])
        self.assertEqual(selected[0].metric("mean"), 8.0)

    def test_writes_report_and_matplotlib_png_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for engine, mean_ms in (("pytorch_eager", 12.0), ("onnxruntime", 4.0), ("onnxruntime_cpp", 3.0)):
                (directory / f"{engine}.json").write_text(
                    json.dumps(_record(engine, mean_ms, "2026-08-21T00:00:00+00:00")), encoding="utf-8"
                )

            paths = write_report([directory], directory / "reports")

            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("ONNX Runtime (C++)", report)
            self.assertIn("OpenVINO (Python): no record found", report)
            self.assertIn("Mean ms", report)
            self.assertEqual(paths["mean_latency_plot"].suffix, ".png")
            self.assertEqual(paths["throughput_plot"].suffix, ".png")
            self.assertTrue(paths["mean_latency_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(paths["throughput_plot"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
