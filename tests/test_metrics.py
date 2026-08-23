"""Unit tests for engine-neutral benchmark measurement calculations."""

import unittest

from inference_bench.metrics import LatencyMetrics, percentile, throughput_samples_per_second


class MetricsTests(unittest.TestCase):
    def test_latency_summary_includes_p99_and_preserves_samples(self) -> None:
        metrics = LatencyMetrics.from_samples((1.0, 2.0, 3.0, 4.0))

        self.assertEqual(metrics.samples_ms, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(metrics.mean_ms, 2.5)
        self.assertEqual(metrics.p50_ms, 2.5)
        self.assertAlmostEqual(metrics.p95_ms, 3.85)
        self.assertAlmostEqual(metrics.p99_ms, 3.97)
        self.assertEqual(metrics.summary()["min"], 1.0)
        self.assertEqual(metrics.summary()["max"], 4.0)

    def test_invalid_measurements_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LatencyMetrics.from_samples(())
        with self.assertRaises(ValueError):
            LatencyMetrics.from_samples((0.0,))
        with self.assertRaises(ValueError):
            percentile((1.0,), 101)
        with self.assertRaises(ValueError):
            throughput_samples_per_second(0, 1.0)

    def test_throughput_uses_explicit_batch_size(self) -> None:
        self.assertEqual(throughput_samples_per_second(2, 4.0), 500.0)


if __name__ == "__main__":
    unittest.main()
