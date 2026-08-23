"""Tests that optional system probes return structured unavailable values."""

import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from inference_bench.environment import process_rss_bytes, sample_gpu_telemetry


class EnvironmentTests(unittest.TestCase):
    @patch("inference_bench.environment.shutil.which", return_value=None)
    def test_missing_nvidia_smi_is_explicitly_unavailable(self, _which: object) -> None:
        self.assertEqual(
            sample_gpu_telemetry(),
            {"status": "unavailable", "reason": "nvidia-smi was not found"},
        )

    def test_process_rss_never_raises_when_collected(self) -> None:
        sample = process_rss_bytes()
        self.assertIn(sample["status"], {"available", "unavailable"})
        if sample["status"] == "available":
            self.assertGreater(sample["value"], 0)
            self.assertEqual(sample["unit"], "bytes")

    @patch("inference_bench.environment.subprocess.run")
    @patch("inference_bench.environment.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_nvidia_smi_sample_is_normalized(
        self, _which: object, run_mock: object
    ) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout="NVIDIA Test GPU, 999.1, 123, 45, 67.8" + chr(10),
        )

        sample = sample_gpu_telemetry()

        self.assertEqual(sample["status"], "available")
        self.assertEqual(sample["gpus"][0]["name"], "NVIDIA Test GPU")
        self.assertEqual(sample["gpus"][0]["memory_used_mib"], 123.0)
        self.assertEqual(sample["gpus"][0]["utilization_percent"], 45.0)
        self.assertEqual(sample["gpus"][0]["power_watts"], 67.8)


if __name__ == "__main__":
    unittest.main()
