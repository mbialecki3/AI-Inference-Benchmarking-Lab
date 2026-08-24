"""Tests for automatic result and report output-directory conventions."""

import unittest
from pathlib import Path

from inference_bench.output_paths import (
    default_reports_directory,
    default_results_directory,
    device_directory_name,
)


class OutputPathTests(unittest.TestCase):
    def test_result_directory_scopes_model_and_device(self) -> None:
        self.assertEqual(
            default_results_directory("mobilenet_v3_large", "cuda:0"),
            Path("results/mobilenet_v3_large/cuda_0"),
        )

    def test_device_directory_name_is_portable(self) -> None:
        self.assertEqual(device_directory_name(" cuda:0 "), "cuda_0")

    def test_report_directory_mirrors_one_scoped_results_directory(self) -> None:
        self.assertEqual(
            default_reports_directory([Path("results/resnet50/cpu")]),
            Path("reports/resnet50/cpu"),
        )
        self.assertEqual(
            default_reports_directory([Path("results/resnet50/cpu/record.json")]),
            Path("reports/resnet50/cpu"),
        )

    def test_aggregate_inputs_keep_the_legacy_reports_directory(self) -> None:
        self.assertEqual(default_reports_directory([Path("results")]), Path("reports"))
        self.assertEqual(
            default_reports_directory([Path("results/resnet50/cpu"), Path("other.json")]),
            Path("reports"),
        )


if __name__ == "__main__":
    unittest.main()
