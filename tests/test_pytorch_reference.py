"""Tests for PyTorch's deterministic reference-output contract."""

import unittest

import torch

from inference_bench.pytorch_runner import run_pytorch


class PyTorchReferenceTests(unittest.TestCase):
    def test_same_seeds_produce_identical_cpu_reference_logits(self) -> None:
        run_arguments = {
            "device": "cpu",
            "input_seed": 69420,
            "model_seed": 67,
            "warmup_iterations": 0,
            "timed_iterations": 1,
        }

        first = run_pytorch("resnet50", **run_arguments)
        second = run_pytorch("resnet50", **run_arguments)

        self.assertTrue(torch.equal(first.output, second.output))
        self.assertEqual(first.summary()["output_sum"], second.summary()["output_sum"])

    def test_mobilenet_v3_large_produces_deterministic_cpu_logits(self) -> None:
        run_arguments = {
            "device": "cpu",
            "input_seed": 69420,
            "model_seed": 67,
            "warmup_iterations": 0,
            "timed_iterations": 1,
        }

        first = run_pytorch("mobilenet_v3_large", **run_arguments)
        second = run_pytorch("mobilenet_v3_large", **run_arguments)

        self.assertEqual(first.output.shape, (1, 1000))
        self.assertTrue(torch.equal(first.output, second.output))


if __name__ == "__main__":
    unittest.main()
