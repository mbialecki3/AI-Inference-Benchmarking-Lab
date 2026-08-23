"""Regression tests for the initial PyTorch reference runner."""

import unittest

import torch

from inference_bench.pytorch_runner import run_pytorch


class PyTorchRunnerTests(unittest.TestCase):
    def test_cpu_run_returns_resnet50_logits(self) -> None:
        result = run_pytorch(
            "resnet50",
            device="cpu",
            warmup_iterations=0,
            timed_iterations=1,
        )

        self.assertEqual(result.input_shape, (1, 3, 224, 224))
        self.assertEqual(tuple(result.output.shape), (1, 1000))
        self.assertEqual(result.output.dtype, torch.float32)
        self.assertEqual(len(result.latencies_ms), 1)
        self.assertGreater(result.latencies_ms[0], 0)


if __name__ == "__main__":
    unittest.main()
