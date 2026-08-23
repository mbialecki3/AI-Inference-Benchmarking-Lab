"""PyTorch eager-mode reference runner.

The runner establishes the reference output and initial latency measurement
against which exported-engine runners will later be compared.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass

import torch

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.models import available_models, build_model


DEFAULT_MODEL_SEED = 67
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMED_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class PyTorchRunResult:
    """The output and timing data from one PyTorch reference run.

    ``output`` is moved to CPU after timing so a later ONNX runner can compare
    its output without depending on which device produced the reference.
    """

    model_name: str
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    model_seed: int
    warmup_iterations: int
    timed_iterations: int
    output: torch.Tensor
    latencies_ms: tuple[float, ...]

    def summary(self) -> dict[str, object]:
        """Return JSON-friendly metadata without serializing the full output."""

        return {
            "model": self.model_name,
            "device": self.device,
            "input_shape": list(self.input_shape),
            "input_seed": self.input_seed,
            "model_seed": self.model_seed,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "output_shape": list(self.output.shape),
            "output_dtype": str(self.output.dtype),
            "output_sum": float(self.output.sum().item()),
            "latency_ms": {
                "mean": statistics.fmean(self.latencies_ms),
                "p50": _percentile(self.latencies_ms, 50),
                "p95": _percentile(self.latencies_ms, 95),
            },
        }


def run_pytorch(
    model_name: str,
    *,
    device: torch.device | str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = DEFAULT_TIMED_ITERATIONS,
) -> PyTorchRunResult:
    """Run a model in PyTorch eager mode and return its reference output.

    Inputs are created before the timed region.  CUDA runs synchronize before
    each measurement finishes, so host timing includes queued GPU work instead
    of only timing kernel submission.
    """

    _validate_iteration_counts(warmup_iterations, timed_iterations)
    resolved_device = _resolve_device(device)

    # Model initialization is seeded, then the caller's random state is
    # restored.  This makes the randomly initialized reference reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = build_model(model_name)

    model = model.to(resolved_device).eval()
    input_tensor = make_input(
        model_name,
        batch_size=batch_size,
        seed=input_seed,
        device=resolved_device,
    )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(input_tensor)
        _synchronize(resolved_device)

        latencies_ms: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(timed_iterations):
            started_ns = time.perf_counter_ns()
            output = model(input_tensor)
            _synchronize(resolved_device)
            latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "The initial PyTorch runner supports models that return one tensor."
        )

    return PyTorchRunResult(
        model_name=model_name,
        device=str(resolved_device),
        input_shape=tuple(input_tensor.shape),
        input_seed=input_seed,
        model_seed=model_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        output=output.detach().cpu().contiguous(),
        latencies_ms=tuple(latencies_ms),
    )


def _resolve_device(device: torch.device | str) -> torch.device:
    """Validate the two device types in the initial benchmark matrix."""

    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("The initial PyTorch runner supports only CPU and CUDA.")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return resolved_device


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work before taking a host-side timing sample."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_iteration_counts(warmup_iterations: int, timed_iterations: int) -> None:
    if isinstance(warmup_iterations, bool) or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a non-negative integer.")
    if isinstance(timed_iterations, bool) or timed_iterations <= 0:
        raise ValueError("timed_iterations must be a positive integer.")


def _percentile(samples: tuple[float, ...], percentile: int) -> float:
    """Compute a linearly interpolated percentile without another dependency."""

    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch reference model.")
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_TIMED_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point and print a compact JSON summary."""

    arguments = _parse_arguments()
    result = run_pytorch(
        arguments.model,
        device=arguments.device,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
        model_seed=arguments.model_seed,
        warmup_iterations=arguments.warmup,
        timed_iterations=arguments.iterations,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
