"""CPU reference runner for validated ONNX benchmark artifacts.

This is intentionally the first ONNX Runtime step: it reuses the deterministic
input contract and explicitly selects the CPU execution provider.  CUDA is
added only after CPU output parity is established.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.models import available_models
from inference_bench.onnx_export import INPUT_NAME, OUTPUT_NAME


CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMED_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class OnnxRunResult:
    """The output and warm-run timing data from one ONNX Runtime CPU run."""

    model_name: str
    model_path: Path
    input_shape: tuple[int, ...]
    input_seed: int
    warmup_iterations: int
    timed_iterations: int
    active_providers: tuple[str, ...]
    output: np.ndarray
    latencies_ms: tuple[float, ...]

    def summary(self) -> dict[str, object]:
        """Return JSON-friendly metadata without serializing the full output."""

        return {
            "model": self.model_name,
            "model_path": str(self.model_path),
            "input_shape": list(self.input_shape),
            "input_seed": self.input_seed,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "active_providers": list(self.active_providers),
            "output_shape": list(self.output.shape),
            "output_dtype": str(self.output.dtype),
            "output_sum": float(self.output.sum()),
            "latency_ms": {
                "mean": statistics.fmean(self.latencies_ms),
                "p50": _percentile(self.latencies_ms, 50),
                "p95": _percentile(self.latencies_ms, 95),
            },
        }


def run_onnx(
    model_name: str,
    model_path: Path | str,
    *,
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = DEFAULT_TIMED_ITERATIONS,
) -> OnnxRunResult:
    """Run one exported model with ONNX Runtime's CPU execution provider.

    Session creation is deliberately outside the timed region.  These warm-run
    samples therefore measure request latency after model loading and warm-up,
    not process startup or export time.
    """

    _validate_iteration_counts(warmup_iterations, timed_iterations)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")

    session = ort.InferenceSession(str(path), providers=[CPU_EXECUTION_PROVIDER])
    active_providers = tuple(session.get_providers())
    if CPU_EXECUTION_PROVIDER not in active_providers:
        raise RuntimeError(
            "ONNX Runtime did not activate the requested CPUExecutionProvider. "
            f"Active providers: {active_providers!r}"
        )
    _validate_session_interface(session)

    input_array = make_input(
        model_name,
        batch_size=batch_size,
        seed=input_seed,
        device="cpu",
    ).numpy()
    feeds = {INPUT_NAME: input_array}

    for _ in range(warmup_iterations):
        _run_once(session, feeds)

    latencies_ms: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        started_ns = time.perf_counter_ns()
        output = _run_once(session, feeds)
        latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)

    if output is None:
        raise RuntimeError("ONNX Runtime did not produce a timed output.")

    return OnnxRunResult(
        model_name=model_name,
        model_path=path,
        input_shape=tuple(input_array.shape),
        input_seed=input_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        active_providers=active_providers,
        output=output,
        latencies_ms=tuple(latencies_ms),
    )


def _validate_session_interface(session: ort.InferenceSession) -> None:
    """Ensure the artifact still matches the exporter's stable I/O contract."""

    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if input_names != [INPUT_NAME]:
        raise ValueError(f"Expected ONNX input {INPUT_NAME!r}, got {input_names!r}.")
    if output_names != [OUTPUT_NAME]:
        raise ValueError(f"Expected ONNX output {OUTPUT_NAME!r}, got {output_names!r}.")


def _run_once(
    session: ort.InferenceSession, feeds: dict[str, np.ndarray]
) -> np.ndarray:
    """Run one request and return the sole logits tensor."""

    outputs = session.run([OUTPUT_NAME], feeds)
    if len(outputs) != 1 or not isinstance(outputs[0], np.ndarray):
        raise TypeError("The initial ONNX runner expects one NumPy logits array.")
    return outputs[0]


def _validate_iteration_counts(warmup_iterations: int, timed_iterations: int) -> None:
    if (
        isinstance(warmup_iterations, bool)
        or not isinstance(warmup_iterations, int)
        or warmup_iterations < 0
    ):
        raise ValueError("warmup_iterations must be a non-negative integer.")
    if (
        isinstance(timed_iterations, bool)
        or not isinstance(timed_iterations, int)
        or timed_iterations <= 0
    ):
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
    parser = argparse.ArgumentParser(description="Run a benchmark ONNX model on CPU.")
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/resnet50.onnx"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_TIMED_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point and print a compact JSON summary."""

    arguments = _parse_arguments()
    result = run_onnx(
        arguments.model,
        arguments.model_path,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
        warmup_iterations=arguments.warmup,
        timed_iterations=arguments.iterations,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
