"""ONNX Runtime runners for validated benchmark artifacts.

CPU remains the default reference path. CUDA is an explicit second path that
uses ONNX Runtime's CUDA execution provider and synchronizes GPU work before
recording a host-side latency sample.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.metrics import LatencyMetrics
from inference_bench.models import available_models
from inference_bench.onnx_export import INPUT_NAME, OUTPUT_NAME
from inference_bench.runtime_options import OnnxRuntimeOptions
from inference_bench.timing import elapsed_ms


CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMED_ITERATIONS = 20


@dataclass(frozen=True, slots=True)
class OnnxRunResult:
    """The output and warm-run timing data from one ONNX Runtime run."""

    model_name: str
    model_path: Path
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    warmup_iterations: int
    timed_iterations: int
    active_providers: tuple[str, ...]
    output: np.ndarray
    latencies_ms: tuple[float, ...]
    model_load_ms: float = 0.0
    device_latencies_ms: tuple[float, ...] = ()
    runtime_options: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        """Return JSON-friendly metadata without serializing the full output."""

        latency = LatencyMetrics.from_samples(self.latencies_ms)
        return {
            "model": self.model_name,
            "model_path": str(self.model_path),
            "device": self.device,
            "input_shape": list(self.input_shape),
            "input_seed": self.input_seed,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "active_providers": list(self.active_providers),
            "output_shape": list(self.output.shape),
            "output_dtype": str(self.output.dtype),
            "output_sum": float(self.output.sum()),
            "latency_ms": latency.summary(),
            "cold_start_model_load_ms": self.model_load_ms,
            "device_latency_ms": None,
            "runtime_options": self.runtime_options or {},
        }


def run_onnx(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = DEFAULT_TIMED_ITERATIONS,
    runtime_options: OnnxRuntimeOptions | None = None,
) -> OnnxRunResult:
    """Run one exported model with the requested ONNX Runtime device.

    Session creation is deliberately outside the timed region.  These warm-run
    samples therefore measure request latency after model loading and warm-up,
    not process startup or export time. CUDA calls are synchronized before
    taking each host-side timing sample, so queued GPU work is included.
    """

    _validate_iteration_counts(warmup_iterations, timed_iterations)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")

    resolved_device, requested_provider, _ = _resolve_device(device)
    options = runtime_options or OnnxRuntimeOptions()
    def create_session() -> ort.InferenceSession:
        return ort.InferenceSession(
            str(path), sess_options=options.session_options(), providers=options.providers(resolved_device),
        )
    session, model_load_ms = elapsed_ms(create_session)
    active_providers = tuple(session.get_providers())
    if active_providers[0] != requested_provider:
        raise RuntimeError(
            "ONNX Runtime did not activate the requested execution provider first. "
            f"Requested: {requested_provider!r}; active: {active_providers!r}"
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
    _synchronize(resolved_device)

    latencies_ms: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        output, latency_ms = elapsed_ms(lambda: _synchronized_run(session, feeds, resolved_device))
        latencies_ms.append(latency_ms)

    if output is None:
        raise RuntimeError("ONNX Runtime did not produce a timed output.")

    return OnnxRunResult(
        model_name=model_name,
        model_path=path,
        device=resolved_device,
        input_shape=tuple(input_array.shape),
        input_seed=input_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        active_providers=active_providers,
        output=output,
        latencies_ms=tuple(latencies_ms),
        model_load_ms=model_load_ms,
        runtime_options=options.summary(),
    )


def _resolve_device(device: str) -> tuple[str, str, list[str]]:
    """Validate the initial ONNX Runtime CPU/CUDA execution matrix."""

    if device == "cpu":
        return device, CPU_EXECUTION_PROVIDER, [CPU_EXECUTION_PROVIDER]
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("The initial ONNX runner supports only cpu and cuda:0.")
    if CUDA_EXECUTION_PROVIDER not in ort.get_available_providers():
        raise RuntimeError(
            "CUDA was requested, but ONNX Runtime does not expose "
            "CUDAExecutionProvider."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")

    # The CPU provider is deliberate fallback for graph operations unsupported
    # by CUDA. CUDA remains first, which is the requested execution path.
    return "cuda:0", CUDA_EXECUTION_PROVIDER, [
        CUDA_EXECUTION_PROVIDER,
        CPU_EXECUTION_PROVIDER,
    ]


def _synchronize(device: str) -> None:
    """Wait for queued CUDA work before a host-side timing sample ends."""

    if device == "cuda:0":
        torch.cuda.synchronize(0)


def _synchronized_run(session: ort.InferenceSession, feeds: dict[str, np.ndarray], device: str) -> np.ndarray:
    output = _run_once(session, feeds)
    _synchronize(device)
    return output


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


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark ONNX model on CPU or CUDA."
    )
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_TIMED_ITERATIONS)
    parser.add_argument("--ort-graph-optimization", choices=("disable", "basic", "extended", "all"), default="all")
    parser.add_argument("--ort-execution-mode", choices=("sequential", "parallel"), default="sequential")
    parser.add_argument("--ort-intra-op-threads", type=int)
    parser.add_argument("--ort-inter-op-threads", type=int)
    parser.add_argument("--ort-cuda-conv-algorithm", choices=("exhaustive", "heuristic", "default"))
    arguments = parser.parse_args()
    if arguments.model_path is None:
        arguments.model_path = Path("artifacts") / f"{arguments.model}.onnx"
    return arguments


def main() -> None:
    """Run the command-line entry point and print a compact JSON summary."""

    arguments = _parse_arguments()
    result = run_onnx(
        arguments.model,
        arguments.model_path,
        device=arguments.device,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
        warmup_iterations=arguments.warmup,
        timed_iterations=arguments.iterations,
        runtime_options=OnnxRuntimeOptions(
            graph_optimization_level=arguments.ort_graph_optimization,
            execution_mode=arguments.ort_execution_mode,
            intra_op_num_threads=arguments.ort_intra_op_threads,
            inter_op_num_threads=arguments.ort_inter_op_threads,
            cuda_conv_algorithm=arguments.ort_cuda_conv_algorithm,
        ),
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
