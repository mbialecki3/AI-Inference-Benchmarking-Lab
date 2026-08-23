"""OpenVINO CPU runner for validated ONNX benchmark artifacts.

The initial OpenVINO path intentionally targets CPU only. It loads the same
ONNX artifact and uses the same deterministic NumPy inputs as ONNX Runtime,
so output-parity checks isolate engine behavior from model conversion changes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.metrics import LatencyMetrics
from inference_bench.models import available_models
from inference_bench.onnx_export import INPUT_NAME, OUTPUT_NAME


CPU_DEVICE = "CPU"
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_TIMED_ITERATIONS = 20

# The runtime is designed for one Core to manage models and device plugins for
# a process. Reusing it also avoids repeated plugin lifecycle work per run.
_CORE = ov.Core()


@dataclass(frozen=True, slots=True)
class OpenVinoRunResult:
    """The output and warm-run timing data from one OpenVINO CPU run."""

    model_name: str
    model_path: Path
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    warmup_iterations: int
    timed_iterations: int
    active_devices: tuple[str, ...]
    inference_precision: str
    output: np.ndarray
    latencies_ms: tuple[float, ...]

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
            "active_devices": list(self.active_devices),
            "inference_precision": self.inference_precision,
            "output_shape": list(self.output.shape),
            "output_dtype": str(self.output.dtype),
            "output_sum": float(self.output.sum()),
            "latency_ms": latency.summary(),
        }


def run_openvino(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = DEFAULT_TIMED_ITERATIONS,
) -> OpenVinoRunResult:
    """Run one exported ONNX model with OpenVINO on CPU.

    Model loading and compilation are deliberately outside the timed region.
    Each recorded sample therefore measures warm synchronous inference, using
    the same fixed input contract as the PyTorch and ONNX Runtime runners.
    """

    _validate_iteration_counts(warmup_iterations, timed_iterations)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")

    resolved_device = _resolve_device(device, _CORE)
    model = _CORE.read_model(str(path))
    _validate_model_interface(model)
    # OpenVINO CPU may otherwise select lower-precision kernels. This runner
    # is a numerical-parity reference, so it explicitly retains float32.
    compiled_model = _CORE.compile_model(
        model,
        CPU_DEVICE,
        {hint.inference_precision: ov.Type.f32},
    )
    active_devices = tuple(
        str(active_device)
        for active_device in compiled_model.get_property("EXECUTION_DEVICES")
    )
    if CPU_DEVICE not in active_devices:
        raise RuntimeError(
            "OpenVINO did not report CPU as an active execution device. "
            f"Reported: {active_devices!r}"
        )

    input_array = make_input(
        model_name,
        batch_size=batch_size,
        seed=input_seed,
        device="cpu",
    ).numpy()

    for _ in range(warmup_iterations):
        _run_once(compiled_model, input_array)

    latencies_ms: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        started_ns = time.perf_counter_ns()
        output = _run_once(compiled_model, input_array)
        latencies_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)

    if output is None:
        raise RuntimeError("OpenVINO did not produce a timed output.")

    # ``CompiledModel.__call__`` may expose OpenVINO-owned output memory. Copy
    # after timing so the returned parity artifact outlives the compiled model
    # without making host copying part of the measured inference latency.
    stable_output = output.copy()

    return OpenVinoRunResult(
        model_name=model_name,
        model_path=path,
        device=resolved_device,
        input_shape=tuple(input_array.shape),
        input_seed=input_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        active_devices=active_devices,
        inference_precision="f32",
        output=stable_output,
        latencies_ms=tuple(latencies_ms),
    )


def _resolve_device(device: str, core: ov.Core) -> str:
    """Validate the initial OpenVINO CPU-only benchmark matrix."""

    if device != "cpu":
        raise ValueError("The initial OpenVINO runner supports only cpu.")
    if CPU_DEVICE not in core.available_devices:
        raise RuntimeError("OpenVINO does not expose a CPU execution device.")
    return device


def _validate_model_interface(model: ov.Model) -> None:
    """Ensure the artifact still matches the exporter's stable I/O contract."""

    input_names = [sorted(port.get_names()) for port in model.inputs]
    output_names = [sorted(port.get_names()) for port in model.outputs]
    if input_names != [[INPUT_NAME]]:
        raise ValueError(
            f"Expected OpenVINO input {INPUT_NAME!r}, got {input_names!r}."
        )
    if output_names != [[OUTPUT_NAME]]:
        raise ValueError(
            f"Expected OpenVINO output {OUTPUT_NAME!r}, got {output_names!r}."
        )


def _run_once(compiled_model: ov.CompiledModel, input_array: np.ndarray) -> np.ndarray:
    """Run one synchronous request and return the named logits tensor."""

    outputs = compiled_model({INPUT_NAME: input_array})
    output = outputs[compiled_model.output(OUTPUT_NAME)]
    if not isinstance(output, np.ndarray):
        raise TypeError("The initial OpenVINO runner expects one NumPy logits array.")
    return output


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
    parser = argparse.ArgumentParser(description="Run a benchmark ONNX model with OpenVINO.")
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/resnet50.onnx"))
    parser.add_argument("--device", default="cpu", help="cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_TIMED_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point and print a compact JSON summary."""

    arguments = _parse_arguments()
    result = run_openvino(
        arguments.model,
        arguments.model_path,
        device=arguments.device,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
        warmup_iterations=arguments.warmup,
        timed_iterations=arguments.iterations,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
