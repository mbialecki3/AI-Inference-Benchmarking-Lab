"""Run an existing inference runner and persist a comparable benchmark record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_bench.benchmark_result import BenchmarkResult, compare_outputs
from inference_bench.environment import (
    collect_environment,
    process_rss_bytes,
    sample_gpu_telemetry,
)
from inference_bench.inputs import DEFAULT_INPUT_SEED
from inference_bench.models import available_models
from inference_bench.onnx_runner import (
    DEFAULT_TIMED_ITERATIONS as ONNX_DEFAULT_TIMED_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS as ONNX_DEFAULT_WARMUP_ITERATIONS,
    run_onnx,
)
from inference_bench.openvino_runner import (
    DEFAULT_TIMED_ITERATIONS as OPENVINO_DEFAULT_TIMED_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS as OPENVINO_DEFAULT_WARMUP_ITERATIONS,
    run_openvino,
)
from inference_bench.pytorch_runner import (
    DEFAULT_MODEL_SEED,
    DEFAULT_TIMED_ITERATIONS as PYTORCH_DEFAULT_TIMED_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS as PYTORCH_DEFAULT_WARMUP_ITERATIONS,
    run_pytorch,
)
from inference_bench.results import save_result


def benchmark_pytorch(
    model_name: str,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    warmup_iterations: int = PYTORCH_DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = PYTORCH_DEFAULT_TIMED_ITERATIONS,
    project_root: Path | str = ".",
) -> BenchmarkResult:
    """Benchmark PyTorch eager mode and return a persistent-schema record."""

    environment = collect_environment(project_root)
    telemetry_before = sample_gpu_telemetry()
    run = run_pytorch(
        model_name,
        device=device,
        batch_size=batch_size,
        input_seed=input_seed,
        model_seed=model_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
    )
    telemetry_after = sample_gpu_telemetry()
    return BenchmarkResult.create(
        engine="pytorch_eager",
        model_name=run.model_name,
        device=run.device,
        input_shape=run.input_shape,
        input_seed=run.input_seed,
        model_seed=run.model_seed,
        warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations,
        latency_samples_ms=run.latencies_ms,
        process_rss=process_rss_bytes(),
        environment=environment,
        gpu_telemetry_before=telemetry_before,
        gpu_telemetry_after=telemetry_after,
    )


def benchmark_onnx(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    warmup_iterations: int = ONNX_DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = ONNX_DEFAULT_TIMED_ITERATIONS,
    verify_parity: bool = False,
    project_root: Path | str = ".",
) -> BenchmarkResult:
    """Benchmark ONNX Runtime and optionally compare it with PyTorch output."""

    environment = collect_environment(project_root)
    telemetry_before = sample_gpu_telemetry()
    run = run_onnx(
        model_name,
        model_path,
        device=device,
        batch_size=batch_size,
        input_seed=input_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
    )
    telemetry_after = sample_gpu_telemetry()

    parity = None
    if verify_parity:
        reference = run_pytorch(
            model_name,
            device=run.device,
            batch_size=batch_size,
            input_seed=input_seed,
            warmup_iterations=0,
            timed_iterations=1,
        )
        parity = compare_outputs(reference.output.numpy(), run.output)

    return BenchmarkResult.create(
        engine="onnxruntime",
        model_name=run.model_name,
        device=run.device,
        input_shape=run.input_shape,
        input_seed=run.input_seed,
        model_seed=DEFAULT_MODEL_SEED if verify_parity else None,
        warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations,
        active_providers=run.active_providers,
        artifact_path=run.model_path,
        latency_samples_ms=run.latencies_ms,
        process_rss=process_rss_bytes(),
        environment=environment,
        gpu_telemetry_before=telemetry_before,
        gpu_telemetry_after=telemetry_after,
        parity=parity,
    )


def benchmark_openvino(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    warmup_iterations: int = OPENVINO_DEFAULT_WARMUP_ITERATIONS,
    timed_iterations: int = OPENVINO_DEFAULT_TIMED_ITERATIONS,
    verify_parity: bool = False,
    project_root: Path | str = ".",
) -> BenchmarkResult:
    """Benchmark OpenVINO CPU and optionally compare it with PyTorch output."""

    environment = collect_environment(project_root)
    telemetry_before = sample_gpu_telemetry()
    run = run_openvino(
        model_name,
        model_path,
        device=device,
        batch_size=batch_size,
        input_seed=input_seed,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
    )
    telemetry_after = sample_gpu_telemetry()

    parity = None
    if verify_parity:
        reference = run_pytorch(
            model_name,
            device="cpu",
            batch_size=batch_size,
            input_seed=input_seed,
            warmup_iterations=0,
            timed_iterations=1,
        )
        parity = compare_outputs(reference.output.numpy(), run.output)

    return BenchmarkResult.create(
        engine="openvino",
        model_name=run.model_name,
        device=run.device,
        input_shape=run.input_shape,
        input_seed=run.input_seed,
        model_seed=DEFAULT_MODEL_SEED if verify_parity else None,
        warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations,
        active_providers=run.active_devices,
        engine_configuration={"inference_precision": run.inference_precision},
        artifact_path=run.model_path,
        latency_samples_ms=run.latencies_ms,
        process_rss=process_rss_bytes(),
        environment=environment,
        gpu_telemetry_before=telemetry_before,
        gpu_telemetry_after=telemetry_after,
        parity=parity,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark and save a versioned result JSON record."
    )
    parser.add_argument(
        "--engine", choices=("pytorch", "onnxruntime", "openvino"), required=True
    )
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/resnet50.onnx"))
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Run one engine, save its record, and print its location and contents."""

    arguments = _parse_arguments()
    if arguments.engine == "pytorch":
        result = benchmark_pytorch(
            arguments.model,
            device=arguments.device,
            batch_size=arguments.batch_size,
            input_seed=arguments.input_seed,
            model_seed=arguments.model_seed,
            warmup_iterations=(
                arguments.warmup
                if arguments.warmup is not None
                else PYTORCH_DEFAULT_WARMUP_ITERATIONS
            ),
            timed_iterations=(
                arguments.iterations
                if arguments.iterations is not None
                else PYTORCH_DEFAULT_TIMED_ITERATIONS
            ),
        )
    elif arguments.engine == "onnxruntime":
        result = benchmark_onnx(
            arguments.model,
            arguments.model_path,
            device=arguments.device,
            batch_size=arguments.batch_size,
            input_seed=arguments.input_seed,
            warmup_iterations=(
                arguments.warmup
                if arguments.warmup is not None
                else ONNX_DEFAULT_WARMUP_ITERATIONS
            ),
            timed_iterations=(
                arguments.iterations
                if arguments.iterations is not None
                else ONNX_DEFAULT_TIMED_ITERATIONS
            ),
            verify_parity=arguments.verify_parity,
        )
    else:
        result = benchmark_openvino(
            arguments.model,
            arguments.model_path,
            device=arguments.device,
            batch_size=arguments.batch_size,
            input_seed=arguments.input_seed,
            warmup_iterations=(
                arguments.warmup
                if arguments.warmup is not None
                else OPENVINO_DEFAULT_WARMUP_ITERATIONS
            ),
            timed_iterations=(
                arguments.iterations
                if arguments.iterations is not None
                else OPENVINO_DEFAULT_TIMED_ITERATIONS
            ),
            verify_parity=arguments.verify_parity,
        )
    path = save_result(result, arguments.output_dir)
    print(json.dumps({"result_path": str(path), "record": result.summary()}, indent=2))


if __name__ == "__main__":
    main()
