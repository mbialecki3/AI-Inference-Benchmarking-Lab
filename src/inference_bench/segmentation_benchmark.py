"""Persist comparable schema-v1 records for segmentation benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_bench.benchmark_result import BenchmarkResult, OutputParity
from inference_bench.environment import collect_environment, process_rss_bytes, sample_gpu_telemetry
from inference_bench.output_paths import default_results_directory
from inference_bench.results import save_result
from inference_bench.segmentation import (
    available_segmentation_models,
    compare_segmentation_outputs,
    get_segmentation_model_spec,
)
from inference_bench.segmentation_openvino_runner import run_segmentation_openvino
from inference_bench.segmentation_runner import SegmentationRun, run_segmentation_onnx, run_segmentation_pytorch


def benchmark_segmentation_pytorch(
    model_name: str, *, device: str = "cpu", input_seed: int = 69420, model_seed: int = 67,
    warmup_iterations: int = 5, timed_iterations: int = 20,
) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_segmentation_pytorch(
        model_name, device=device, input_seed=input_seed, model_seed=model_seed,
        warmup_iterations=warmup_iterations, timed_iterations=timed_iterations,
    )
    return _record(model_name, run, before, sample_gpu_telemetry(), None)


def benchmark_segmentation_onnx(
    model_name: str, model_path: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    model_seed: int = 67, warmup_iterations: int = 5, timed_iterations: int = 20,
    verify_parity: bool = False,
) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_segmentation_onnx(
        model_name, model_path, device=device, input_seed=input_seed, model_seed=model_seed,
        warmup_iterations=warmup_iterations, timed_iterations=timed_iterations,
    )
    parity = _reference_parity(model_name, run, model_seed) if verify_parity else None
    return _record(model_name, run, before, sample_gpu_telemetry(), parity)


def benchmark_segmentation_openvino(
    model_name: str, model_path: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    model_seed: int = 67, warmup_iterations: int = 5, timed_iterations: int = 20,
    verify_parity: bool = False,
) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_segmentation_openvino(
        model_name, model_path, device=device, input_seed=input_seed, model_seed=model_seed,
        warmup_iterations=warmup_iterations, timed_iterations=timed_iterations,
    )
    parity = _reference_parity(model_name, run, model_seed) if verify_parity else None
    return _record(model_name, run, before, sample_gpu_telemetry(), parity)


def _reference_parity(model_name: str, run: SegmentationRun, model_seed: int) -> OutputParity:
    reference = run_segmentation_pytorch(
        model_name, device=run.device if run.engine == "onnxruntime" else "cpu",
        input_seed=run.input_seed, model_seed=model_seed, warmup_iterations=0, timed_iterations=1,
    )
    return compare_segmentation_outputs(
        reference.output, run.output, spec=get_segmentation_model_spec(model_name),
    )


def _record(
    model_name: str, run: SegmentationRun, before: dict[str, object], after: dict[str, object],
    parity: OutputParity | None,
) -> BenchmarkResult:
    configuration = get_segmentation_model_spec(model_name).benchmark_metadata()
    if run.engine == "openvino":
        configuration["inference_precision"] = "f32"
    return BenchmarkResult.create(
        engine=run.engine, model_name=model_name, device=run.device, input_shape=run.input_shape,
        input_seed=run.input_seed, model_seed=run.model_seed, warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations, active_providers=run.active_providers,
        artifact_path=run.model_path, latency_samples_ms=run.latencies_ms, process_rss=process_rss_bytes(),
        environment=collect_environment(), gpu_telemetry_before=before, gpu_telemetry_after=after,
        engine_configuration=configuration, parity=parity,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark registered raw segmentation inference.")
    parser.add_argument("--model", choices=available_segmentation_models(), default="deeplabv3_resnet50")
    parser.add_argument("--engine", choices=("pytorch", "onnxruntime", "openvino"), required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-seed", type=int, default=69420)
    parser.add_argument("--model-seed", type=int, default=67)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    spec = get_segmentation_model_spec(arguments.model)
    arguments.model_path = arguments.model_path or spec.onnx_path
    if arguments.output_dir is None:
        arguments.output_dir = default_results_directory(arguments.model, arguments.device)
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    common = dict(
        device=arguments.device, input_seed=arguments.input_seed, model_seed=arguments.model_seed,
        warmup_iterations=arguments.warmup, timed_iterations=arguments.iterations,
    )
    if arguments.engine == "pytorch":
        result = benchmark_segmentation_pytorch(arguments.model, **common)
    elif arguments.engine == "onnxruntime":
        result = benchmark_segmentation_onnx(
            arguments.model, arguments.model_path, verify_parity=arguments.verify_parity, **common,
        )
    else:
        result = benchmark_segmentation_openvino(
            arguments.model, arguments.model_path, verify_parity=arguments.verify_parity, **common,
        )
    path = save_result(result, arguments.output_dir)
    print(json.dumps({"result_path": str(path), "record": result.summary()}, indent=2))


if __name__ == "__main__":
    main()
