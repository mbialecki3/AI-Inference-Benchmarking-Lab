"""Persist comparable schema-v1 records for the YOLO11n detection baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_bench.benchmark_result import BenchmarkResult
from inference_bench.detection import YOLO11N, YOLO11N_ONNX, YOLO11N_WEIGHTS, compare_detection_outputs
from inference_bench.environment import collect_environment, process_rss_bytes, sample_gpu_telemetry
from inference_bench.output_paths import default_results_directory
from inference_bench.results import save_result
from inference_bench.yolo_runner import run_yolo_onnx, run_yolo_pytorch


def benchmark_yolo_pytorch(weights: Path | str, *, device: str = "cpu", input_seed: int = 69420,
                           warmup_iterations: int = 5, timed_iterations: int = 20) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_yolo_pytorch(weights, device=device, input_seed=input_seed,
                           warmup_iterations=warmup_iterations, timed_iterations=timed_iterations)
    return _record(run, before, sample_gpu_telemetry(), None)


def benchmark_yolo_onnx(model_path: Path | str, weights: Path | str, *, device: str = "cpu", input_seed: int = 69420,
                        warmup_iterations: int = 5, timed_iterations: int = 20, verify_parity: bool = False) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_yolo_onnx(model_path, device=device, input_seed=input_seed,
                        warmup_iterations=warmup_iterations, timed_iterations=timed_iterations)
    parity = None
    if verify_parity:
        reference = run_yolo_pytorch(weights, device=run.device, input_seed=input_seed,
                                     warmup_iterations=0, timed_iterations=1)
        parity = compare_detection_outputs(reference.output, run.output)
    return _record(run, before, sample_gpu_telemetry(), parity)


def _record(run: object, before: dict[str, object], after: dict[str, object], parity: object) -> BenchmarkResult:
    return BenchmarkResult.create(
        engine=run.engine, model_name=YOLO11N, device=run.device, input_shape=run.input_shape,
        input_seed=run.input_seed, model_seed=None, warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations, active_providers=run.active_providers, artifact_path=run.model_path,
        latency_samples_ms=run.latencies_ms, process_rss=process_rss_bytes(), environment=collect_environment(),
        gpu_telemetry_before=before, gpu_telemetry_after=after,
        engine_configuration={"task": "detection", "output": "raw_pre_nms", "class_count": 80}, parity=parity,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark raw YOLO11n detection inference.")
    parser.add_argument("--engine", choices=("pytorch", "onnxruntime"), required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--weights", type=Path, default=YOLO11N_WEIGHTS)
    parser.add_argument("--model-path", type=Path, default=YOLO11N_ONNX)
    parser.add_argument("--input-seed", type=int, default=69420)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.output_dir is None:
        arguments.output_dir = default_results_directory(YOLO11N, arguments.device)
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    if arguments.engine == "pytorch":
        result = benchmark_yolo_pytorch(arguments.weights, device=arguments.device, input_seed=arguments.input_seed,
                                        warmup_iterations=arguments.warmup, timed_iterations=arguments.iterations)
    else:
        result = benchmark_yolo_onnx(arguments.model_path, arguments.weights, device=arguments.device,
                                     input_seed=arguments.input_seed, warmup_iterations=arguments.warmup,
                                     timed_iterations=arguments.iterations, verify_parity=arguments.verify_parity)
    path = save_result(result, arguments.output_dir)
    print(json.dumps({"result_path": str(path), "record": result.summary()}, indent=2))


if __name__ == "__main__":
    main()
