"""Persist comparable schema-v1 records for registered detection benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_bench.benchmark_result import BenchmarkResult
from inference_bench.detection import (
    available_detection_models,
    compare_detection_outputs,
    get_detection_model_spec,
)
from inference_bench.environment import collect_environment, process_rss_bytes, sample_gpu_telemetry
from inference_bench.output_paths import default_results_directory
from inference_bench.results import save_result
from inference_bench.detection_openvino_runner import run_detection_openvino
from inference_bench.detection_runner import run_detection_onnx, run_detection_pytorch
from inference_bench.runtime_options import OnnxRuntimeOptions, OpenVinoOptions


def benchmark_detection_pytorch(
    model_name: str,
    weights: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
    precision: str = "fp32",
) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_detection_pytorch(model_name, weights, device=device, input_seed=input_seed,
                                warmup_iterations=warmup_iterations, timed_iterations=timed_iterations, precision=precision)
    return _record(model_name, run, before, sample_gpu_telemetry(), None)


def benchmark_detection_onnx(
    model_name: str,
    model_path: Path | str,
    weights: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
    verify_parity: bool = False,
    runtime_options: OnnxRuntimeOptions | None = None,
) -> BenchmarkResult:
    before = sample_gpu_telemetry()
    run = run_detection_onnx(model_name, model_path, device=device, input_seed=input_seed,
                             warmup_iterations=warmup_iterations, timed_iterations=timed_iterations, runtime_options=runtime_options)
    parity = None
    if verify_parity:
        reference = run_detection_pytorch(model_name, weights, device=run.device, input_seed=input_seed,
                                          warmup_iterations=0, timed_iterations=1)
        parity = compare_detection_outputs(
            reference.output, run.output, layout=get_detection_model_spec(model_name).layout
        )
    return _record(model_name, run, before, sample_gpu_telemetry(), parity)


def benchmark_detection_openvino(
    model_name: str,
    model_path: Path | str,
    weights: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
    verify_parity: bool = False,
    runtime_options: OpenVinoOptions | None = None,
) -> BenchmarkResult:
    """Benchmark one raw detector on OpenVINO CPU and optionally check parity."""

    before = sample_gpu_telemetry()
    run = run_detection_openvino(model_name, model_path, device=device, input_seed=input_seed,
                                 warmup_iterations=warmup_iterations, timed_iterations=timed_iterations, runtime_options=runtime_options)
    parity = None
    if verify_parity:
        reference = run_detection_pytorch(model_name, weights, device="cpu", input_seed=input_seed,
                                          warmup_iterations=0, timed_iterations=1)
        parity = compare_detection_outputs(
            reference.output, run.output, layout=get_detection_model_spec(model_name).layout
        )
    return _record(model_name, run, before, sample_gpu_telemetry(), parity)


def _record(
    model_name: str,
    run: object,
    before: dict[str, object],
    after: dict[str, object],
    parity: object,
) -> BenchmarkResult:
    configuration = get_detection_model_spec(model_name).layout.benchmark_metadata()
    if run.engine == "openvino":
        configuration.update({"inference_precision": "f32", "performance_hint": "latency"})
    configuration.update(run.runtime_configuration or {})
    return BenchmarkResult.create(
        engine=run.engine, model_name=model_name, device=run.device, input_shape=run.input_shape,
        input_seed=run.input_seed, model_seed=None, warmup_iterations=run.warmup_iterations,
        timed_iterations=run.timed_iterations, active_providers=run.active_providers, artifact_path=run.model_path,
        latency_samples_ms=run.latencies_ms, process_rss=process_rss_bytes(), environment=collect_environment(),
        gpu_telemetry_before=before, gpu_telemetry_after=after,
        engine_configuration=configuration, parity=parity,
        cold_start_model_load_ms=run.model_load_ms, device_latency_samples_ms=run.device_latencies_ms,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark registered raw detection inference.")
    parser.add_argument("--model", choices=available_detection_models(), default="yolo11n")
    parser.add_argument("--engine", choices=("pytorch", "onnxruntime", "openvino"), required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--input-seed", type=int, default=69420)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-parity", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--ort-graph-optimization", choices=("disable", "basic", "extended", "all"), default="all")
    parser.add_argument("--ort-execution-mode", choices=("sequential", "parallel"), default="sequential")
    parser.add_argument("--ort-intra-op-threads", type=int)
    parser.add_argument("--ort-inter-op-threads", type=int)
    parser.add_argument("--ort-cuda-conv-algorithm", choices=("exhaustive", "heuristic", "default"))
    parser.add_argument("--openvino-performance-hint", choices=("latency", "throughput"), default="latency")
    parser.add_argument("--openvino-inference-precision", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    spec = get_detection_model_spec(arguments.model)
    arguments.weights = arguments.weights or spec.weights_path
    arguments.model_path = arguments.model_path or spec.onnx_path
    if arguments.output_dir is None:
        arguments.output_dir = default_results_directory(arguments.model, arguments.device)
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    if arguments.engine == "pytorch":
        result = benchmark_detection_pytorch(
            arguments.model, arguments.weights, device=arguments.device, input_seed=arguments.input_seed,
            warmup_iterations=arguments.warmup, timed_iterations=arguments.iterations,
            precision=arguments.precision,
        )
    elif arguments.engine == "onnxruntime":
        result = benchmark_detection_onnx(
            arguments.model, arguments.model_path, arguments.weights, device=arguments.device,
            input_seed=arguments.input_seed, warmup_iterations=arguments.warmup,
            timed_iterations=arguments.iterations, verify_parity=arguments.verify_parity,
            runtime_options=OnnxRuntimeOptions(
                graph_optimization_level=arguments.ort_graph_optimization, execution_mode=arguments.ort_execution_mode,
                intra_op_num_threads=arguments.ort_intra_op_threads, inter_op_num_threads=arguments.ort_inter_op_threads,
                cuda_conv_algorithm=arguments.ort_cuda_conv_algorithm,
            ),
        )
    else:
        result = benchmark_detection_openvino(
            arguments.model, arguments.model_path, arguments.weights, device=arguments.device,
            input_seed=arguments.input_seed, warmup_iterations=arguments.warmup,
            timed_iterations=arguments.iterations, verify_parity=arguments.verify_parity,
            runtime_options=OpenVinoOptions(
                performance_hint=arguments.openvino_performance_hint,
                inference_precision=arguments.openvino_inference_precision,
            ),
        )
    path = save_result(result, arguments.output_dir)
    print(json.dumps({"result_path": str(path), "record": result.summary()}, indent=2))


if __name__ == "__main__":
    main()
