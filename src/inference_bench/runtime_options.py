"""Validated, serializable runtime optimization settings.

The defaults intentionally preserve the project's raw-output parity baseline.
Non-default values are experiments: records retain the exact settings so they
cannot be mistaken for an otherwise identical baseline result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import onnxruntime as ort
import openvino as ov
from openvino.properties import hint


_ORT_GRAPH_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
_OPENVINO_PRECISIONS = {"f32": ov.Type.f32, "f16": ov.Type.f16, "bf16": ov.Type.bf16}


@dataclass(frozen=True, slots=True)
class OnnxRuntimeOptions:
    """Session/provider controls used for one ONNX Runtime experiment."""

    graph_optimization_level: str = "all"
    execution_mode: str = "sequential"
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    enable_cpu_mem_arena: bool = True
    enable_mem_pattern: bool = True
    cuda_conv_algorithm: str | None = None

    def session_options(self) -> ort.SessionOptions:
        if self.graph_optimization_level not in _ORT_GRAPH_LEVELS:
            raise ValueError(f"Unknown ONNX Runtime graph optimization level: {self.graph_optimization_level!r}.")
        if self.execution_mode not in {"sequential", "parallel"}:
            raise ValueError("ONNX Runtime execution_mode must be 'sequential' or 'parallel'.")
        for name, value in (("intra_op_num_threads", self.intra_op_num_threads), ("inter_op_num_threads", self.inter_op_num_threads)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer when provided.")
        if self.cuda_conv_algorithm not in {None, "exhaustive", "heuristic", "default"}:
            raise ValueError("cuda_conv_algorithm must be exhaustive, heuristic, default, or None.")

        options = ort.SessionOptions()
        options.graph_optimization_level = _ORT_GRAPH_LEVELS[self.graph_optimization_level]
        options.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL
            if self.execution_mode == "parallel"
            else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        if self.intra_op_num_threads is not None:
            options.intra_op_num_threads = self.intra_op_num_threads
        if self.inter_op_num_threads is not None:
            options.inter_op_num_threads = self.inter_op_num_threads
        options.enable_cpu_mem_arena = self.enable_cpu_mem_arena
        options.enable_mem_pattern = self.enable_mem_pattern
        return options

    def providers(self, device: str) -> list[object]:
        if device != "cuda:0":
            return ["CPUExecutionProvider"]
        provider_options: dict[str, str] = {}
        if self.cuda_conv_algorithm is not None:
            provider_options["cudnn_conv_algo_search"] = self.cuda_conv_algorithm.upper()
        return [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]

    def summary(self) -> dict[str, Any]:
        return {
            "graph_optimization_level": self.graph_optimization_level,
            "execution_mode": self.execution_mode,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "enable_cpu_mem_arena": self.enable_cpu_mem_arena,
            "enable_mem_pattern": self.enable_mem_pattern,
            "cuda_conv_algorithm": self.cuda_conv_algorithm,
        }


@dataclass(frozen=True, slots=True)
class OpenVinoOptions:
    """Compile-time OpenVINO performance and precision experiment controls."""

    performance_hint: str = "latency"
    inference_precision: str = "f32"

    def compile_configuration(self) -> dict[object, object]:
        if self.performance_hint not in {"latency", "throughput"}:
            raise ValueError("OpenVINO performance_hint must be 'latency' or 'throughput'.")
        if self.inference_precision not in _OPENVINO_PRECISIONS:
            raise ValueError("OpenVINO inference_precision must be f32, f16, or bf16.")
        performance_mode = (
            hint.PerformanceMode.LATENCY
            if self.performance_hint == "latency"
            else hint.PerformanceMode.THROUGHPUT
        )
        return {
            hint.performance_mode: performance_mode,
            hint.inference_precision: _OPENVINO_PRECISIONS[self.inference_precision],
        }

    def summary(self) -> dict[str, str]:
        return {
            "performance_hint": self.performance_hint,
            "inference_precision": self.inference_precision,
        }
