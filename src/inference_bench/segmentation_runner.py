"""Warm-run PyTorch and ONNX Runtime runners for static segmentation logits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from inference_bench.segmentation import (
    build_segmentation_model,
    get_segmentation_model_spec,
    make_segmentation_input,
    raw_segmentation_tensor,
)


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass(frozen=True, slots=True)
class SegmentationRun:
    engine: str
    model_path: Path | None
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    model_seed: int
    warmup_iterations: int
    timed_iterations: int
    output: np.ndarray
    latencies_ms: tuple[float, ...]
    active_providers: tuple[str, ...] = ()


def run_segmentation_pytorch(
    model_name: str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    model_seed: int = 67,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
) -> SegmentationRun:
    """Run an untrained seeded segmentation model in PyTorch eager mode."""

    _validate_counts(warmup_iterations, timed_iterations)
    spec = get_segmentation_model_spec(model_name)
    resolved_device = _resolve_torch_device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = build_segmentation_model(spec.name)
    model = model.to(resolved_device).eval()
    inputs = make_segmentation_input(spec.name, seed=input_seed, device=resolved_device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _validate_pytorch_output(raw_segmentation_tensor(model(inputs)), spec.output_shape)
        _synchronize(resolved_device)
        samples: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(timed_iterations):
            started_at = time.perf_counter_ns()
            output = raw_segmentation_tensor(model(inputs))
            _synchronize(resolved_device)
            samples.append((time.perf_counter_ns() - started_at) / 1_000_000)
    if output is None:
        raise RuntimeError(f"{spec.name} did not produce a timed PyTorch output.")
    _validate_pytorch_output(output, spec.output_shape)
    return SegmentationRun(
        "pytorch_eager", None, str(resolved_device), tuple(inputs.shape), input_seed, model_seed,
        warmup_iterations, timed_iterations, output.detach().cpu().numpy().copy(), tuple(samples),
    )


def run_segmentation_onnx(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    model_seed: int = 67,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
) -> SegmentationRun:
    """Run one static segmentation ONNX artifact with ONNX Runtime."""

    _validate_counts(warmup_iterations, timed_iterations)
    spec = get_segmentation_model_spec(model_name)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name} ONNX model does not exist: {path}")
    resolved_device, providers = _resolve_ort_device(device)
    session = ort.InferenceSession(str(path), providers=providers)
    active_providers = tuple(session.get_providers())
    if active_providers[0] != providers[0]:
        raise RuntimeError(f"ONNX Runtime did not activate {providers[0]} first: {active_providers!r}")
    _validate_onnx_interface(session, spec.input_name, spec.output_name, spec.input_shape, spec.output_shape)
    inputs = make_segmentation_input(spec.name, seed=input_seed).numpy()
    feeds = {spec.input_name: inputs}
    for _ in range(warmup_iterations):
        _onnx_once(session, feeds, spec.output_name, spec.output_shape)
    _synchronize(torch.device(resolved_device))
    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        started_at = time.perf_counter_ns()
        output = _onnx_once(session, feeds, spec.output_name, spec.output_shape)
        _synchronize(torch.device(resolved_device))
        samples.append((time.perf_counter_ns() - started_at) / 1_000_000)
    if output is None:
        raise RuntimeError(f"{spec.name} did not produce a timed ONNX output.")
    return SegmentationRun(
        "onnxruntime", path, resolved_device, tuple(inputs.shape), input_seed, model_seed,
        warmup_iterations, timed_iterations, output, tuple(samples), active_providers,
    )


def _onnx_once(session: ort.InferenceSession, feeds: dict[str, np.ndarray], output_name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = session.run([output_name], feeds)
    if len(values) != 1 or not isinstance(values[0], np.ndarray) or tuple(values[0].shape) != expected_shape:
        raise TypeError(f"ONNX Runtime must return one float32 segmentation output shaped {expected_shape}.")
    return values[0]


def _validate_onnx_interface(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    inputs_metadata, outputs_metadata = session.get_inputs(), session.get_outputs()
    if len(inputs_metadata) != 1 or len(outputs_metadata) != 1:
        raise ValueError("Segmentation ONNX must expose exactly one input and one raw output.")
    input_metadata, output_metadata = inputs_metadata[0], outputs_metadata[0]
    if input_metadata.name != input_name or output_metadata.name != output_name:
        raise ValueError(f"Segmentation ONNX must expose {input_name} -> {output_name}.")
    if tuple(input_metadata.shape) != input_shape or tuple(output_metadata.shape) != output_shape:
        raise ValueError(f"Segmentation ONNX must use {input_shape} -> {output_shape} static shapes.")
    if input_metadata.type != "tensor(float)" or output_metadata.type != "tensor(float)":
        raise ValueError("Segmentation ONNX ports must use float32 tensors.")


def _validate_pytorch_output(output: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    if tuple(output.shape) != expected_shape or output.dtype != torch.float32:
        raise TypeError(f"PyTorch must return one float32 segmentation output shaped {expected_shape}.")


def _resolve_torch_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Segmentation runners support only cpu and cuda:0.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return resolved


def _resolve_ort_device(device: str) -> tuple[str, list[str]]:
    if device == "cpu":
        return "cpu", [CPU_PROVIDER]
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("Segmentation runners support only cpu and cuda:0.")
    if CUDA_PROVIDER not in ort.get_available_providers() or not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but ONNX Runtime or PyTorch cannot access it.")
    return "cuda:0", [CUDA_PROVIDER, CPU_PROVIDER]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_counts(warmup_iterations: int, timed_iterations: int) -> None:
    if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int) or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a non-negative integer.")
    if isinstance(timed_iterations, bool) or not isinstance(timed_iterations, int) or timed_iterations <= 0:
        raise ValueError("timed_iterations must be a positive integer.")
