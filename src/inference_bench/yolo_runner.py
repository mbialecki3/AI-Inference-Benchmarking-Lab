"""Warm-run PyTorch and ONNX Runtime runners for raw YOLO11n detection output."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from inference_bench.detection import YOLO11N, load_yolo11n, make_detection_input, raw_detection_tensor


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass(frozen=True, slots=True)
class YoloRun:
    engine: str
    model_path: Path | None
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    warmup_iterations: int
    timed_iterations: int
    output: np.ndarray
    latencies_ms: tuple[float, ...]
    active_providers: tuple[str, ...] = ()


def run_yolo_pytorch(
    weights: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    warmup_iterations: int = 5, timed_iterations: int = 20,
) -> YoloRun:
    """Run the underlying pretrained YOLO11n PyTorch module without NMS."""

    _validate_counts(warmup_iterations, timed_iterations)
    resolved_device = _resolve_torch_device(device)
    model = load_yolo11n(weights).model.to(resolved_device).eval()
    inputs = make_detection_input(seed=input_seed, device=resolved_device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            raw_detection_tensor(model(inputs))
        _synchronize(resolved_device)
        samples: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(timed_iterations):
            started_at = time.perf_counter_ns()
            output = raw_detection_tensor(model(inputs))
            _synchronize(resolved_device)
            samples.append((time.perf_counter_ns() - started_at) / 1_000_000)
    if output is None:
        raise RuntimeError("YOLO11n did not produce a timed PyTorch output.")
    return YoloRun("pytorch_eager", Path(weights), str(resolved_device), tuple(inputs.shape), input_seed,
                   warmup_iterations, timed_iterations, output.detach().cpu().numpy().copy(), tuple(samples))


def run_yolo_onnx(
    model_path: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    warmup_iterations: int = 5, timed_iterations: int = 20,
) -> YoloRun:
    """Run a static raw-output YOLO11n ONNX artifact with ONNX Runtime."""

    _validate_counts(warmup_iterations, timed_iterations)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO11n ONNX model does not exist: {path}")
    resolved_device, providers = _resolve_ort_device(device)
    session = ort.InferenceSession(str(path), providers=providers)
    active_providers = tuple(session.get_providers())
    if active_providers[0] != providers[0]:
        raise RuntimeError(f"ONNX Runtime did not activate {providers[0]} first: {active_providers!r}")
    inputs_metadata, outputs_metadata = session.get_inputs(), session.get_outputs()
    if len(inputs_metadata) != 1 or len(outputs_metadata) != 1:
        raise ValueError("YOLO11n ONNX must expose exactly one input and one raw output.")
    inputs = make_detection_input(seed=input_seed).numpy()
    feeds = {inputs_metadata[0].name: inputs}
    for _ in range(warmup_iterations):
        _onnx_once(session, outputs_metadata[0].name, feeds)
    _synchronize(torch.device(resolved_device))
    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        started_at = time.perf_counter_ns()
        output = _onnx_once(session, outputs_metadata[0].name, feeds)
        _synchronize(torch.device(resolved_device))
        samples.append((time.perf_counter_ns() - started_at) / 1_000_000)
    if output is None:
        raise RuntimeError("YOLO11n did not produce a timed ONNX output.")
    return YoloRun("onnxruntime", path, resolved_device, tuple(inputs.shape), input_seed,
                   warmup_iterations, timed_iterations, output, tuple(samples), active_providers)


def _onnx_once(session: ort.InferenceSession, output_name: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    values = session.run([output_name], feeds)
    if len(values) != 1 or not isinstance(values[0], np.ndarray) or values[0].ndim != 3:
        raise TypeError("YOLO11n ONNX must return one rank-3 raw detection array.")
    return values[0]


def _resolve_torch_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("YOLO11n runner supports only cpu and cuda:0.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return resolved


def _resolve_ort_device(device: str) -> tuple[str, list[str]]:
    if device == "cpu":
        return "cpu", [CPU_PROVIDER]
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("YOLO11n runner supports only cpu and cuda:0.")
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
