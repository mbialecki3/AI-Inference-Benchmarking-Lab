"""Warm-run PyTorch and ONNX Runtime runners for registered raw detectors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from inference_bench.detection import (
    get_detection_model_spec,
    load_detection_model,
    make_detection_input,
    raw_detection_tensor,
)
from inference_bench.runtime_options import OnnxRuntimeOptions
from inference_bench.timing import elapsed_ms, measure_inference


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass(frozen=True, slots=True)
class DetectionRun:
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
    model_load_ms: float = 0.0
    device_latencies_ms: tuple[float, ...] = ()
    runtime_configuration: dict[str, object] | None = None


def run_detection_pytorch(
    model_name: str, weights: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    warmup_iterations: int = 5, timed_iterations: int = 20, precision: str = "fp32",
) -> DetectionRun:
    """Run one registered pretrained detector without post-processing."""

    _validate_counts(warmup_iterations, timed_iterations)
    spec = get_detection_model_spec(model_name)
    resolved_device = _resolve_torch_device(device)
    if precision not in {"fp32", "fp16"} or (precision == "fp16" and resolved_device.type != "cuda"):
        raise ValueError("Detection fp16 is only supported as a CUDA experiment; use fp32 otherwise.")
    def prepare() -> tuple[torch.nn.Module, torch.Tensor]:
        model = load_detection_model(spec.name, weights).model.to(resolved_device).eval()
        inputs = make_detection_input(spec.name, seed=input_seed, device=resolved_device)
        return (model.half(), inputs.half()) if precision == "fp16" else (model, inputs)
    (model, inputs), model_load_ms = elapsed_ms(prepare)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            raw_detection_tensor(model(inputs))
        _synchronize(resolved_device)
        samples: list[float] = []
        device_samples: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(timed_iterations):
            output, host_ms, device_ms = measure_inference(lambda: raw_detection_tensor(model(inputs)), resolved_device)
            _synchronize(resolved_device)
            samples.append(host_ms)
            if device_ms is not None:
                device_samples.append(device_ms)
    if output is None:
        raise RuntimeError(f"{spec.name} did not produce a timed PyTorch output.")
    return DetectionRun("pytorch_eager", Path(weights), str(resolved_device), tuple(inputs.shape), input_seed,
                   warmup_iterations, timed_iterations, output.detach().cpu().numpy().copy(), tuple(samples), (),
                   model_load_ms, tuple(device_samples), {"precision": precision})


def run_detection_onnx(
    model_name: str, model_path: Path | str, *, device: str = "cpu", input_seed: int = 69420,
    warmup_iterations: int = 5, timed_iterations: int = 20, runtime_options: OnnxRuntimeOptions | None = None,
) -> DetectionRun:
    """Run one registered static raw-output ONNX artifact with ONNX Runtime."""

    _validate_counts(warmup_iterations, timed_iterations)
    spec = get_detection_model_spec(model_name)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name} ONNX model does not exist: {path}")
    resolved_device, providers = _resolve_ort_device(device)
    options = runtime_options or OnnxRuntimeOptions()
    session, model_load_ms = elapsed_ms(lambda: ort.InferenceSession(
        str(path), sess_options=options.session_options(), providers=options.providers(resolved_device),
    ))
    active_providers = tuple(session.get_providers())
    if active_providers[0] != providers[0]:
        raise RuntimeError(f"ONNX Runtime did not activate {providers[0]} first: {active_providers!r}")
    inputs_metadata, outputs_metadata = session.get_inputs(), session.get_outputs()
    if len(inputs_metadata) != 1 or len(outputs_metadata) != 1:
        raise ValueError(f"{spec.name} ONNX must expose exactly one input and one raw output.")
    if inputs_metadata[0].name != spec.input_name or outputs_metadata[0].name != spec.output_name:
        raise ValueError(f"{spec.name} ONNX must expose {spec.input_name} -> {spec.output_name}.")
    inputs = make_detection_input(spec.name, seed=input_seed).numpy()
    feeds = {inputs_metadata[0].name: inputs}
    for _ in range(warmup_iterations):
        _onnx_once(session, outputs_metadata[0].name, feeds, spec.output_shape)
    _synchronize(torch.device(resolved_device))
    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        output, host_ms = elapsed_ms(lambda: _synchronized_onnx(session, outputs_metadata[0].name, feeds, spec.output_shape, resolved_device))
        samples.append(host_ms)
    if output is None:
        raise RuntimeError(f"{spec.name} did not produce a timed ONNX output.")
    return DetectionRun("onnxruntime", path, resolved_device, tuple(inputs.shape), input_seed,
                   warmup_iterations, timed_iterations, output, tuple(samples), active_providers,
                   model_load_ms, (), options.summary())


def _onnx_once(
    session: ort.InferenceSession,
    output_name: str,
    feeds: dict[str, np.ndarray],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    values = session.run([output_name], feeds)
    if len(values) != 1 or not isinstance(values[0], np.ndarray) or tuple(values[0].shape) != expected_shape:
        raise TypeError(f"ONNX Runtime must return one raw detection array shaped {expected_shape}.")
    return values[0]


def _synchronized_onnx(session: ort.InferenceSession, output_name: str, feeds: dict[str, np.ndarray], expected_shape: tuple[int, ...], device: str) -> np.ndarray:
    output = _onnx_once(session, output_name, feeds, expected_shape)
    _synchronize(torch.device(device))
    return output


def _resolve_torch_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Detection runners support only cpu and cuda:0.")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return resolved


def _resolve_ort_device(device: str) -> tuple[str, list[str]]:
    if device == "cpu":
        return "cpu", [CPU_PROVIDER]
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("Detection runners support only cpu and cuda:0.")
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
