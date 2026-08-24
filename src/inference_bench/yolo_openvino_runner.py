"""OpenVINO CPU runner for the raw pre-NMS YOLO11n ONNX contract."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.properties import hint

from inference_bench.detection import YOLO11N_INPUT_SHAPE, YOLO11N_OUTPUT_SHAPE, make_detection_input
from inference_bench.openvino_runner import CPU_DEVICE, get_openvino_core
from inference_bench.yolo_runner import YoloRun, _validate_counts


def run_yolo_openvino(
    model_path: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
) -> YoloRun:
    """Run static YOLO11n ONNX on OpenVINO CPU without post-processing.

    Model loading and compilation are outside the timed region. The recorded
    samples therefore contain synchronous raw-tensor inference only, matching
    the PyTorch and ONNX Runtime detection runners.
    """

    _validate_counts(warmup_iterations, timed_iterations)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO11n ONNX model does not exist: {path}")

    core = get_openvino_core()
    _validate_cpu_device(device, core)
    model = core.read_model(str(path))
    input_name, output_name = _validate_model_interface(model)
    compiled_model = core.compile_model(
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

    inputs = make_detection_input(seed=input_seed).numpy()
    for _ in range(warmup_iterations):
        _run_once(compiled_model, input_name, output_name, inputs)

    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        started_at = time.perf_counter_ns()
        output = _run_once(compiled_model, input_name, output_name, inputs)
        samples.append((time.perf_counter_ns() - started_at) / 1_000_000)

    if output is None:
        raise RuntimeError("YOLO11n did not produce a timed OpenVINO output.")
    return YoloRun(
        "openvino",
        path,
        "cpu",
        tuple(inputs.shape),
        input_seed,
        warmup_iterations,
        timed_iterations,
        output.copy(),
        tuple(samples),
        active_devices,
    )


def _validate_cpu_device(device: str, core: ov.Core) -> None:
    if device != "cpu":
        raise ValueError("The initial YOLO11n OpenVINO runner supports only cpu.")
    if CPU_DEVICE not in core.available_devices:
        raise RuntimeError("OpenVINO does not expose a CPU execution device.")


def _validate_model_interface(model: ov.Model) -> tuple[str, str]:
    """Validate the static raw YOLO tensor contract without fixing port names."""

    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("YOLO11n ONNX must expose exactly one input and one raw output.")
    input_port, output_port = model.input(0), model.output(0)
    if tuple(input_port.shape) != YOLO11N_INPUT_SHAPE:
        raise ValueError(
            f"Expected YOLO11n input shape {YOLO11N_INPUT_SHAPE}, got {tuple(input_port.shape)}."
        )
    if tuple(output_port.shape) != YOLO11N_OUTPUT_SHAPE:
        raise ValueError(
            f"Expected YOLO11n raw output shape {YOLO11N_OUTPUT_SHAPE}, got {tuple(output_port.shape)}."
        )
    if input_port.get_element_type() != ov.Type.f32 or output_port.get_element_type() != ov.Type.f32:
        raise ValueError("YOLO11n OpenVINO ports must use float32 tensors.")
    return input_port.get_any_name(), output_port.get_any_name()


def _run_once(
    compiled_model: ov.CompiledModel,
    input_name: str,
    output_name: str,
    inputs: np.ndarray,
) -> np.ndarray:
    outputs = compiled_model({input_name: inputs})
    output = outputs[compiled_model.output(output_name)]
    if not isinstance(output, np.ndarray) or tuple(output.shape) != YOLO11N_OUTPUT_SHAPE:
        raise TypeError("YOLO11n OpenVINO must return one float32 raw detection array.")
    if output.dtype != np.float32:
        raise TypeError("YOLO11n OpenVINO must return float32 raw detection values.")
    return output
