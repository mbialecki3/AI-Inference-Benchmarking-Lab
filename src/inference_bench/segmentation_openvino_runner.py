"""OpenVINO CPU runner for registered static segmentation models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov

from inference_bench.openvino_runner import CPU_DEVICE, get_openvino_core
from inference_bench.segmentation import SegmentationModelSpec, get_segmentation_model_spec, make_segmentation_input
from inference_bench.segmentation_runner import SegmentationRun, _validate_counts
from inference_bench.runtime_options import OpenVinoOptions
from inference_bench.timing import elapsed_ms


def run_segmentation_openvino(
    model_name: str,
    model_path: Path | str,
    *,
    device: str = "cpu",
    input_seed: int = 69420,
    model_seed: int = 67,
    warmup_iterations: int = 5,
    timed_iterations: int = 20,
    runtime_options: OpenVinoOptions | None = None,
) -> SegmentationRun:
    """Run static raw segmentation logits on OpenVINO CPU.

    Compilation stays outside the timed region. Samples include only synchronous
    inference, matching the PyTorch and ONNX Runtime segmentation runners.
    """

    _validate_counts(warmup_iterations, timed_iterations)
    spec = get_segmentation_model_spec(model_name)
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"{spec.name} ONNX model does not exist: {path}")

    core = get_openvino_core()
    _validate_cpu_device(device, core)
    options = runtime_options or OpenVinoOptions()
    def load_and_compile() -> tuple[ov.CompiledModel, str, str]:
        model = core.read_model(str(path))
        input_name, output_name = _validate_model_interface(model, spec)
        return core.compile_model(model, CPU_DEVICE, options.compile_configuration()), input_name, output_name
    (compiled_model, input_name, output_name), model_load_ms = elapsed_ms(load_and_compile)
    active_devices = tuple(str(item) for item in compiled_model.get_property("EXECUTION_DEVICES"))
    if CPU_DEVICE not in active_devices:
        raise RuntimeError(f"OpenVINO did not report CPU as active: {active_devices!r}")

    inputs = make_segmentation_input(spec.name, seed=input_seed).numpy()
    for _ in range(warmup_iterations):
        _run_once(compiled_model, input_name, output_name, inputs, spec)

    samples: list[float] = []
    output: np.ndarray | None = None
    for _ in range(timed_iterations):
        output, host_ms = elapsed_ms(lambda: _run_once(compiled_model, input_name, output_name, inputs, spec))
        samples.append(host_ms)
    if output is None:
        raise RuntimeError(f"{spec.name} did not produce a timed OpenVINO output.")
    return SegmentationRun(
        "openvino", path, "cpu", tuple(inputs.shape), input_seed, model_seed,
        warmup_iterations, timed_iterations, output.copy(), tuple(samples), active_devices,
        model_load_ms, (), options.summary(),
    )


def _validate_cpu_device(device: str, core: ov.Core) -> None:
    if device != "cpu":
        raise ValueError("The segmentation OpenVINO runner supports only cpu.")
    if CPU_DEVICE not in core.available_devices:
        raise RuntimeError("OpenVINO does not expose a CPU execution device.")


def _validate_model_interface(model: ov.Model, spec: SegmentationModelSpec) -> tuple[str, str]:
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(f"{spec.name} ONNX must expose exactly one input and one raw output.")
    input_port, output_port = model.input(0), model.output(0)
    if tuple(input_port.shape) != spec.input_shape or tuple(output_port.shape) != spec.output_shape:
        raise ValueError(f"{spec.name} must use {spec.input_shape} -> {spec.output_shape} static shapes.")
    if input_port.get_element_type() != ov.Type.f32 or output_port.get_element_type() != ov.Type.f32:
        raise ValueError(f"{spec.name} OpenVINO ports must use float32 tensors.")
    return input_port.get_any_name(), output_port.get_any_name()


def _run_once(
    compiled_model: ov.CompiledModel,
    input_name: str,
    output_name: str,
    inputs: np.ndarray,
    spec: SegmentationModelSpec,
) -> np.ndarray:
    outputs = compiled_model({input_name: inputs})
    output = outputs[compiled_model.output(output_name)]
    if not isinstance(output, np.ndarray) or tuple(output.shape) != spec.output_shape:
        raise TypeError(f"{spec.name} OpenVINO must return one float32 raw segmentation array.")
    if output.dtype != np.float32:
        raise TypeError(f"{spec.name} OpenVINO must return float32 raw segmentation values.")
    return output
