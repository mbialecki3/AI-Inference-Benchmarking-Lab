"""Export a registered detector checkpoint to a static raw-output ONNX artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx

from inference_bench.detection import (
    YOLO11N,
    YOLO11S,
    available_detection_models,
    get_detection_model_spec,
    load_detection_model,
)


@dataclass(frozen=True, slots=True)
class DetectionExportResult:
    model_name: str
    output_path: Path
    input_shape: tuple[int, ...]
    output_shape: tuple[int | str, ...]
    artifact_size_bytes: int

    def summary(self) -> dict[str, object]:
        summary = asdict(self)
        summary["output_path"] = str(self.output_path)
        summary["input_shape"] = list(self.input_shape)
        summary["output_shape"] = list(self.output_shape)
        return summary


def export_detection_onnx(
    model_name: str,
    weights: Path | str,
    output_path: Path | str,
) -> DetectionExportResult:
    """Export one registered detector with a static raw, pre-NMS output."""

    spec = get_detection_model_spec(model_name)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec.name not in {YOLO11N, YOLO11S}:
        raise RuntimeError(f"No ONNX exporter is registered for {spec.name}.")
    exported = Path(
        load_detection_model(spec.name, weights).export(
            format="onnx",
            imgsz=spec.input_shape[-1],
            batch=spec.input_shape[0],
            dynamic=False,
            simplify=False,
            opset=18,
            nms=False,
        )
    )
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics did not create the requested ONNX artifact: {exported}")
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    model = onnx.load(str(destination))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"{spec.name} ONNX export must expose exactly one input and one raw output.")
    input_shape = tuple(item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim)
    output_shape = tuple(
        item.dim_value if item.dim_value else item.dim_param
        for item in model.graph.output[0].type.tensor_type.shape.dim
    )
    if input_shape != spec.input_shape:
        raise ValueError(f"Expected static {spec.name} input {spec.input_shape}, got {input_shape}.")
    if output_shape != spec.output_shape:
        raise ValueError(
            f"Expected static {spec.name} raw output {spec.output_shape}, got {output_shape}."
        )
    return DetectionExportResult(spec.name, destination, input_shape, output_shape, destination.stat().st_size)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a registered detector to static raw-output ONNX.")
    parser.add_argument("--model", choices=available_detection_models(), default="yolo11n")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    spec = get_detection_model_spec(arguments.model)
    arguments.weights = arguments.weights or spec.weights_path
    arguments.output = arguments.output or spec.onnx_path
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    print(json.dumps(export_detection_onnx(arguments.model, arguments.weights, arguments.output).summary(), indent=2))


if __name__ == "__main__":
    main()
