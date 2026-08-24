"""Export the explicit YOLO11n detection checkpoint to a static ONNX artifact."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx

from inference_bench.detection import (
    YOLO11N,
    YOLO11N_INPUT_SHAPE,
    YOLO11N_ONNX,
    YOLO11N_OUTPUT_SHAPE,
    YOLO11N_WEIGHTS,
    load_yolo11n,
)


@dataclass(frozen=True, slots=True)
class YoloExportResult:
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


def export_yolo11n(weights: Path | str, output_path: Path | str) -> YoloExportResult:
    """Export YOLO11n with a static 640-square raw (no-NMS) detection output."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(
        load_yolo11n(weights).export(
            format="onnx", imgsz=640, batch=1, dynamic=False, simplify=False, opset=18, nms=False
        )
    )
    if not exported.is_file():
        raise RuntimeError(f"Ultralytics did not create the requested ONNX artifact: {exported}")
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
    model = onnx.load(str(destination))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("YOLO11n ONNX export must expose exactly one input and one raw output.")
    input_shape = tuple(item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim)
    output_shape = tuple(
        item.dim_value if item.dim_value else item.dim_param
        for item in model.graph.output[0].type.tensor_type.shape.dim
    )
    if input_shape != YOLO11N_INPUT_SHAPE:
        raise ValueError(f"Expected static YOLO11n input {YOLO11N_INPUT_SHAPE}, got {input_shape}.")
    if output_shape != YOLO11N_OUTPUT_SHAPE:
        raise ValueError(
            f"Expected static YOLO11n raw output {YOLO11N_OUTPUT_SHAPE}, got {output_shape}."
        )
    return YoloExportResult(YOLO11N, destination, input_shape, output_shape, destination.stat().st_size)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11n to static raw-output ONNX.")
    parser.add_argument("--weights", type=Path, default=YOLO11N_WEIGHTS)
    parser.add_argument("--output", type=Path, default=YOLO11N_ONNX)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    print(json.dumps(export_yolo11n(arguments.weights, arguments.output).summary(), indent=2))


if __name__ == "__main__":
    main()
