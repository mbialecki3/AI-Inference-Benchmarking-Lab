"""Export registered segmentation models to static raw-logit ONNX artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import torch

from inference_bench.segmentation import (
    available_segmentation_models,
    build_segmentation_model,
    get_segmentation_model_spec,
    make_segmentation_input,
    raw_segmentation_tensor,
)


@dataclass(frozen=True, slots=True)
class SegmentationExportResult:
    model_name: str
    output_path: Path
    input_shape: tuple[int, ...]
    output_shape: tuple[int | str, ...]
    input_seed: int
    model_seed: int
    opset_version: int
    artifact_size_bytes: int

    def summary(self) -> dict[str, object]:
        summary = asdict(self)
        summary["output_path"] = str(self.output_path)
        summary["input_shape"] = list(self.input_shape)
        summary["output_shape"] = list(self.output_shape)
        return summary


class _RawSegmentationOutput(torch.nn.Module):
    """Expose TorchVision's primary segmentation tensor as ONNX's sole output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return raw_segmentation_tensor(self.model(images))


def export_segmentation_onnx(
    model_name: str,
    output_path: Path | str,
    *,
    input_seed: int = 69420,
    model_seed: int = 67,
    opset_version: int = 18,
) -> SegmentationExportResult:
    """Export a seeded static segmentation model and validate its I/O contract."""

    if isinstance(opset_version, bool) or not isinstance(opset_version, int) or opset_version <= 0:
        raise ValueError("opset_version must be a positive integer.")
    spec = get_segmentation_model_spec(model_name)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = _RawSegmentationOutput(build_segmentation_model(spec.name)).eval()
    inputs = make_segmentation_input(spec.name, seed=input_seed)
    torch.onnx.export(
        model,
        (inputs,),
        str(destination),
        input_names=[spec.input_name],
        output_names=[spec.output_name],
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )
    model_proto = onnx.load(str(destination))
    onnx.checker.check_model(model_proto)
    input_names = [value.name for value in model_proto.graph.input]
    output_names = [value.name for value in model_proto.graph.output]
    if input_names != [spec.input_name] or output_names != [spec.output_name]:
        raise ValueError(f"Expected ONNX interface {spec.input_name} -> {spec.output_name}.")
    input_shape = tuple(item.dim_value for item in model_proto.graph.input[0].type.tensor_type.shape.dim)
    output_shape = tuple(item.dim_value for item in model_proto.graph.output[0].type.tensor_type.shape.dim)
    if input_shape != spec.input_shape or output_shape != spec.output_shape:
        raise ValueError(f"ONNX shapes must be {spec.input_shape} -> {spec.output_shape}.")
    return SegmentationExportResult(
        spec.name, destination, input_shape, output_shape, input_seed, model_seed, opset_version, destination.stat().st_size
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a registered segmentation model to static ONNX.")
    parser.add_argument("--model", choices=available_segmentation_models(), default="deeplabv3_resnet50")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-seed", type=int, default=69420)
    parser.add_argument("--model-seed", type=int, default=67)
    parser.add_argument("--opset", type=int, default=18)
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = Path("artifacts") / f"{arguments.model}.onnx"
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    result = export_segmentation_onnx(
        arguments.model, arguments.output, input_seed=arguments.input_seed,
        model_seed=arguments.model_seed, opset_version=arguments.opset,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
