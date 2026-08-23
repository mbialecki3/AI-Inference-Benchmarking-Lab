"""Export the deterministic PyTorch benchmark contract to ONNX.

This module deliberately exports the same seeded CPU model and static input
shape used by the PyTorch reference runner.  It establishes a portable model
artifact before an ONNX Runtime runner or any performance measurement exists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import torch

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.models import available_models, build_model
from inference_bench.pytorch_runner import DEFAULT_MODEL_SEED


DEFAULT_OPSET_VERSION = 18
INPUT_NAME = "images"
OUTPUT_NAME = "logits"


@dataclass(frozen=True, slots=True)
class OnnxExportResult:
    """Metadata describing one validated ONNX export."""

    model_name: str
    output_path: Path
    input_shape: tuple[int, ...]
    input_seed: int
    model_seed: int
    opset_version: int
    artifact_size_bytes: int

    def summary(self) -> dict[str, object]:
        """Return JSON-friendly export metadata."""

        summary = asdict(self)
        summary["output_path"] = str(self.output_path)
        summary["input_shape"] = list(self.input_shape)
        return summary


def export_onnx(
    model_name: str,
    output_path: Path | str,
    *,
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
    model_seed: int = DEFAULT_MODEL_SEED,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> OnnxExportResult:
    """Export and structurally validate one deterministic benchmark model.

    The first export contract intentionally uses a fixed batch size and image
    resolution.  Dynamic shapes will be introduced only after the equivalent
    static PyTorch/ONNX Runtime parity path is proven.
    """

    if (
        isinstance(opset_version, bool)
        or not isinstance(opset_version, int)
        or opset_version <= 0
    ):
        raise ValueError("opset_version must be a positive integer.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = _build_seeded_model(model_name, model_seed)
    input_tensor = make_input(
        model_name,
        batch_size=batch_size,
        seed=input_seed,
        device="cpu",
    )

    torch.onnx.export(
        model,
        (input_tensor,),
        str(destination),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
    )
    _validate_onnx_model(destination)

    return OnnxExportResult(
        model_name=model_name,
        output_path=destination,
        input_shape=tuple(input_tensor.shape),
        input_seed=input_seed,
        model_seed=model_seed,
        opset_version=opset_version,
        artifact_size_bytes=destination.stat().st_size,
    )


def _build_seeded_model(model_name: str, model_seed: int) -> torch.nn.Module:
    """Create the same deterministic CPU model as the reference runner."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        model = build_model(model_name)
    return model.eval()


def _validate_onnx_model(path: Path) -> None:
    """Reject an invalid artifact before it can enter the benchmark matrix."""

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != [INPUT_NAME]:
        raise ValueError(f"Expected ONNX input {INPUT_NAME!r}, got {input_names!r}.")
    if output_names != [OUTPUT_NAME]:
        raise ValueError(f"Expected ONNX output {OUTPUT_NAME!r}, got {output_names!r}.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a benchmark model to ONNX.")
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument("--output", type=Path, default=Path("artifacts/resnet50.onnx"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION)
    return parser.parse_args()


def main() -> None:
    """Run the command-line exporter and print its artifact metadata."""

    arguments = _parse_arguments()
    result = export_onnx(
        arguments.model,
        arguments.output,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
        model_seed=arguments.model_seed,
        opset_version=arguments.opset,
    )
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
