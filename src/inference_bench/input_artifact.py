"""Create byte-identical synthetic inputs for native benchmark runners.

The Python reference runners construct inputs with PyTorch's private CPU
generator. Native runners deliberately consume this little-endian float32
binary representation instead of attempting to replicate PyTorch's RNG.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input
from inference_bench.models import available_models


@dataclass(frozen=True, slots=True)
class InputArtifact:
    """Metadata for a portable raw float32 NCHW input artifact."""

    model_name: str
    output_path: Path
    input_shape: tuple[int, ...]
    input_seed: int
    dtype: str
    size_bytes: int

    def summary(self) -> dict[str, object]:
        """Return the cross-language input contract as JSON-friendly metadata."""

        return {
            "model": self.model_name,
            "output_path": str(self.output_path),
            "input_shape": list(self.input_shape),
            "input_seed": self.input_seed,
            "dtype": self.dtype,
            "byte_order": "little-endian",
            "size_bytes": self.size_bytes,
        }


def export_input_artifact(
    model_name: str,
    output_path: Path | str,
    *,
    batch_size: int | None = None,
    input_seed: int = DEFAULT_INPUT_SEED,
) -> InputArtifact:
    """Write the project's seeded input tensor as little-endian float32 bytes.

    This is a benchmark input, not a dataset sample. Its purpose is to let
    Python and C++ runners receive precisely the same values for parity work.
    """

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor = make_input(
        model_name,
        batch_size=batch_size,
        seed=input_seed,
        device="cpu",
    )
    values = tensor.numpy()
    # ONNX Runtime's initial C++ runner accepts a simple dependency-free binary
    # format. Explicit conversion also makes byte order stable across hosts.
    little_endian_values = np.ascontiguousarray(values, dtype="<f4")
    little_endian_values.tofile(destination)
    return InputArtifact(
        model_name=model_name,
        output_path=destination,
        input_shape=tuple(little_endian_values.shape),
        input_seed=input_seed,
        dtype="float32",
        size_bytes=destination.stat().st_size,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a deterministic float32 NCHW input binary for native runners."
    )
    parser.add_argument("--model", choices=available_models(), default="resnet50")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/inputs/resnet50_seed69420_f32_nchw.bin"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_INPUT_SEED)
    return parser.parse_args()


def main() -> None:
    """Create a native-runner input artifact and print its contract metadata."""

    arguments = _parse_arguments()
    artifact = export_input_artifact(
        arguments.model,
        arguments.output,
        batch_size=arguments.batch_size,
        input_seed=arguments.input_seed,
    )
    print(json.dumps(artifact.summary(), indent=2))


if __name__ == "__main__":
    main()
