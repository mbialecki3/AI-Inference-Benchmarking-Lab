"""Static contracts and parity helpers for dense semantic segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from inference_bench.benchmark_result import OutputParity


DEEPLABV3_RESNET50 = "deeplabv3_resnet50"
DEEPLABV3_RESNET50_INPUT_SHAPE = (1, 3, 224, 224)
DEEPLABV3_RESNET50_OUTPUT_SHAPE = (1, 21, 224, 224)
DEEPLABV3_RESNET50_ONNX = Path("artifacts/deeplabv3_resnet50.onnx")


@dataclass(frozen=True, slots=True)
class SegmentationModelSpec:
    """Static raw-logit contract for one semantic-segmentation model."""

    name: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    class_channel_axis: int
    class_count: int
    onnx_path: Path

    def benchmark_metadata(self) -> dict[str, object]:
        return {
            "task": "semantic_segmentation",
            "output": "raw_logits",
            "class_channel_axis": self.class_channel_axis,
            "class_count": self.class_count,
        }


DEEPLABV3_RESNET50_SPEC = SegmentationModelSpec(
    name=DEEPLABV3_RESNET50,
    input_name="images",
    output_name="logits",
    input_shape=DEEPLABV3_RESNET50_INPUT_SHAPE,
    output_shape=DEEPLABV3_RESNET50_OUTPUT_SHAPE,
    class_channel_axis=1,
    class_count=21,
    onnx_path=DEEPLABV3_RESNET50_ONNX,
)
_SEGMENTATION_MODELS = {DEEPLABV3_RESNET50: DEEPLABV3_RESNET50_SPEC}


def available_segmentation_models() -> tuple[str, ...]:
    """Return segmentation models accepted by the dedicated runner path."""

    return tuple(_SEGMENTATION_MODELS)


def get_segmentation_model_spec(model_name: str) -> SegmentationModelSpec:
    """Look up a segmentation contract with an actionable error."""

    try:
        return _SEGMENTATION_MODELS[model_name]
    except KeyError as error:
        supported = ", ".join(available_segmentation_models())
        raise ValueError(f"Unsupported segmentation model {model_name!r}. Supported: {supported}.") from error


def build_segmentation_model(model_name: str) -> torch.nn.Module:
    """Build one untrained segmentation model without implicit downloads."""

    spec = get_segmentation_model_spec(model_name)
    if spec.name != DEEPLABV3_RESNET50:
        raise RuntimeError(f"No eager model factory is registered for {spec.name}.")
    return deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)


def make_segmentation_input(
    model_name: str,
    *,
    batch_size: int | None = None,
    seed: int = 69420,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a repeatable float32 NCHW input for one segmentation contract."""

    spec = get_segmentation_model_spec(model_name)
    resolved_batch_size = spec.input_shape[0] if batch_size is None else batch_size
    if isinstance(resolved_batch_size, bool) or not isinstance(resolved_batch_size, int) or resolved_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(
        (resolved_batch_size, *spec.input_shape[1:]), generator=generator, dtype=torch.float32
    ).to(device=device).contiguous()


def raw_segmentation_tensor(value: object) -> torch.Tensor:
    """Normalize TorchVision's segmentation result to its primary raw-logit tensor."""

    candidate = value.get("out") if isinstance(value, dict) else value
    if not isinstance(candidate, torch.Tensor) or candidate.ndim != 4:
        raise TypeError("The segmentation model must return one rank-4 raw-logit tensor.")
    return candidate


def compare_segmentation_outputs(reference: np.ndarray, candidate: np.ndarray, *, spec: SegmentationModelSpec) -> OutputParity:
    """Compare dense raw logits and per-pixel winning classes, not mIoU."""

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape or tuple(reference_array.shape) != spec.output_shape:
        raise ValueError(f"Segmentation outputs must share static shape {spec.output_shape}.")
    absolute_error = np.abs(candidate_array - reference_array)
    relative_error = absolute_error / np.maximum(np.abs(reference_array), 1e-12)
    reference_classes = np.argmax(reference_array, axis=spec.class_channel_axis)
    candidate_classes = np.argmax(candidate_array, axis=spec.class_channel_axis)
    return OutputParity(
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        prediction_agreement=float(np.mean(reference_classes == candidate_classes)),
    )
