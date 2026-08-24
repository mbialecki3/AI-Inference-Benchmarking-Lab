"""Shared contracts for the initial YOLO11n detection benchmark slice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference_bench.benchmark_result import OutputParity


YOLO11N = "yolo11n"
YOLO11N_INPUT_SHAPE = (1, 3, 640, 640)
YOLO11N_OUTPUT_SHAPE = (1, 84, 8400)
YOLO11N_WEIGHTS = Path("artifacts/yolo11n.pt")
YOLO11N_ONNX = Path("artifacts/yolo11n.onnx")


@dataclass(frozen=True, slots=True)
class DetectionLayout:
    """Serializable raw-tensor layout needed to compare one detector safely.

    Detector outputs are not interchangeable: this metadata identifies where
    class scores reside without assuming a particular model family. A future
    detector can provide a different instance while preserving the common
    benchmark-record and parity interfaces.
    """

    output: str
    box_coordinate_channels: int
    class_channel_axis: int
    candidate_axis: int
    class_channel_start: int
    class_count: int

    def __post_init__(self) -> None:
        if self.box_coordinate_channels < 0:
            raise ValueError("box_coordinate_channels must be non-negative.")
        if self.class_channel_start < self.box_coordinate_channels:
            raise ValueError("class_channel_start cannot precede box coordinates.")
        if self.class_count <= 0:
            raise ValueError("class_count must be positive.")
        if self.class_channel_axis == self.candidate_axis:
            raise ValueError("class and candidate axes must be distinct.")

    def benchmark_metadata(self) -> dict[str, object]:
        """Return record configuration shared by Python and native runners."""

        return {
            "task": "detection",
            "output": self.output,
            "box_coordinate_channels": self.box_coordinate_channels,
            "class_channel_axis": self.class_channel_axis,
            "candidate_axis": self.candidate_axis,
            "class_channel_start": self.class_channel_start,
            "class_count": self.class_count,
        }


YOLO11N_LAYOUT = DetectionLayout(
    output="raw_pre_nms",
    box_coordinate_channels=4,
    class_channel_axis=1,
    candidate_axis=2,
    class_channel_start=4,
    class_count=80,
)


def make_detection_input(
    *, batch_size: int | None = None, seed: int = 69420, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Return a repeatable float32 NCHW tensor for raw detection parity."""

    resolved_batch_size = YOLO11N_INPUT_SHAPE[0] if batch_size is None else batch_size
    if isinstance(resolved_batch_size, bool) or not isinstance(resolved_batch_size, int) or resolved_batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(
        (resolved_batch_size, *YOLO11N_INPUT_SHAPE[1:]),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device).contiguous()


def load_yolo11n(weights: Path | str) -> Any:
    """Load an explicit YOLO11n checkpoint without hiding a package dependency."""

    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO11n weights do not exist: {path}. Download the official checkpoint "
            f"to that path before running this benchmark."
        )
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "YOLO11n support requires the pinned 'ultralytics' package. "
            "Create the environment from environment.yml."
        ) from error
    return YOLO(str(path))


def raw_detection_tensor(value: object) -> torch.Tensor:
    """Normalize Ultralytics eager output to its raw pre-NMS prediction tensor."""

    candidate = value[0] if isinstance(value, tuple) else value
    if not isinstance(candidate, torch.Tensor) or candidate.ndim != 3:
        raise TypeError("YOLO11n must return one rank-3 raw detection tensor.")
    return candidate


def compare_detection_outputs(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    layout: DetectionLayout = YOLO11N_LAYOUT,
) -> OutputParity:
    """Compare raw YOLO tensors and class selection at every prediction location.

    ``prediction_agreement`` is the fraction of matching highest-scoring class
    indices per candidate location, using the supplied raw-tensor layout. It is
    a runner-parity signal, not mAP.
    """

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError("Detection outputs must have identical shapes.")
    if reference_array.ndim != 3:
        raise ValueError("Detection outputs must be rank-3 arrays.")
    class_axis = _normalize_axis(layout.class_channel_axis, reference_array.ndim, "class_channel_axis")
    candidate_axis = _normalize_axis(layout.candidate_axis, reference_array.ndim, "candidate_axis")
    if class_axis == candidate_axis:
        raise ValueError("class and candidate axes must resolve to distinct output axes.")
    class_stop = layout.class_channel_start + layout.class_count
    if reference_array.shape[class_axis] < class_stop:
        raise ValueError(
            "Detection output does not contain the configured class-channel range "
            f"[{layout.class_channel_start}, {class_stop})."
        )
    absolute_error = np.abs(candidate_array - reference_array)
    relative_error = absolute_error / np.maximum(np.abs(reference_array), 1e-12)
    class_selector = [slice(None)] * reference_array.ndim
    class_selector[class_axis] = slice(layout.class_channel_start, class_stop)
    reference_classes = np.argmax(reference_array[tuple(class_selector)], axis=class_axis)
    candidate_classes = np.argmax(candidate_array[tuple(class_selector)], axis=class_axis)
    return OutputParity(
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        prediction_agreement=float(np.mean(reference_classes == candidate_classes)),
    )


def _normalize_axis(axis: int, rank: int, name: str) -> int:
    """Return a valid non-negative output axis for layout validation."""

    if not -rank <= axis < rank:
        raise ValueError(f"{name} {axis} is invalid for rank-{rank} detection output.")
    return axis % rank
