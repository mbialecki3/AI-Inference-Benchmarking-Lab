"""Shared contracts for the initial YOLO11n detection benchmark slice."""

from __future__ import annotations

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


def compare_detection_outputs(reference: np.ndarray, candidate: np.ndarray) -> OutputParity:
    """Compare raw YOLO tensors and class selection at every prediction location.

    Raw detection output uses channels 0--3 for boxes and the remaining channels
    for COCO class scores. ``prediction_agreement`` is therefore the fraction of
    matching highest-scoring classes per batch/prediction location, not mAP.
    """

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape or reference_array.ndim != 3:
        raise ValueError("Detection outputs must be rank-3 arrays with identical shapes.")
    if reference_array.shape[1] <= 4:
        raise ValueError("Detection output must contain four box and one or more class channels.")
    absolute_error = np.abs(candidate_array - reference_array)
    relative_error = absolute_error / np.maximum(np.abs(reference_array), 1e-12)
    reference_classes = np.argmax(reference_array[:, 4:, :], axis=1)
    candidate_classes = np.argmax(candidate_array[:, 4:, :], axis=1)
    return OutputParity(
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        prediction_agreement=float(np.mean(reference_classes == candidate_classes)),
    )
