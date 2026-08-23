"""Model definitions used by every inference runner.

This module owns *what* model is being tested.  It does not perform inference
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torchvision.models import resnet50


InputShape = tuple[int, int, int, int]
ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """The stable contract for one benchmarkable model.

    ``input_shape`` is in PyTorch's NCHW order: batch, channels, height,
    width.  All initial image-classification models will use float32 tensors
    with this layout.
    """

    name: str
    task: str
    input_shape: InputShape
    output_description: str
    factory: ModelFactory


RESNET50 = ModelSpec(
    name="resnet50",
    task="classification",
    input_shape=(1, 3, 224, 224),
    output_description="float32 logits with shape (batch_size, 1000)",
    # ``weights=None`` prevents an implicit network download.  A later,
    # explicit configuration can request and record pretrained weights.
    factory=lambda: resnet50(weights=None),
)


MODEL_SPECS: dict[str, ModelSpec] = {RESNET50.name: RESNET50}


def available_models() -> tuple[str, ...]:
    """Return model names in a stable order for CLIs and configuration checks."""

    return tuple(sorted(MODEL_SPECS))


def get_model_spec(name: str) -> ModelSpec:
    """Look up a model contract, with a useful error for unsupported names."""

    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(
            f"Unknown model {name!r}. Available models: {choices}."
        ) from error


def build_model(name: str) -> nn.Module:
    """Create an untrained model instance for the named benchmark contract.

    The caller must set the device and call ``model.eval()``.  Keeping those
    actions outside this factory makes device selection and measurement policy
    explicit in each runner.
    """

    return get_model_spec(name).factory()
