"""Deterministic synthetic inputs for export and runner-parity checks.

These tensors are intentionally not a validation dataset.  They prove that
two engines receive exactly the same values; they do not measure model
accuracy on real images.
"""

from __future__ import annotations

import torch

from inference_bench.models import get_model_spec


DEFAULT_INPUT_SEED = 69420


def make_input(
    model_name: str,
    *,
    batch_size: int | None = None,
    seed: int = DEFAULT_INPUT_SEED,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a repeatable float32 NCHW tensor for ``model_name``.

    Random values are generated on the CPU with a private generator.  This
    keeps the global PyTorch random state untouched and means CPU and CUDA
    runners can start with identical tensor values.  The requested device is
    used only after generation.

    Args:
        model_name: A name accepted by :func:`get_model_spec`.
        batch_size: Overrides the model contract's batch dimension.
        seed: Seed for this input only.  The default is deliberately stable.
        device: Destination device, such as ``"cpu"`` or ``"cuda:0"``.
    """

    spec = get_model_spec(model_name)
    resolved_batch_size = spec.input_shape[0] if batch_size is None else batch_size
    if isinstance(resolved_batch_size, bool) or resolved_batch_size <= 0 or not isinstance(resolved_batch_size, int):
        raise ValueError("batch_size must be a positive integer.")

    shape = (resolved_batch_size, *spec.input_shape[1:])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.rand(
        shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return tensor.to(device=device).contiguous()
