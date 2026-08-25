"""Small, explicit helpers for non-overlapping benchmark timing scopes."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import torch


T = TypeVar("T")


def elapsed_ms(operation: Callable[[], T]) -> tuple[T, float]:
    """Run an operation once and return its host elapsed time in milliseconds."""

    started_ns = time.perf_counter_ns()
    value = operation()
    return value, (time.perf_counter_ns() - started_ns) / 1_000_000


def measure_inference(operation: Callable[[], T], device: torch.device) -> tuple[T, float, float | None]:
    """Return host end-to-end and CUDA-event timing from the same inference.

    Callers keep input allocation outside this helper.  A CUDA result therefore
    measures work enqueued by the inference call rather than Python setup.
    """

    if device.type != "cuda":
        value, host_ms = elapsed_ms(operation)
        return value, host_ms, None
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started_ns = time.perf_counter_ns()
    started.record(torch.cuda.current_stream(device))
    value = operation()
    finished.record(torch.cuda.current_stream(device))
    finished.synchronize()
    return value, (time.perf_counter_ns() - started_ns) / 1_000_000, float(started.elapsed_time(finished))
