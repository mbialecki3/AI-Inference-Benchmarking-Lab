"""Shared calculations for warm-run inference measurements.

Keeping these calculations separate from an inference engine prevents small
differences in percentile or throughput math from making result files
incomparable.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    """Summary statistics and source samples for one warm benchmark run."""

    samples_ms: tuple[float, ...]
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_samples(cls, samples_ms: tuple[float, ...] | list[float]) -> "LatencyMetrics":
        """Validate samples and calculate a consistent latency summary."""

        samples = tuple(float(sample) for sample in samples_ms)
        if not samples:
            raise ValueError("At least one latency sample is required.")
        if any(not math.isfinite(sample) or sample <= 0 for sample in samples):
            raise ValueError("Latency samples must be finite positive numbers.")

        return cls(
            samples_ms=samples,
            mean_ms=statistics.fmean(samples),
            min_ms=min(samples),
            max_ms=max(samples),
            p50_ms=percentile(samples, 50),
            p95_ms=percentile(samples, 95),
            p99_ms=percentile(samples, 99),
        )

    def summary(self) -> dict[str, float]:
        """Return JSON-friendly latency summaries, excluding raw samples."""

        return {
            "mean": self.mean_ms,
            "min": self.min_ms,
            "max": self.max_ms,
            "p50": self.p50_ms,
            "p95": self.p95_ms,
            "p99": self.p99_ms,
        }


def percentile(samples: tuple[float, ...] | list[float], percentile_value: float) -> float:
    """Calculate a linearly interpolated percentile without another dependency."""

    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100.")
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        raise ValueError("At least one sample is required.")

    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def throughput_samples_per_second(batch_size: int, mean_latency_ms: float) -> float:
    """Return completed input samples per second for one explicit batch size."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not math.isfinite(mean_latency_ms) or mean_latency_ms <= 0:
        raise ValueError("mean_latency_ms must be a finite positive number.")
    return batch_size * 1_000 / mean_latency_ms
