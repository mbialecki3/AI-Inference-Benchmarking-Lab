"""Versioned, engine-neutral records for reproducible benchmark measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from inference_bench.metrics import LatencyMetrics, throughput_samples_per_second


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OutputParity:
    """Numerical and predicted-class agreement with a PyTorch reference."""

    max_absolute_error: float
    max_relative_error: float
    prediction_agreement: float

    def summary(self) -> dict[str, float]:
        return {
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "prediction_agreement": self.prediction_agreement,
        }


def compare_outputs(reference: np.ndarray, candidate: np.ndarray) -> OutputParity:
    """Compare raw outputs without claiming dataset-level task accuracy."""

    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "Cannot compare outputs with different shapes: "
            f"{reference_array.shape!r} and {candidate_array.shape!r}."
        )
    if reference_array.ndim < 1:
        raise ValueError("Output tensors must include a batch dimension.")

    absolute_error = np.abs(candidate_array - reference_array)
    relative_error = absolute_error / np.maximum(np.abs(reference_array), 1e-12)
    return OutputParity(
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        prediction_agreement=float(
            np.mean(np.argmax(candidate_array, axis=-1) == np.argmax(reference_array, axis=-1))
        ),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """A complete, JSON-serializable record of one warm benchmark run."""

    run_id: str
    created_at_utc: str
    engine: str
    model_name: str
    device: str
    input_shape: tuple[int, ...]
    input_seed: int
    model_seed: int | None
    warmup_iterations: int
    timed_iterations: int
    active_providers: tuple[str, ...]
    artifact_path: Path | None
    artifact_size_bytes: int | None
    latency: LatencyMetrics
    process_rss: dict[str, object]
    environment: dict[str, Any]
    gpu_telemetry_before: dict[str, object]
    gpu_telemetry_after: dict[str, object]
    engine_configuration: dict[str, Any] = field(default_factory=dict)
    parity: OutputParity | None = None
    cold_start_model_load_ms: float | None = None
    device_latency: LatencyMetrics | None = None

    @classmethod
    def create(
        cls,
        *,
        engine: str,
        model_name: str,
        device: str,
        input_shape: tuple[int, ...],
        input_seed: int,
        model_seed: int | None,
        warmup_iterations: int,
        timed_iterations: int,
        active_providers: tuple[str, ...] = (),
        artifact_path: Path | None = None,
        latency_samples_ms: tuple[float, ...],
        process_rss: dict[str, object],
        environment: dict[str, Any],
        gpu_telemetry_before: dict[str, object],
        gpu_telemetry_after: dict[str, object],
        engine_configuration: Mapping[str, Any] | None = None,
        parity: OutputParity | None = None,
        cold_start_model_load_ms: float | None = None,
        device_latency_samples_ms: tuple[float, ...] = (),
    ) -> "BenchmarkResult":
        """Create a result with a stable id, UTC timestamp, and derived metrics."""

        latency = LatencyMetrics.from_samples(latency_samples_ms)
        if cold_start_model_load_ms is not None and (
            not np.isfinite(cold_start_model_load_ms) or cold_start_model_load_ms < 0
        ):
            raise ValueError("cold_start_model_load_ms must be a finite non-negative value.")
        resolved_artifact = Path(artifact_path) if artifact_path is not None else None
        return cls(
            run_id=str(uuid4()),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            engine=engine,
            model_name=model_name,
            device=device,
            input_shape=input_shape,
            input_seed=input_seed,
            model_seed=model_seed,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
            active_providers=active_providers,
            artifact_path=resolved_artifact,
            artifact_size_bytes=(
                resolved_artifact.stat().st_size if resolved_artifact is not None else None
            ),
            latency=latency,
            process_rss=process_rss,
            environment=environment,
            gpu_telemetry_before=gpu_telemetry_before,
            gpu_telemetry_after=gpu_telemetry_after,
            engine_configuration=dict(engine_configuration or {}),
            parity=parity,
            cold_start_model_load_ms=cold_start_model_load_ms,
            device_latency=(
                LatencyMetrics.from_samples(device_latency_samples_ms)
                if device_latency_samples_ms else None
            ),
        )

    def summary(self) -> dict[str, object]:
        """Return a JSON-ready record whose field names form schema version 1."""

        batch_size = self.input_shape[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "runner": {
                "engine": self.engine,
                "device": self.device,
                "active_providers": list(self.active_providers),
                "configuration": self.engine_configuration,
            },
            "model": {
                "name": self.model_name,
                "input_shape": list(self.input_shape),
                "input_seed": self.input_seed,
                "model_seed": self.model_seed,
                "artifact_path": str(self.artifact_path) if self.artifact_path else None,
                "artifact_size_bytes": self.artifact_size_bytes,
            },
            "configuration": {
                "warmup_iterations": self.warmup_iterations,
                "timed_iterations": self.timed_iterations,
            },
            "measurement": {
                "latency_ms": {
                    **self.latency.summary(),
                    "samples": list(self.latency.samples_ms),
                },
                "throughput_samples_per_second": throughput_samples_per_second(
                    batch_size, self.latency.mean_ms
                ),
                "cold_start_model_load_ms": self.cold_start_model_load_ms,
                "device_latency_ms": (
                    {**self.device_latency.summary(), "samples": list(self.device_latency.samples_ms)}
                    if self.device_latency else None
                ),
                "process_rss": self.process_rss,
                "gpu_telemetry": {
                    "before": self.gpu_telemetry_before,
                    "after": self.gpu_telemetry_after,
                },
            },
            "correctness": {"parity": self.parity.summary() if self.parity else None},
            "environment": self.environment,
        }
