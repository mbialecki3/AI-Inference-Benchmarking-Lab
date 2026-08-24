"""Conventions for model- and device-scoped benchmark output directories."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path


def device_directory_name(device: str) -> str:
    """Return a portable directory component for a requested execution device."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", device.strip()).strip("._")
    if not normalized:
        raise ValueError("device must contain at least one directory-safe character.")
    return normalized


def default_results_directory(model_name: str, device: str) -> Path:
    """Return the default location for one model/device benchmark matrix cell."""

    return Path("results") / model_name / device_directory_name(device)


def default_reports_directory(inputs: Iterable[Path | str]) -> Path:
    """Infer a sibling reports directory for one ``results/<model>/<device>`` input.

    Aggregate inputs retain the historical ``reports/`` default. An explicit
    reporting ``--output-dir`` always takes precedence over this helper.
    """

    input_paths = tuple(Path(path) for path in inputs)
    if len(input_paths) != 1:
        return Path("reports")

    candidate = input_paths[0]
    directory = candidate.parent if candidate.suffix == ".json" else candidate
    parts = directory.parts
    results_index = len(parts) - 3
    if results_index < 0 or parts[results_index] != "results":
        return Path("reports")

    return Path(*parts[:results_index]) / "reports" / parts[-2] / parts[-1]
