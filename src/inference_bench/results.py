"""Durable JSON persistence for benchmark records."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from inference_bench.benchmark_result import BenchmarkResult


def save_result(result: BenchmarkResult, output_directory: Path | str = "results") -> Path:
    """Atomically write one record, returning its collision-resistant path."""

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.created_at_utc.replace(':', '-')}--{result.run_id}.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".pending-",
        suffix=".json",
        delete=False,
    ) as temporary_file:
        json.dump(result.summary(), temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)
    return path
