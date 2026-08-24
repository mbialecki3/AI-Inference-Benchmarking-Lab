"""Best-effort host, package, and GPU metadata probes for benchmark records."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_environment(project_root: Path | str = ".") -> dict[str, Any]:
    """Capture reproducibility metadata without making a benchmark depend on it.

    A missing package, Git checkout, or NVIDIA utility is represented as data,
    not raised as an error. This lets CPU-only systems create valid records.
    """

    root = Path(project_root)
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            "torch": _package_version("torch"),
            "torchvision": _package_version("torchvision"),
            "onnx": _package_version("onnx"),
            # The GPU wheel exports the ``onnxruntime`` module but publishes
            # itself as ``onnxruntime-gpu``.
            "onnxruntime": _package_version("onnxruntime")
            or _package_version("onnxruntime-gpu"),
            "openvino": _package_version("openvino"),
            "ultralytics": _package_version("ultralytics"),
        },
        "torch": _torch_metadata(),
        "onnxruntime": _onnxruntime_metadata(),
        "git_revision": _git_revision(root),
    }


def process_rss_bytes() -> dict[str, object]:
    """Return the process's maximum resident set size when the OS exposes it."""

    try:
        import resource
    except ImportError:
        return {"status": "unavailable", "reason": "resource module is unavailable"}

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. The benchmark's primary target is
    # Linux, but normalizing makes records portable.
    multiplier = 1 if platform.system() == "Darwin" else 1_024
    return {"status": "available", "value": int(value) * multiplier, "unit": "bytes"}


def sample_gpu_telemetry() -> dict[str, object]:
    """Read NVIDIA telemetry with ``nvidia-smi`` when it is available.

    The probe has a short timeout and never turns missing telemetry into a
    benchmark failure. A caller should capture samples before and after a run.
    """

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi was not found"}

    command = [
        executable,
        "--query-gpu=name,driver_version,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": str(error)}

    gpus: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 5:
            return {"status": "unavailable", "reason": "unexpected nvidia-smi output"}
        gpus.append(
            {
                "name": columns[0],
                "driver_version": columns[1],
                "memory_used_mib": _optional_float(columns[2]),
                "utilization_percent": _optional_float(columns[3]),
                "power_watts": _optional_float(columns[4]),
            }
        )
    return {"status": "available", "gpus": gpus}


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_metadata() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"status": "unavailable"}

    return {
        "status": "available",
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }


def _onnxruntime_metadata() -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "unavailable"}

    return {
        "status": "available",
        "version": ort.__version__,
        "available_providers": list(ort.get_available_providers()),
    }


def _git_revision(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
