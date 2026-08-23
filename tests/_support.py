"""Small helpers that keep successful regression-test output readable."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from inference_bench.onnx_export import OnnxExportResult, export_onnx


def export_onnx_quietly(
    model_name: str,
    output_path: Path,
) -> OnnxExportResult:
    """Export quietly, but reveal PyTorch diagnostics if the export fails."""

    captured_output = io.StringIO()
    try:
        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            return export_onnx(model_name, output_path)
    except Exception:
        sys.stderr.write(captured_output.getvalue())
        raise
