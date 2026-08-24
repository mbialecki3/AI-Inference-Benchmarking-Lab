"""Create auditable comparison reports and Matplotlib plots from schema-v1 records."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

# Benchmark reporting runs in WSL and CI without an interactive display server.
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot


SCHEMA_VERSION = 1
EXPECTED_ENGINES = (
    "pytorch_eager",
    "onnxruntime",
    "openvino",
    "onnxruntime_cpp",
)
ENGINE_LABELS = {
    "pytorch_eager": "PyTorch eager",
    "onnxruntime": "ONNX Runtime (Python)",
    "openvino": "OpenVINO (Python)",
    "onnxruntime_cpp": "ONNX Runtime (C++)",
}


@dataclass(frozen=True, slots=True)
class LoadedRecord:
    """One schema-v1 record plus the file that supplied it."""

    data: Mapping[str, Any]
    source: Path

    @property
    def engine(self) -> str:
        return str(self.data["runner"]["engine"])

    @property
    def label(self) -> str:
        return ENGINE_LABELS.get(self.engine, self.engine)

    @property
    def comparison_key(self) -> tuple[object, ...]:
        model = self.data["model"]
        configuration = self.data["configuration"]
        runner = self.data["runner"]
        return (
            model["name"],
            runner["device"],
            tuple(model["input_shape"]),
            model.get("input_seed"),
            model.get("model_seed"),
            configuration["warmup_iterations"],
            configuration["timed_iterations"],
        )

    @property
    def recorded_at(self) -> datetime:
        value = self.data.get("created_at_utc")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.fromtimestamp(self.source.stat().st_mtime, tz=timezone.utc)

    def metric(self, name: str) -> float:
        return float(self.data["measurement"]["latency_ms"][name])

    @property
    def throughput(self) -> float:
        return float(self.data["measurement"]["throughput_samples_per_second"])


def _record_paths(inputs: Iterable[Path | str]) -> list[Path]:
    """Expand report inputs without accepting unrelated non-JSON files."""

    paths: list[Path] = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Benchmark record input does not exist: {path}")
    return paths


def load_records(inputs: Iterable[Path | str]) -> list[LoadedRecord]:
    """Load and minimally validate Python and native schema-v1 result records."""

    records: list[LoadedRecord] = []
    for path in _record_paths(inputs):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in benchmark record {path}: {error.msg}") from error
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path} is not a schema-v{SCHEMA_VERSION} benchmark record.")
        required_sections = ("runner", "model", "configuration", "measurement")
        if any(not isinstance(data.get(section), dict) for section in required_sections):
            raise ValueError(f"{path} is missing one or more required result sections.")
        latency = data["measurement"].get("latency_ms")
        if not isinstance(latency, dict) or "mean" not in latency:
            raise ValueError(f"{path} does not include measurement.latency_ms.mean.")
        records.append(LoadedRecord(data=data, source=path))
    if not records:
        raise ValueError("No schema-v1 benchmark JSON records were found.")
    return records


def latest_comparable_records(records: Iterable[LoadedRecord]) -> list[LoadedRecord]:
    """Choose the latest result per engine in the largest fair-comparison group."""

    groups: dict[tuple[object, ...], list[LoadedRecord]] = {}
    for record in records:
        groups.setdefault(record.comparison_key, []).append(record)
    _, candidates = max(
        groups.items(),
        key=lambda item: (len({record.engine for record in item[1]}), len(item[1])),
    )
    latest_by_engine: dict[str, LoadedRecord] = {}
    for record in candidates:
        current = latest_by_engine.get(record.engine)
        if current is None or record.recorded_at > current.recorded_at:
            latest_by_engine[record.engine] = record
    return sorted(latest_by_engine.values(), key=lambda record: (record.label, record.source.name))


def _format_number(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _rss_mib(record: LoadedRecord) -> float | None:
    rss = record.data["measurement"].get("process_rss", {})
    if rss.get("status") != "available":
        return None
    value = rss.get("value")
    return float(value) / (1024 * 1024) if isinstance(value, (int, float)) else None


def _parity(record: LoadedRecord) -> Mapping[str, Any] | None:
    correctness = record.data.get("correctness", {})
    return correctness.get("parity") if isinstance(correctness, dict) else None


def render_markdown(records: Iterable[LoadedRecord], all_records: Iterable[LoadedRecord]) -> str:
    """Render an explicit comparison table and data-coverage notes."""

    selected = list(records)
    all_loaded = list(all_records)
    exemplar = selected[0]
    model = exemplar.data["model"]
    configuration = exemplar.data["configuration"]
    rows = []
    for record in selected:
        parity = _parity(record)
        agreement = _format_number(float(parity["prediction_agreement"]), 4) if parity else "not checked"
        rows.append(
            "| {engine} | {device} | {mean} | {p50} | {p95} | {p99} | {throughput} | {rss} | {agreement} |".format(
                engine=record.label,
                device=record.data["runner"]["device"],
                mean=_format_number(record.metric("mean")),
                p50=_format_number(record.metric("p50")),
                p95=_format_number(record.metric("p95")),
                p99=_format_number(record.metric("p99")),
                throughput=_format_number(record.throughput),
                rss=_format_number(_rss_mib(record)),
                agreement=agreement,
            )
        )
    available = {record.engine for record in all_loaded}
    selected_engines = {record.engine for record in selected}
    coverage = "\n".join(
        f"- {ENGINE_LABELS[engine]}: {'included' if engine in selected_engines else ('record available but not protocol-compatible' if engine in available else 'no record found')}"
        for engine in EXPECTED_ENGINES
    )
    sources = "\n".join(f"- `{record.source.as_posix()}`" for record in selected)
    return f"""# Inference-engine comparison

## Fair-comparison scope

- Model: `{model['name']}`
- Device: `{exemplar.data['runner']['device']}`
- Input shape: `{model['input_shape']}`
- Input/model seeds: `{model.get('input_seed')}` / `{model.get('model_seed')}`
- Warmup/timed requests: `{configuration['warmup_iterations']}` / `{configuration['timed_iterations']}`

The table uses the latest record for each engine in the largest group with an identical model, device, input, seeds, and warm-run protocol. It does not compare incompatible runs.

| Engine | Device | Mean ms | p50 ms | p95 ms | p99 ms | Samples/s | RSS MiB | Parity agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Coverage

{coverage}

To include a native C++ run, redirect its schema-v1 JSON standard output into the input directory before generating this report.

## Included source records

{sources}
"""


def _write_bar_plot(
    records: Iterable[LoadedRecord],
    output_path: Path,
    *,
    metric: str,
    title: str,
    unit: str,
) -> None:
    """Render one labelled horizontal bar chart with Matplotlib's Agg backend."""

    selected = list(records)
    values = [record.metric(metric) if metric != "throughput" else record.throughput for record in selected]
    labels = [record.label for record in selected]
    figure, axis = pyplot.subplots(figsize=(9, max(3.5, len(selected) * 0.8 + 1.5)), layout="constrained")
    colors = ("#2864DC", "#0B8A6D", "#A45DE0", "#E07820")
    bars = axis.barh(
        labels, values, color=[colors[index % len(colors)] for index in range(len(selected))]
    )
    axis.invert_yaxis()
    axis.set_title(title, fontweight="bold")
    axis.set_xlabel(unit)
    axis.grid(axis="x", color="#D7DEE8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    maximum = max(values, default=1.0)
    axis.set_xlim(0, maximum * 1.18 if maximum > 0 else 1.0)
    axis.bar_label(bars, labels=[f"{value:.3f} {unit}" for value in values], padding=4)
    figure.savefig(output_path, dpi=200)
    pyplot.close(figure)


def write_report(inputs: Iterable[Path | str], output_directory: Path | str = "reports") -> dict[str, Path]:
    """Write the Markdown report and plots, returning their exact locations."""

    all_records = load_records(inputs)
    selected = latest_comparable_records(all_records)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "comparison.md"
    latency_path = directory / "mean_latency_ms.png"
    throughput_path = directory / "throughput_samples_per_second.png"
    report_path.write_text(render_markdown(selected, all_records), encoding="utf-8")
    _write_bar_plot(selected, latency_path, metric="mean", title="Mean warm-run latency", unit="ms")
    _write_bar_plot(selected, throughput_path, metric="throughput", title="Throughput", unit="samples/s")
    return {"report": report_path, "mean_latency_plot": latency_path, "throughput_plot": throughput_path}


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown comparison and Matplotlib plots from benchmark JSON records.")
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("results")], help="JSON record files or directories (default: results)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    """Run the report generator and print the resulting paths."""

    arguments = _parse_arguments()
    paths = write_report(arguments.inputs, arguments.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
