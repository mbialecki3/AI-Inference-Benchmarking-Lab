# AI Inference Benchmark Lab

This project is a learning-oriented systems benchmark for comparing practical computer-vision models across inference engines and available devices.

The primary runtime target is **Ubuntu on WSL2**. Windows is the host only; the benchmark, CUDA userspace, C++ builds, and result collection run in Linux. The layout is intentionally portable to a native Linux machine.

## Stages

1. Design a fair benchmark and inspect hardware.
2. Create a reproducible Linux + Conda environment and C++ toolchain.
3. Establish deterministic PyTorch reference models and a correctness contract.
4. Export and validate ONNX.
5. Add Python and C++ engine runners plus metric collection.
6. Produce reports and plots; then extend to optimizations.

See [the architecture guide](docs/architecture.md) for the model suite, execution matrix, and measurement rules.

## Current milestone: OpenVINO CPU

The OpenVINO runner loads the validated ResNet-50 ONNX artifact and compiles it
explicitly for `CPU`. It uses the same seeded float32 NCHW input as the
PyTorch and ONNX Runtime runners, validates the `images`/`logits` interface,
and records the execution device reported by OpenVINO. The runner explicitly
requests float32 inference precision: this keeps parity checks about engine
semantics rather than lower-precision CPU optimizations.

From the Ubuntu WSL2 environment, run:

```bash
PYTHONPATH=src python -m inference_bench.openvino_runner --device cpu
```

## Benchmark records

`inference_bench.benchmark` wraps an existing runner in a shared, versioned
record. It stores raw warm-run samples; mean, p50, p95, and p99 latency;
throughput; process RSS; artifact size; package and driver metadata; and
best-effort GPU telemetry. Missing host telemetry is recorded as
`"unavailable"`; it never invalidates an otherwise valid CPU run.

Run and save a PyTorch CPU record:

```bash
PYTHONPATH=src python -m inference_bench.benchmark --engine pytorch --device cpu
```

Run and save an ONNX Runtime CUDA record with an output-parity check against
the corresponding PyTorch reference:

```bash
PYTHONPATH=src python -m inference_bench.benchmark --engine onnxruntime --device cuda:0 --verify-parity
```

Run and save an OpenVINO CPU record with the same parity check:

```bash
PYTHONPATH=src python -m inference_bench.benchmark --engine openvino --device cpu --verify-parity
```

Each command writes one JSON file under `results/` by default. The first
benchmark scope is warm inference; cold process startup/model-load timing and
device-only CUDA-event timing remain later measurement additions. Engine-
specific settings, including OpenVINO's `inference_precision: f32`, are stored
under `runner.configuration` in every result record.

## Tests

For concise, readable test output with timings, run:

```bash
python run_tests.py
```

The standard unittest command remains available when needed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Environment

Create the environment from `environment.yml` **inside Ubuntu WSL2**:

```powershell
conda env create -f environment.yml
conda activate inference-bench
```

The file pins the tested direct Python dependencies, including the official
CUDA 13.2 PyTorch wheel index and OpenVINO for the CPU runner. The earlier
Windows environment remains isolated; a fresh environment is intended for the
WSL2/Linux benchmark target.

## TensorFlow scope

TensorFlow is no longer part of the first engine matrix. The initial framework uses PyTorch, ONNX Runtime, and OpenVINO. This keeps the comparison focused on one PyTorch-to-ONNX path and avoids mixing framework-conversion effects with inference-engine effects.
