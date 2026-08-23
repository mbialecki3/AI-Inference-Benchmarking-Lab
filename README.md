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

## Current milestone: native ONNX Runtime C++ CPU and CUDA

The C++ track now begins with a native ONNX Runtime CPU runner and CMake
scaffolding. It consumes the same validated `images` → `logits` ONNX artifact,
uses the existing warm-run protocol (5 warmups and 20 timed synchronous
requests by default), verifies float32 `NCHW [1,3,224,224]` input and float32
`[1,1000]` logits, and emits the schema-v1 measurement sections used by the
Python benchmark records. The artifact command also saves deterministic
PyTorch CPU logits. Both native runners compare their final logits with that
reference and record maximum absolute/relative error plus predicted-class
agreement. This is a runner correctness check, not a dataset-accuracy metric.

Generate the exact seeded input and reference-output bytes used by the Python
runners. This avoids trying to duplicate PyTorch's RNG behavior or model
initialization in C++:

```bash
PYTHONPATH=src python -m inference_bench.input_artifact
```

Download/extract the ONNX Runtime C/C++ release matching the validated
runtime (currently 1.29.x), then configure and build from Ubuntu WSL2:

```bash
cmake -S . -B build/cpp -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.29.0
cmake --build build/cpp --parallel
./build/cpp/cpp/onnxruntime_cpu_runner \
  --model-path artifacts/resnet50.onnx \
  --input-file artifacts/inputs/resnet50_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/resnet50_seed67_input69420_f32_logits.bin
```

`ONNXRUNTIME_ROOT` must contain `include/onnxruntime_cxx_api.h` and `lib/` (or
`lib64/`) with the ONNX Runtime shared library. Keep that release alongside
the Python `onnxruntime-gpu==1.29.0` pin so CPU results remain attributable to
the same runtime generation. CMake intentionally does not download a runtime:
the release path is explicit and reviewable. On Linux, set
`LD_LIBRARY_PATH` to the release's `lib/` directory if the dynamic loader
cannot locate `libonnxruntime.so`.

### Native ONNX Runtime CUDA

The CUDA executable uses ONNX Runtime's CUDA execution provider on device 0.
It synchronizes CUDA after every warm and timed request, so each latency sample
includes completed GPU work rather than only asynchronous kernel submission.
It also samples `nvidia-smi` immediately before and after the timed loop. The
provider uses the same reference-logit artifact as the CPU runner; GPU and CPU
floating-point reductions may differ numerically, so inspect the error fields
alongside prediction agreement.

The C++ SDK supplies headers, while a matching GPU-enabled ONNX Runtime
distribution supplies the CUDA provider plug-ins. Point CMake at both, plus
the CUDA runtime and cuDNN roots. In the configured WSL2 environment these
are provided by the installed packages:

```bash
ORT_SDK=/home/mitch/opt/onnxruntime-linux-x64-1.29.0
ORT_GPU_LIB=$CONDA_PREFIX/lib/python3.12/site-packages/onnxruntime/capi
CUDA_RUNTIME_ROOT=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13
CUDNN_ROOT=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn

cmake -S . -B build/cpp-cuda \
  -DINFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU=OFF \
  -DINFERENCE_BENCH_BUILD_ONNXRUNTIME_CUDA=ON \
  -DONNXRUNTIME_ROOT="$ORT_SDK" \
  -DONNXRUNTIME_LIBRARY_DIR="$ORT_GPU_LIB" \
  -DCUDA_RUNTIME_ROOT="$CUDA_RUNTIME_ROOT" \
  -DCUDNN_ROOT="$CUDNN_ROOT"
cmake --build build/cpp-cuda --parallel

export LD_LIBRARY_PATH="build/cpp-cuda/cpp:$ORT_GPU_LIB:$CUDA_RUNTIME_ROOT/lib:$CUDNN_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./build/cpp-cuda/cpp/onnxruntime_cuda_runner \
  --model-path artifacts/resnet50.onnx \
  --input-file artifacts/inputs/resnet50_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/resnet50_seed67_input69420_f32_logits.bin
```

The CUDA target creates build-local symlinks to the requested runtime and
provider libraries; it does not copy vendor binaries into the repository or
modify the Conda environment. The CPU target remains separately buildable on
hosts without CUDA.

## Python OpenVINO CPU milestone

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
