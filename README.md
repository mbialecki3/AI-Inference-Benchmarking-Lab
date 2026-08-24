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

## Classification benchmark slices

The benchmark matrix now supports ResNet-50, MobileNetV3-Large, and
EfficientNet-B0. They use the same static `float32 NCHW [1,3,224,224]` input
and produce 1,000-class logits, so they exercise the same PyTorch, ONNX
Runtime, OpenVINO, result, and native C++ measurement contracts. Model-specific
CLI defaults prevent one model command from accidentally consuming another
model's artifact.

Export the EfficientNet-B0 artifact and its paired native inputs:

```bash
PYTHONPATH=src python -m inference_bench.onnx_export --model efficientnet_b0
PYTHONPATH=src python -m inference_bench.input_artifact --model efficientnet_b0
```

Then run any Python engine with `--model efficientnet_b0`; its default ONNX
path is `artifacts/efficientnet_b0.onnx`. Native runners accept the same model
name and use a model-specific default ONNX and reference-output path.

## Native ONNX Runtime C++ CPU and CUDA

The native ONNX Runtime CPU and CUDA runners share one C++ implementation and
CMake pattern. They support the classification `images` → `logits` contract and
the raw YOLO11n `images` → `output0` contract. Each uses the existing warm-run
protocol (5 warmups and 20 timed synchronous requests by default), validates
the exact float32 static shapes, and emits schema-v1 measurement sections used
by the Python benchmark records. The artifact command also saves deterministic
PyTorch CPU outputs. Native runs compare their final raw output with that
reference and record maximum absolute/relative error plus prediction agreement.
For YOLO11n, agreement is the fraction of matching winning COCO classes at all
8,400 prediction locations; it is not mAP or post-processing accuracy.

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

## YOLO11n detection baseline

YOLO11n is a separate detection path: it uses a static `float32 NCHW
[1,3,640,640]` input and preserves the model's raw pre-NMS tensor. Its parity
check reports numerical error plus per-candidate winning-class agreement; it is
not COCO mAP and it does not benchmark post-processing. Each record stores its
raw detection layout (box-channel count, class/candidate axes, class range, and
class count), so a second detector can use a different output arrangement
without changing the parity contract. The detection slice
covers PyTorch eager and ONNX Runtime on CPU/CUDA plus OpenVINO `CPU` with an
explicit float32 inference hint, and native C++ ONNX Runtime on CPU/CUDA.

Detection entry points are model-generic: pass `--model` to
`inference_bench.detection_export` and `inference_bench.detection_benchmark`.
YOLO11n is the first registered detector; adding the next detector means adding
one explicit static contract to the registry rather than another runner stack.

Place the official `yolo11n.pt` checkpoint at `artifacts/yolo11n.pt`, then
export the static ONNX artifact and record a CUDA ONNX Runtime run:

```bash
PYTHONPATH=src python -m inference_bench.detection_export --model yolo11n
PYTHONPATH=src python -m inference_bench.detection_benchmark \
  --model yolo11n \
  --engine onnxruntime --device cuda:0 --verify-parity

# Compile the same static ONNX raw-output artifact for OpenVINO CPU.
PYTHONPATH=src python -m inference_bench.detection_benchmark \
  --model yolo11n \
  --engine openvino --device cpu --verify-parity

# Create byte-identical native input/reference artifacts, then run C++ CPU.
PYTHONPATH=src python -m inference_bench.input_artifact --model yolo11n
mkdir -p results/yolo11n/cpu
./build/cpp/cpp/onnxruntime_cpu_runner \
  --model yolo11n \
  --input-file artifacts/inputs/yolo11n_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/yolo11n_input69420_f32_raw.bin \
  > results/yolo11n/cpu/onnxruntime_cpp.json
```

Use the same `--model yolo11n`, input, and reference-output arguments with
`onnxruntime_cuda_runner` after the CUDA build described above. The native JSON
has `model_seed: null`, matching the Python YOLO records, so the report can
fairly compare all four engines when warmup and timed-request counts match.

The environment pins Ultralytics 8.4.127. Its YOLO11 model and ONNX export API
are documented by [Ultralytics](https://docs.ultralytics.com/models/yolo11) and
[its export guide](https://docs.ultralytics.com/modes/export).

## DeepLabV3-ResNet50 segmentation baseline

DeepLabV3-ResNet50 uses its own semantic-segmentation path, rather than the
classification or detection runners. It benchmarks a static float32
`[1,3,224,224]` input and preserves raw `[1,21,224,224]` VOC-class logits.
The optional parity check reports numerical error and per-pixel winning-class
agreement. It is an engine/export check, not dataset mIoU, image preprocessing,
or overlay rendering.

The seeded model is deliberately untrained and offline, matching the existing
classification baselines. Export once, then benchmark PyTorch or ONNX Runtime
on CPU/CUDA and OpenVINO on CPU:

```bash
PYTHONPATH=src python -m inference_bench.segmentation_export
PYTHONPATH=src python -m inference_bench.segmentation_benchmark \
  --engine onnxruntime --device cuda:0 --verify-parity
PYTHONPATH=src python -m inference_bench.segmentation_benchmark \
  --engine openvino --device cpu --verify-parity
```

OpenVINO remains CPU-only in this environment, so it is intentionally excluded
from CUDA comparisons. Native C++ ONNX Runtime currently supports classification
and YOLO11n only; rank-four DeepLab logits are a visible reporting coverage gap,
not a fabricated native result.

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

When `--output-dir` is omitted, each command writes one JSON record to
`results/<model>/<device>/`; for example,
`results/resnet50/cpu/` or `results/efficientnet_b0/cuda_0/`. The `cuda:0`
device label is normalized to `cuda_0` so the same layout works on Windows and
Linux. Supplying `--output-dir` overrides this convention. The first benchmark
scope is warm inference; cold process startup/model-load timing and device-only
CUDA-event timing remain later measurement additions. Engine-specific settings,
including OpenVINO's `inference_precision: f32`, are stored under
`runner.configuration` in every result record.

## Comparison reports and plots

Generate a reproducible Markdown comparison and two Matplotlib PNG bar plots from saved
schema-v1 records:

```bash
PYTHONPATH=src python -m inference_bench.reporting results/resnet50/cpu
```

For a single `results/<model>/<device>/` input, the report automatically writes
to the matching `reports/<model>/<device>/` directory. Aggregate input such as
`results/` retains the historical `reports/` default; `--output-dir` always
overrides either default. The report selects the latest run per engine only within the largest group with
the same model, device, input shape/seeds, and warm-run configuration. This
avoids silently comparing a CPU run to a CUDA run or two different benchmark
protocols. It produces `comparison.md`, `mean_latency_ms.png`, and
`throughput_samples_per_second.png`. The pinned Matplotlib dependency uses its
non-interactive `Agg` backend, so report generation works in WSL and CI without
a display server.

Native C++ runners write their schema-v1 JSON to standard output. Save that
output in the same automatically selected model/device directory before
generating the report:

```bash
mkdir -p results/resnet50/cpu
./build/cpp/cpp/onnxruntime_cpu_runner \
  --model resnet50 \
  --model-path artifacts/resnet50.onnx \
  --input-file artifacts/inputs/resnet50_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/resnet50_seed67_input69420_f32_logits.bin \
  > results/resnet50/cpu/onnxruntime_cpp_cpu.json
```

The coverage section makes missing or protocol-incompatible engines explicit;
it never fabricates a C++ result when one has not been recorded.

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
