# Architecture: Vision Inference Systems Lab

## Goal

Measure how deployment choices affect practical computer-vision models. Every engine receives the same model semantics, preprocessing, inputs, batch size, warm-up policy, and measurement protocol.

## Initial model suite

| Task | Model | Why it belongs in the suite |
| --- | --- | --- |
| Classification | ResNet-50 | Conventional high-compute convolutional baseline. |
| Classification | MobileNetV3-Large | Mobile-oriented, low-latency baseline. |
| Classification | EfficientNet-B0 | Compound-scaled classification design. |
| Object detection | YOLO11n | Practical low-overhead detector with post-processing costs. |
| Semantic segmentation | DeepLabV3-ResNet50 | Dense prediction with a different memory and output profile. |

The first implementation uses the smallest practical variants. Larger variants are configurations, not new framework code.

YOLO11 is deliberately pinned rather than silently tracking the newest YOLO family. Its current Ultralytics distribution is offered under AGPL-3.0 or an Enterprise license; any use beyond this learning benchmark must choose an appropriate license.

## Execution matrix

| Runner | Language | Devices in the initial matrix |
| --- | --- | --- |
| PyTorch eager | Python | CPU, CUDA |
| ONNX Runtime | Python | CPU, CUDA |
| OpenVINO | Python | CPU |
| ONNX Runtime native API | C++ | CPU, CUDA |
| OpenVINO native API | C++ | CPU |

OpenCV is shared preprocessing and image I/O. It will also provide visual overlays for detector and segmentation sanity checks. OpenCV DNN can become an additional runner later, but is not conflated with ONNX Runtime or OpenVINO.

## CUDA compatibility contract

CUDA is a required execution path for PyTorch and ONNX Runtime. A result is labelled `cuda` only after all of these checks pass:

1. Ubuntu WSL2 can run `nvidia-smi` and reports the selected NVIDIA GPU.
2. PyTorch reports `torch.cuda.is_available() == True`, records its CUDA and cuDNN versions, and runs a probe tensor on `cuda:0`.
3. ONNX Runtime lists `CUDAExecutionProvider`; the session is created with that provider first and its active providers are saved with the result.
4. A sampled GPU-memory or utilization change corroborates execution. If WSL telemetry does not expose a metric, the result says `unavailable` rather than claiming a value.

The driver is installed on the Windows host and mapped into WSL2. Never install an NVIDIA Linux display driver in WSL2. Python wheels supply the runtime dependencies for Python runners; the Linux CUDA toolkit is installed only for the C++ CUDA build path. The toolkit setup must use the WSL-Ubuntu package or a `cuda-toolkit-*` package, never `cuda`, `cuda-*`, or `cuda-drivers`, because those attempt to install a conflicting Linux driver.

The initial compatibility target is CUDA 13.x with cuDNN 9.x. PyTorch and ONNX Runtime GPU packages must share those major versions. Each result stores the exact framework, CUDA, cuDNN, driver, and execution-provider versions so incompatible comparisons are visible.

## Correctness contract

The framework records two different result families:

1. **Parity correctness** uses a fixed, deterministic input corpus. It compares every runner's raw output to PyTorch with maximum absolute error, relative error, and prediction agreement. This validates an export and runner setup.
2. **Task accuracy** uses a named validation dataset and records the task metric: classification top-1/top-5, detection mAP, or segmentation mIoU. It is not reported until a real dataset has been configured.

## Measurement contract

- A cold subprocess result reports process startup and model-load time.
- Warm runs report per-request latency percentiles (p50/p95/p99) after warm-up and synchronized CUDA timing.
- GPU timing uses CUDA events and synchronization. We will separately report end-to-end latency (including host-to-device transfer) and device-only latency, because they answer different deployment questions.
- The OpenVINO CPU parity runner explicitly selects float32 inference precision. Lower-precision OpenVINO optimizations are valuable later configurations, but they must not be mixed into a reference-equivalence result.
- Throughput is samples per second at an explicit batch size.
- Process RSS, GPU memory, CPU/GPU utilization, and NVIDIA power are sampled independently of the timed inference loop.
- Artifact size is measured for the model file(s) consumed by a runner. The
  initial OpenVINO path consumes ONNX; a later OpenVINO-IR path will measure
  the XML and BIN together.
- Every result records hardware, drivers, package versions, git revision, configuration, and unavailable metrics.

## Result-record contract

Every completed warm-run benchmark writes a schema-versioned JSON record. The
record has `runner`, `model`, `configuration`, `measurement`, `correctness`,
and `environment` sections. `measurement.latency_ms.samples` preserves every
warm-run observation, while its summary provides mean/min/max/p50/p95/p99;
throughput is derived from that mean and the explicit batch size.

The first record collector captures process peak RSS and before/after
`nvidia-smi` samples when available. It stores a structured `unavailable`
status otherwise. This prevents host-observability gaps from being confused
with a successful zero-valued metric. Engine-specific configuration belongs in
`runner.configuration`; for example, OpenVINO records its `f32` inference
precision there. ONNX Runtime and OpenVINO records can additionally include
parity against the seeded PyTorch reference; this is distinct from dataset-
level task accuracy.

### Native C++ boundary

The initial native ONNX Runtime CPU runner reads the same ONNX artifact and
validates the same `images`/`logits` names, float32 types, and static
`[1,3,224,224]`/`[1,1000]` shapes as the Python runner. To preserve exact
cross-language inputs, Python writes its seeded tensor as a little-endian
float32 binary artifact; C++ reads those bytes rather than attempting to
reproduce PyTorch's random-number generator. This input artifact is synthetic
parity data, never a validation dataset.

Native C++ CPU output is schema-versioned JSON with the same `runner`,
`model`, `configuration`, `measurement`, `correctness`, and `environment`
sections. It records raw latency samples and the same interpolated p50/p95/p99
calculation. Its initial correctness field is `null` until a portable saved
PyTorch-output artifact is introduced; this is intentionally not represented
as task accuracy or successful parity.

## Directory plan

```text
src/inference_bench/       Python orchestration, runners, measurement, reporting
cpp/                       CMake C++ runners
configs/                   Model and experiment configurations
tests/                     Correctness and regression tests
artifacts/                 Exported models (ignored)
results/                   Raw benchmark data (ignored)
plots/                     Generated plots (ignored)
```
