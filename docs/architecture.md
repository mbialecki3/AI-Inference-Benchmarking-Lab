# Architecture: Vision Inference Systems Lab

## Goal

Measure how deployment choices affect practical computer-vision models. Every engine receives the same model semantics, preprocessing, inputs, batch size, warm-up policy, and measurement protocol.

## Initial model suite

| Task | Model | Why it belongs in the suite |
| --- | --- | --- |
| Classification | ResNet-50 | Conventional high-compute convolutional baseline. |
| Classification | MobileNetV3-Large | Mobile-oriented, low-latency baseline. |
| Classification | EfficientNet-B0 | Compound-scaled classification design. |
| Object detection | YOLO11n, YOLO11s | Small and compact detector baselines sharing one raw-output contract. |
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

YOLO11n and YOLO11s use a separate detection pipeline. They use a static `float32`
`[1,3,640,640]` input and compares raw pre-NMS output tensors, where the first
four channels encode boxes and later channels encode COCO class scores. This
keeps engine parity distinct from NMS policy, image preprocessing, and COCO mAP.
The Python slice provides PyTorch and ONNX Runtime runners plus an OpenVINO
CPU runner. All preserve the raw output boundary and explicitly request
float32 OpenVINO inference for parity. The native ONNX Runtime CPU/CUDA runner
uses this same raw-output contract and does not add post-processing.
Each detection record serializes the raw layout: box-coordinate channel count,
class and candidate axes, class-channel start, and class count. Parity consumes
that layout instead of relying on a detector-family-specific channel ordering.

DeepLabV3-ResNet50 is a dedicated segmentation path with static float32
`images [1,3,224,224]` and `logits [1,21,224,224]` contracts. Its parity metric
uses the argmax across the class channel at every pixel; it validates engine
equivalence but is not semantic-segmentation mIoU. The Python matrix covers
PyTorch eager and ONNX Runtime on CPU/CUDA plus OpenVINO CPU with an explicit
float32 hint. The native C++ ONNX Runtime runner also validates the rank-four
raw-logit contract and measures its CPU/CUDA path.

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

- A fresh-run `cold_start_model_load_ms` sample reports model construction or
  artifact loading plus runtime session/compile setup. It intentionally excludes
  Python interpreter process startup; a future subprocess harness can add that
  broader deployment measurement without contaminating warm-run records.
- Warm runs report per-request latency percentiles (p50/p95/p99) after warm-up and synchronized CUDA timing.
- CUDA PyTorch runs additionally record device-only CUDA-event samples. ONNX
  Runtime host timing remains end-to-end because its opaque execution stream
  cannot be truthfully event-timed without an I/O-binding/user-stream path.
- ONNX Runtime session/provider settings and OpenVINO performance hints are
  serializable experiments. OpenVINO defaults to float32/latency; `f16` and
  `bf16` variants must retain a parity result before interpretation.
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
throughput is derived from that mean and the explicit batch size. The separate
`measurement.cold_start_model_load_ms` and optional
`measurement.device_latency_ms` sections never replace warm latency.

The first record collector captures process peak RSS and before/after
`nvidia-smi` samples when available. It stores a structured `unavailable`
status otherwise. This prevents host-observability gaps from being confused
with a successful zero-valued metric. Engine-specific configuration belongs in
`runner.configuration`; for example, OpenVINO records its `f32` inference
precision and performance hints there. ONNX Runtime and OpenVINO records can additionally include
parity against the seeded PyTorch reference; this is distinct from dataset-
level task accuracy.

### Native C++ boundary

The native ONNX Runtime CPU and CUDA runners read the same ONNX artifacts and
support the ResNet-50, MobileNetV3-Large, and EfficientNet-B0 classification
contracts, YOLO11n and YOLO11s raw detection, and DeepLabV3-ResNet50 semantic
segmentation. Classification validates `images`/`logits`, float32, and static
`[1,3,224,224]`/`[1,1000]`; the YOLO11 models validate `images`/`output0`,
float32, and static `[1,3,640,640]`/`[1,84,8400]`; DeepLab validates
`images`/`logits`, float32, and static `[1,3,224,224]`/`[1,21,224,224]`. To preserve exact
cross-language inputs, Python writes its seeded tensor as a little-endian
float32 binary artifact; C++ reads those bytes rather than attempting to
reproduce PyTorch's random-number generator. This input artifact is synthetic
parity data, never a validation dataset.

Native C++ CPU and CUDA output are schema-versioned JSON with the same `runner`,
`model`, `configuration`, `measurement`, `correctness`, and `environment`
sections. It records raw latency samples and the same interpolated p50/p95/p99
calculation. Python also saves the matching seeded PyTorch CPU output as a
little-endian float32 artifact. Native runners load it and record maximum
absolute error, maximum relative error, and prediction agreement. For the YOLO11 models,
the C++ runner calculates agreement over the highest-scoring class among
channels 4--83 at each of the 8,400 locations, and emits `model_seed: null` to
match the pretrained Python records. This is intentionally not represented as
task accuracy.

The CUDA variant appends ONNX Runtime's CUDA execution provider for device 0
before creating its session, synchronizes after every warm and timed request,
and samples `nvidia-smi` before and after the timed loop. The C++ SDK and a
matching GPU-enabled ONNX Runtime distribution are separate inputs because the
provider is a dynamically loaded plug-in. CMake keeps the required loader
symlinks in the ignored build tree rather than copying or modifying vendor
libraries. CMake embeds the resolved source Git revision in each native result;
when Git is unavailable at configure time the field is explicitly `null`.

### Comparison reporting

The reporting command consumes schema-v1 JSON from both Python persistence and
native C++ standard output. It selects the latest record per engine from one
comparison group whose model, device, input shape/seeds, warmup count, and timed
request count are identical. A Markdown table reports mean and tail latency,
throughput, RSS, and parity agreement; two Matplotlib PNG plots present mean
latency and throughput. The renderer uses Matplotlib's non-interactive `Agg`
backend for WSL and CI. Missing or protocol-incompatible runners are
reported as coverage gaps instead of being included in a misleading chart.

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
