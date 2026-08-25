# AI Inference Benchmark Lab

A practical benchmark project for comparing computer-vision inference across
PyTorch, ONNX Runtime, and OpenVINO. It is designed for learning how deployment
choices affect latency, throughput, and output consistency—not for declaring a
universal winner.

The project runs primarily in Ubuntu on WSL2. Windows is the host; Python
benchmarking, CUDA userspace, C++ builds, and result collection run in Linux.
The layout should also work on a native Linux machine.

## What is included

- Classification: ResNet-50, MobileNetV3-Large, and EfficientNet-B0
- Detection: YOLO11n and YOLO11s, compared at their raw pre-NMS output
- Segmentation: DeepLabV3-ResNet50, compared at raw logits
- Runtimes: PyTorch eager, ONNX Runtime (Python and native C++), and OpenVINO
- Devices: CPU throughout; CUDA for PyTorch and ONNX Runtime where available
- Reports: versioned JSON records, Markdown summaries, and PNG charts

Every comparison uses a fixed model/input seed, five warm-up requests, and
twenty timed synchronous requests by default. The same input and output
contracts are used across engines, so the reports make it clear what was
actually compared.

## A recent result

This ResNet-50 study was run on the project's WSL2 host with an RTX 5080. It
is a useful starting point for that machine, not a performance guarantee for
other hardware or software versions.

| Configuration | Device | Mean / p95 | Throughput | Takeaway |
| --- | --- | ---: | ---: | --- |
| PyTorch eager, fp32 | CUDA | 3.137 / 3.975 ms | 318.8 samples/s | Reference baseline |
| PyTorch eager, fp16 | CUDA | 4.675 / 14.567 ms | 213.9 samples/s | Matched predicted classes, but was slower with tail spikes |
| ONNX Runtime, default | CUDA | 2.165 / 2.485 ms | 461.9 samples/s | Best result in this run |
| ONNX Runtime, tuned session | CUDA | 3.121 / 6.325 ms | 320.4 samples/s | Extra tuning made this workload slower |
| OpenVINO, f32 / latency | CPU | 7.826 / 9.119 ms | 127.8 samples/s | High-fidelity CPU baseline |
| OpenVINO, bf16 / throughput | CPU | 4.762 / 7.769 ms | 210.0 samples/s | Faster, with larger raw-logit differences |

Each saved result includes its model, inputs, runtime settings, package and
driver versions, latency samples, telemetry, and parity information. That
context matters as much as the headline number.

## Quick start

Run these commands from the repository root inside Ubuntu WSL2. CPU commands
work without an NVIDIA GPU. Use `cuda:0` only after `nvidia-smi` and the CUDA
execution provider are working in WSL2.

```bash
conda env create -f environment.yml
conda activate inference-bench

# Confirm the checkout is healthy.
python run_tests.py

# Export ResNet-50, benchmark ONNX Runtime on CPU, and generate a report.
PYTHONPATH=src python -m inference_bench.onnx_export --model resnet50
PYTHONPATH=src python -m inference_bench.benchmark \
  --engine onnxruntime --device cpu --verify-parity
PYTHONPATH=src python -m inference_bench.reporting results/resnet50/cpu
```

For CUDA, change `--device cpu` to `--device cuda:0`. Results are written to
`results/<model>/<device>/` (`cuda:0` becomes `cuda_0`), and the matching
Markdown report and PNG charts go under `reports/<model>/<device>/`.

## Running the models

### Classification

ResNet-50, MobileNetV3-Large, and EfficientNet-B0 all use a static
`float32 [1,3,224,224]` input and produce 1,000-class logits. Choose a model
with `--model` when exporting, benchmarking, or producing native input files.

```bash
PYTHONPATH=src python -m inference_bench.onnx_export --model efficientnet_b0
PYTHONPATH=src python -m inference_bench.input_artifact --model efficientnet_b0
PYTHONPATH=src python -m inference_bench.benchmark \
  --model efficientnet_b0 --engine onnxruntime --device cuda:0 --verify-parity
```

### Detection

YOLO11n and YOLO11s use a static `float32 [1,3,640,640]` input. Their output
is deliberately measured before NMS, so the parity result checks engine/export
behavior rather than detection mAP or post-processing choices.

Put an official checkpoint such as `yolo11n.pt` in `artifacts/`, then run:

```bash
PYTHONPATH=src python -m inference_bench.detection_export --model yolo11n
PYTHONPATH=src python -m inference_bench.detection_benchmark \
  --model yolo11n --engine onnxruntime --device cuda:0 --verify-parity
PYTHONPATH=src python -m inference_bench.detection_benchmark \
  --model yolo11n --engine openvino --device cpu --verify-parity
```

The detection record saves its raw layout as well, so adding a detector with a
different output shape does not require duplicating the runner stack. The
environment pins Ultralytics 8.4.127; see the
[YOLO11 documentation](https://docs.ultralytics.com/models/yolo11) and
[export guide](https://docs.ultralytics.com/modes/export) for checkpoint and
export details.

### Segmentation

DeepLabV3-ResNet50 uses a `float32 [1,3,224,224]` input and retains raw
`[1,21,224,224]` logits. Per-pixel winning-class agreement is an export/engine
sanity check, not segmentation mIoU.

```bash
PYTHONPATH=src python -m inference_bench.segmentation_export
PYTHONPATH=src python -m inference_bench.segmentation_benchmark \
  --engine onnxruntime --device cuda:0 --verify-parity
PYTHONPATH=src python -m inference_bench.segmentation_benchmark \
  --engine openvino --device cpu --verify-parity
```

OpenVINO is intentionally CPU-only in the current environment.

## Native C++ ONNX Runtime

The native CPU and CUDA runners share a C++ implementation. They use the same
ONNX files and seeded input/reference-output artifacts as the Python runners,
then write the same schema-v1 JSON shape to standard output.

First create the input artifacts and build against a matching ONNX Runtime C++
release (currently 1.29.x):

```bash
PYTHONPATH=src python -m inference_bench.input_artifact --model resnet50

cmake -S . -B build/cpp -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.29.0
cmake --build build/cpp --parallel

./build/cpp/cpp/onnxruntime_cpu_runner \
  --model resnet50 \
  --model-path artifacts/resnet50.onnx \
  --input-file artifacts/inputs/resnet50_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/resnet50_seed67_input69420_f32_logits.bin
```

`ONNXRUNTIME_ROOT` needs `include/onnxruntime_cxx_api.h` and the runtime
library in `lib/` or `lib64/`. Set `LD_LIBRARY_PATH` to that library directory
if the Linux loader cannot find `libonnxruntime.so`.

For the CUDA runner, configure a separate build with a GPU-enabled ONNX Runtime
library directory plus CUDA and cuDNN roots:

```bash
ORT_SDK=/path/to/onnxruntime-linux-x64-1.29.0
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
  --model resnet50 \
  --input-file artifacts/inputs/resnet50_seed69420_f32_nchw.bin \
  --reference-output artifacts/reference_outputs/resnet50_seed67_input69420_f32_logits.bin
```

The build creates local library links as needed; it does not copy vendor
binaries into the repository or modify the Conda environment.

## Experiments and reports

The default benchmark is fp32 with a latency-oriented setup. You can run
explicit variants under the same input and timing protocol, including ONNX
Runtime session options, OpenVINO precision/performance hints, and PyTorch
fp16 on CUDA.

```bash
# ONNX Runtime session experiment
PYTHONPATH=src python -m inference_bench.benchmark --engine onnxruntime \
  --device cuda:0 --ort-graph-optimization extended \
  --ort-execution-mode parallel --ort-intra-op-threads 4 \
  --ort-cuda-conv-algorithm heuristic --verify-parity

# OpenVINO throughput experiment
PYTHONPATH=src python -m inference_bench.benchmark --engine openvino \
  --openvino-performance-hint throughput \
  --openvino-inference-precision bf16 --verify-parity
```

Generate a comparison from one result directory with:

```bash
PYTHONPATH=src python -m inference_bench.reporting results/resnet50/cpu
```

The report only groups compatible runs: same model, device, inputs, warm-up
count, and timed-request count. Missing or incompatible engines are called out
instead of being quietly compared.

## Tests

```bash
python run_tests.py
```

For standard unittest output:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Reading the numbers

- Parity is not task accuracy. Synthetic seeded tensors validate export and
  runner equivalence; they do not measure ImageNet top-1/top-5, COCO mAP, or
  segmentation mIoU.
- Some baselines are intentionally offline and untrained. They exercise a
  reproducible inference path, not an application-quality model.
- Results are hardware- and version-specific. Re-run the saved protocol on the
  target deployment machine before choosing a production configuration.
- The current scope is synchronous inference. It does not yet cover full
  interpreter startup, preprocessing, NMS, visual overlays, or async serving
  throughput.
