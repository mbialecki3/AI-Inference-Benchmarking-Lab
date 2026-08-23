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

## Environment

Create the environment from `environment.yml` **inside Ubuntu WSL2**:

```powershell
conda env create -f environment.yml
conda activate inference-bench
```

Python packages for an engine are added only in the stage that uses it. This keeps compatibility failures easy to identify and makes results reproducible. The earlier Windows environment remains isolated but is no longer the benchmark target.

## TensorFlow scope

TensorFlow is no longer part of the first engine matrix. The initial framework uses PyTorch, ONNX Runtime, and OpenVINO. This keeps the comparison focused on one PyTorch-to-ONNX path and avoids mixing framework-conversion effects with inference-engine effects.
