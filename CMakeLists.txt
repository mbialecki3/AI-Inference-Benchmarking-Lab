cmake_minimum_required(VERSION 3.25)

project(ai_inference_bench LANGUAGES CXX)

option(
    INFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU
    "Build the native ONNX Runtime CPU benchmark runner."
    ON
)

option(
    INFERENCE_BENCH_BUILD_ONNXRUNTIME_CUDA
    "Build the native ONNX Runtime CUDA benchmark runner."
    OFF
)

if(INFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU OR INFERENCE_BENCH_BUILD_ONNXRUNTIME_CUDA)
    add_subdirectory(cpp)
endif()
