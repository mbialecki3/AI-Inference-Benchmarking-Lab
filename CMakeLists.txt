cmake_minimum_required(VERSION 3.25)

project(ai_inference_bench LANGUAGES CXX)

option(
    INFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU
    "Build the native ONNX Runtime CPU benchmark runner."
    ON
)

if(INFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU)
    add_subdirectory(cpp)
endif()
