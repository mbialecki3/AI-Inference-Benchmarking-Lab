set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an extracted ONNX Runtime C/C++ release")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is required. Set it to the ONNX Runtime C/C++ release "
        "directory containing include/ and lib/."
    )
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED
)
find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
    REQUIRED
)

find_package(Threads REQUIRED)

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

add_executable(onnxruntime_cpu_runner onnxruntime_cpu_runner.cpp)
target_compile_features(onnxruntime_cpu_runner PRIVATE cxx_std_20)
target_link_libraries(onnxruntime_cpu_runner PRIVATE onnxruntime::onnxruntime Threads::Threads)

if(MSVC)
    target_compile_options(onnxruntime_cpu_runner PRIVATE /W4 /permissive-)
else()
    target_compile_options(onnxruntime_cpu_runner PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
