set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an extracted ONNX Runtime C/C++ release")
set(
    ONNXRUNTIME_LIBRARY_DIR
    ""
    CACHE PATH
    "Directory containing ONNX Runtime shared libraries; defaults to ONNXRUNTIME_ROOT/lib."
)
set(
    CUDA_RUNTIME_ROOT
    ""
    CACHE PATH
    "Root of a CUDA runtime distribution containing lib/libcudart.so."
)
set(
    CUDNN_ROOT
    ""
    CACHE PATH
    "Root of a cuDNN runtime distribution containing lib/libcudnn.so."
)

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
if(ONNXRUNTIME_LIBRARY_DIR)
    # The Python GPU wheel ships an exact-version filename rather than an
    # unversioned linker symlink. Prefer it when explicitly requested so the
    # CUDA provider plug-ins and core runtime come from the same distribution.
    find_file(
        ONNXRUNTIME_LIBRARY_FROM_DIR
        NAMES libonnxruntime.so libonnxruntime.so.1.29.0
        PATHS "${ONNXRUNTIME_LIBRARY_DIR}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIBRARY_FROM_DIR}" CACHE FILEPATH "ONNX Runtime library" FORCE)
else()
    find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime libonnxruntime.so.1.29.0
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
        REQUIRED
    )
endif()

find_package(Threads REQUIRED)

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

function(configure_native_runner target_name)
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    target_link_libraries(${target_name} PRIVATE onnxruntime::onnxruntime Threads::Threads)

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
    endif()
endfunction()

if(INFERENCE_BENCH_BUILD_ONNXRUNTIME_CPU)
    add_executable(onnxruntime_cpu_runner onnxruntime_cpu_runner.cpp)
    configure_native_runner(onnxruntime_cpu_runner)
endif()

if(INFERENCE_BENCH_BUILD_ONNXRUNTIME_CUDA)
    find_library(
        CUDA_RUNTIME_LIBRARY
        NAMES cudart libcudart.so.13
        HINTS "${CUDA_RUNTIME_ROOT}/lib" "${CUDA_RUNTIME_ROOT}/lib64"
        REQUIRED
    )
    find_library(
        CUDNN_LIBRARY
        NAMES cudnn libcudnn.so.9
        HINTS "${CUDNN_ROOT}/lib" "${CUDNN_ROOT}/lib64"
        REQUIRED
    )

    add_executable(onnxruntime_cuda_runner onnxruntime_cpu_runner.cpp)
    configure_native_runner(onnxruntime_cuda_runner)
    target_compile_definitions(onnxruntime_cuda_runner PRIVATE INFERENCE_BENCH_CUDA_RUNNER=1)
    target_link_libraries(onnxruntime_cuda_runner PRIVATE "${CUDA_RUNTIME_LIBRARY}")

    # GPU-enabled ONNX Runtime distributions load their execution-provider
    # plug-ins beside libonnxruntime. Keep that directory and libcudart on the
    # executable's runtime search path without copying vendor binaries.
    get_filename_component(ONNXRUNTIME_LIBRARY_PARENT "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
    get_filename_component(CUDA_RUNTIME_LIBRARY_PARENT "${CUDA_RUNTIME_LIBRARY}" DIRECTORY)
    get_filename_component(CUDNN_LIBRARY_PARENT "${CUDNN_LIBRARY}" DIRECTORY)
    find_file(
        ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY
        NAMES libonnxruntime_providers_shared.so
        PATHS "${ONNXRUNTIME_LIBRARY_PARENT}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    find_file(
        ONNXRUNTIME_CUDA_PROVIDER_LIBRARY
        NAMES libonnxruntime_providers_cuda.so
        PATHS "${ONNXRUNTIME_LIBRARY_PARENT}"
        NO_DEFAULT_PATH
        REQUIRED
    )
    set_target_properties(onnxruntime_cuda_runner PROPERTIES
        BUILD_RPATH "$ORIGIN;${ONNXRUNTIME_LIBRARY_PARENT};${CUDA_RUNTIME_LIBRARY_PARENT};${CUDNN_LIBRARY_PARENT}"
    )
    # The Python GPU wheel names the core library with its full patch version,
    # while its ELF SONAME is libonnxruntime.so.1. Put a build-local symlink
    # beside the executable so the loader can resolve that SONAME without
    # changing the managed Python environment.
    add_custom_command(
        TARGET onnxruntime_cuda_runner
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${ONNXRUNTIME_LIBRARY}"
            "$<TARGET_FILE_DIR:onnxruntime_cuda_runner>/libonnxruntime.so.1"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY}"
            "$<TARGET_FILE_DIR:onnxruntime_cuda_runner>/libonnxruntime_providers_shared.so"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${ONNXRUNTIME_CUDA_PROVIDER_LIBRARY}"
            "$<TARGET_FILE_DIR:onnxruntime_cuda_runner>/libonnxruntime_providers_cuda.so"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
            "${CUDNN_LIBRARY}"
            "$<TARGET_FILE_DIR:onnxruntime_cuda_runner>/libcudnn.so"
    )
endif()
