// Native ONNX Runtime CPU runner for the benchmark's validated ONNX artifact.
//
// The runner deliberately consumes a float32 binary input exported by
// inference_bench.input_artifact. That keeps cross-language parity about one
// byte-identical tensor, rather than two unrelated pseudorandom generators.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(INFERENCE_BENCH_CUDA_RUNNER)
// The CUDA runtime ABI is stable for these calls. Declaring this tiny surface
// locally lets the runner use pip-distributed CUDA runtime libraries, whose
// headers omit NVCC-only internal headers needed by cuda_runtime_api.h.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
extern "C" cudaError_t cudaDeviceSynchronize();
extern "C" cudaError_t cudaGetDeviceCount(int* count);
extern "C" const char* cudaGetErrorString(cudaError_t error);
extern "C" cudaError_t cudaRuntimeGetVersion(int* runtime_version);
#endif

namespace {

constexpr std::string_view kDefaultModelName = "resnet50";
constexpr std::int64_t kDefaultInputSeed = 69420;
constexpr std::int64_t kDefaultModelSeed = 67;
constexpr int kDefaultWarmupIterations = 5;
constexpr int kDefaultTimedIterations = 20;

[[noreturn]] void fail(const std::string& message);

enum class Task { kClassification, kDetection };

struct DetectionLayout {
    std::string_view output;
    std::int64_t box_coordinate_channels;
    std::int64_t class_channel_axis;
    std::int64_t candidate_axis;
    std::int64_t class_channel_start;
    std::int64_t class_count;
};

struct ModelContract {
    std::string_view name;
    std::string_view input_name;
    std::string_view output_name;
    std::span<const std::int64_t> input_shape;
    std::span<const std::int64_t> output_shape;
    Task task;
    std::optional<DetectionLayout> detection_layout;
};

constexpr std::array<std::int64_t, 4> kClassificationInputShape{1, 3, 224, 224};
constexpr std::array<std::int64_t, 2> kClassificationOutputShape{1, 1000};
constexpr std::array<std::int64_t, 4> kYolo11nInputShape{1, 3, 640, 640};
constexpr std::array<std::int64_t, 3> kYolo11nOutputShape{1, 84, 8400};

const ModelContract& model_contract(std::string_view name) {
    static constexpr ModelContract kResnet50{
        "resnet50", "images", "logits", kClassificationInputShape, kClassificationOutputShape,
        Task::kClassification, std::nullopt};
    static constexpr ModelContract kMobilenetV3Large{
        "mobilenet_v3_large", "images", "logits", kClassificationInputShape, kClassificationOutputShape,
        Task::kClassification, std::nullopt};
    static constexpr ModelContract kEfficientnetB0{
        "efficientnet_b0", "images", "logits", kClassificationInputShape, kClassificationOutputShape,
        Task::kClassification, std::nullopt};
    static constexpr ModelContract kYolo11n{
        "yolo11n", "images", "output0", kYolo11nInputShape, kYolo11nOutputShape, Task::kDetection,
        DetectionLayout{"raw_pre_nms", 4, 1, 2, 4, 80}};
    if (name == kResnet50.name) return kResnet50;
    if (name == kMobilenetV3Large.name) return kMobilenetV3Large;
    if (name == kEfficientnetB0.name) return kEfficientnetB0;
    if (name == kYolo11n.name) return kYolo11n;
    fail("Supported native models: resnet50, mobilenet_v3_large, efficientnet_b0, yolo11n.");
}

struct Options {
    std::filesystem::path model_path{"artifacts/resnet50.onnx"};
    std::filesystem::path input_file;
    std::filesystem::path reference_output_file;
    std::string model_name{kDefaultModelName};
    std::int64_t input_seed{kDefaultInputSeed};
    std::optional<std::int64_t> model_seed{kDefaultModelSeed};
    int warmup_iterations{kDefaultWarmupIterations};
    int timed_iterations{kDefaultTimedIterations};
};

struct LatencySummary {
    std::vector<double> samples_ms;
    double mean_ms{};
    double min_ms{};
    double max_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
};

struct OutputParity {
    double max_absolute_error{};
    double max_relative_error{};
    double prediction_agreement{};
};

#if defined(INFERENCE_BENCH_CUDA_RUNNER)
constexpr std::string_view kRunnerExecutable = "onnxruntime_cuda_runner";
constexpr std::string_view kRunnerDevice = "cuda:0";
constexpr std::string_view kPrimaryProvider = "CUDAExecutionProvider";
#else
constexpr std::string_view kRunnerExecutable = "onnxruntime_cpu_runner";
constexpr std::string_view kRunnerDevice = "cpu";
constexpr std::string_view kPrimaryProvider = "CPUExecutionProvider";
#endif

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string_view require_value(int& index, int argc, char* argv[], std::string_view option) {
    if (++index >= argc) {
        fail("Missing value for " + std::string(option) + ".");
    }
    return argv[index];
}

int parse_positive_int(std::string_view value, std::string_view option, bool allow_zero) {
    std::size_t consumed = 0;
    int parsed{};
    try {
        parsed = std::stoi(std::string(value), &consumed);
    } catch (const std::exception&) {
        fail(std::string(option) + " must be an integer.");
    }
    if (consumed != value.size() || parsed < 0 || (!allow_zero && parsed == 0)) {
        fail(std::string(option) + (allow_zero ? " must be a non-negative integer." : " must be a positive integer."));
    }
    return parsed;
}

std::int64_t parse_int64(std::string_view value, std::string_view option) {
    std::size_t consumed = 0;
    std::int64_t parsed{};
    try {
        parsed = std::stoll(std::string(value), &consumed);
    } catch (const std::exception&) {
        fail(std::string(option) + " must be an integer.");
    }
    if (consumed != value.size()) {
        fail(std::string(option) + " must be an integer.");
    }
    return parsed;
}

void print_usage(std::ostream& stream) {
    stream << "Usage: " << kRunnerExecutable << " --input-file PATH [options]\n"
           << "\n"
           << "Runs a validated classification or raw YOLO11n ONNX artifact with ONNX Runtime's "
           << kPrimaryProvider << ".\n"
           << "The input file must be the float32 NCHW binary emitted by\n"
           << "python -m inference_bench.input_artifact.\n"
           << "\n"
           << "Options:\n"
           << "  --model NAME         resnet50, mobilenet_v3_large, efficientnet_b0, or yolo11n "
              "(default: resnet50)\n"
           << "  --model-path PATH    ONNX artifact (default: artifacts/<model>.onnx)\n"
           << "  --input-file PATH    Required deterministic float32 input binary\n"
           << "  --reference-output PATH  PyTorch float32 raw output for numerical parity\n"
           << "  --input-seed N       Metadata only; must match input artifact (default: 69420)\n"
           << "  --model-seed N       PyTorch seed used for classification reference logits (default: 67)\n"
           << "  --warmup N           Warm synchronous inferences (default: 5)\n"
           << "  --iterations N       Timed synchronous inferences (default: 20)\n"
           << "  --help               Show this help text\n";
}

Options parse_arguments(int argc, char* argv[]) {
    Options options;
    bool has_model_path = false;
    bool has_reference_output = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help") {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--model") {
            const auto model = require_value(index, argc, argv, argument);
            (void)model_contract(model);
            options.model_name = model;
        } else if (argument == "--model-path") {
            options.model_path = require_value(index, argc, argv, argument);
            has_model_path = true;
        } else if (argument == "--input-file") {
            options.input_file = require_value(index, argc, argv, argument);
        } else if (argument == "--reference-output") {
            options.reference_output_file = require_value(index, argc, argv, argument);
            has_reference_output = true;
        } else if (argument == "--input-seed") {
            options.input_seed = parse_int64(require_value(index, argc, argv, argument), argument);
        } else if (argument == "--model-seed") {
            options.model_seed = parse_int64(require_value(index, argc, argv, argument), argument);
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_positive_int(require_value(index, argc, argv, argument), argument, true);
        } else if (argument == "--iterations") {
            options.timed_iterations = parse_positive_int(require_value(index, argc, argv, argument), argument, false);
        } else {
            fail("Unknown option: " + std::string(argument));
        }
    }
    if (options.input_file.empty()) {
        fail("--input-file is required to preserve byte-identical cross-language inputs.");
    }
    const auto& contract = model_contract(options.model_name);
    if (!has_model_path) {
        options.model_path = std::filesystem::path{"artifacts"} / (options.model_name + ".onnx");
    }
    if (!has_reference_output) {
        if (contract.task == Task::kDetection) {
            options.reference_output_file = std::filesystem::path{"artifacts/reference_outputs"}
                / (options.model_name + "_input" + std::to_string(options.input_seed) + "_f32_raw.bin");
        } else {
            options.reference_output_file = std::filesystem::path{"artifacts/reference_outputs"}
                / (options.model_name + "_seed" + std::to_string(*options.model_seed)
                   + "_input" + std::to_string(options.input_seed) + "_f32_logits.bin");
        }
    }
    if (contract.task == Task::kDetection) options.model_seed = std::nullopt;
    return options;
}

std::string shape_description(std::span<const std::int64_t> shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) stream << ',';
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

std::size_t expected_elements(std::span<const std::int64_t> shape) {
    return static_cast<std::size_t>(std::accumulate(
        shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>{}));
}

std::vector<float> read_input_file(const std::filesystem::path& path, const ModelContract& contract) {
    if constexpr (std::endian::native != std::endian::little) {
        fail("The native CPU runner currently requires a little-endian host.");
    }
    if (!std::filesystem::is_regular_file(path)) {
        fail("Input file does not exist: " + path.string());
    }
    const auto expected_bytes = expected_elements(contract.input_shape) * sizeof(float);
    const auto actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        fail("Input file has " + std::to_string(actual_bytes) + " bytes; expected "
             + std::to_string(expected_bytes) + " for float32 NCHW " + shape_description(contract.input_shape) + ".");
    }

    std::vector<float> values(expected_elements(contract.input_shape));
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(actual_bytes));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(actual_bytes)) {
        fail("Could not read the complete input file: " + path.string());
    }
    return values;
}

std::vector<float> read_reference_output_file(
    const std::filesystem::path& path, const ModelContract& contract) {
    if constexpr (std::endian::native != std::endian::little) {
        fail("The native runner currently requires a little-endian host.");
    }
    if (!std::filesystem::is_regular_file(path)) {
        fail("Reference-output file does not exist: " + path.string());
    }
    const auto output_elements = expected_elements(contract.output_shape);
    const auto expected_bytes = output_elements * sizeof(float);
    const auto actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        fail("Reference-output file has " + std::to_string(actual_bytes) + " bytes; expected "
             + std::to_string(expected_bytes) + " for float32 raw output " + shape_description(contract.output_shape) + ".");
    }

    std::vector<float> values(output_elements);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(actual_bytes));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(actual_bytes)) {
        fail("Could not read the complete reference-output file: " + path.string());
    }
    return values;
}

template <typename TensorTypeAndShapeInfo>
std::vector<std::int64_t> tensor_shape(const TensorTypeAndShapeInfo& info) {
    return info.GetShape();
}

void validate_interface(
    Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, const ModelContract& contract) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        fail("Expected exactly one ONNX input and output.");
    }
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    if (std::string_view{input_name.get()} != contract.input_name
        || std::string_view{output_name.get()} != contract.output_name) {
        fail("ONNX interface does not match " + std::string(contract.input_name) + " -> "
             + std::string(contract.output_name) + ".");
    }

    // GetTensorTypeAndShapeInfo returns an unowned view. Keep TypeInfo alive
    // while validating that view, especially with ONNX Runtime 1.29+.
    const auto input_type_info = session.GetInputTypeInfo(0);
    const auto output_type_info = session.GetOutputTypeInfo(0);
    const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        fail("Expected float32 ONNX input and output tensors.");
    }
    if (tensor_shape(input_info) != std::vector<std::int64_t>(contract.input_shape.begin(), contract.input_shape.end())
        || tensor_shape(output_info) != std::vector<std::int64_t>(contract.output_shape.begin(), contract.output_shape.end())) {
        fail("ONNX tensor shapes do not match the validated " + std::string(contract.name) + " contract.");
    }
}

std::vector<Ort::Value> run_once(
    Ort::Session& session,
    const Ort::MemoryInfo& memory_info,
    std::span<float> input_values,
    const ModelContract& contract) {
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_values.data(),
        input_values.size(),
        contract.input_shape.data(),
        contract.input_shape.size());
    const char* input_names[] = {contract.input_name.data()};
    const char* output_names[] = {contract.output_name.data()};
    return session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
}

void validate_output(const std::vector<Ort::Value>& outputs, const ModelContract& contract) {
    if (outputs.size() != 1 || !outputs.front().IsTensor()) {
        fail("ONNX Runtime did not return one raw output tensor.");
    }
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || tensor_shape(info) != std::vector<std::int64_t>(contract.output_shape.begin(), contract.output_shape.end())) {
        fail("ONNX Runtime output does not match float32 " + shape_description(contract.output_shape) + ".");
    }
}

std::size_t winning_class(
    std::span<const float> values,
    std::span<const std::int64_t> shape,
    const DetectionLayout& layout,
    std::size_t batch_index,
    std::size_t candidate_index) {
    const auto class_axis = static_cast<std::size_t>(layout.class_channel_axis);
    const auto candidate_axis = static_cast<std::size_t>(layout.candidate_axis);
    const auto batch_axis = 3U - class_axis - candidate_axis;
    const std::array<std::size_t, 3> strides{
        static_cast<std::size_t>(shape[1] * shape[2]),
        static_cast<std::size_t>(shape[2]),
        1};
    const auto base = batch_index * strides[batch_axis] + candidate_index * strides[candidate_axis];
    std::size_t winner = static_cast<std::size_t>(layout.class_channel_start);
    const auto class_stop = winner + static_cast<std::size_t>(layout.class_count);
    for (std::size_t channel = winner + 1; channel < class_stop; ++channel) {
        if (values[base + channel * strides[class_axis]] > values[base + winner * strides[class_axis]]) {
            winner = channel;
        }
    }
    return winner;
}

OutputParity compare_outputs(
    std::span<const float> reference, std::span<const float> candidate, const ModelContract& contract) {
    if (reference.size() != candidate.size() || reference.empty()) {
        fail("Reference and candidate logits must have the same non-zero size.");
    }
    double max_absolute_error{};
    double max_relative_error{};
    std::size_t reference_index{};
    std::size_t candidate_index{};
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double reference_value = static_cast<double>(reference[index]);
        const double candidate_value = static_cast<double>(candidate[index]);
        const double absolute_error = std::abs(candidate_value - reference_value);
        max_absolute_error = std::max(max_absolute_error, absolute_error);
        max_relative_error = std::max(
            max_relative_error,
            absolute_error / std::max(std::abs(reference_value), 1.0e-12));
        if (contract.task == Task::kClassification) {
            if (reference[index] > reference[reference_index]) reference_index = index;
            if (candidate[index] > candidate[candidate_index]) candidate_index = index;
        }
    }
    if (contract.task == Task::kClassification) {
        return {max_absolute_error, max_relative_error, reference_index == candidate_index ? 1.0 : 0.0};
    }

    if (!contract.detection_layout) {
        fail("The detection model contract is missing raw-output layout metadata.");
    }
    const auto& layout = *contract.detection_layout;
    if (layout.class_channel_axis < 0 || layout.class_channel_axis >= 3
        || layout.candidate_axis < 0 || layout.candidate_axis >= 3
        || layout.class_channel_axis == layout.candidate_axis) {
        fail("Detection class and candidate axes must be distinct rank-3 axes.");
    }
    const auto class_axis = static_cast<std::size_t>(layout.class_channel_axis);
    const auto candidate_axis = static_cast<std::size_t>(layout.candidate_axis);
    const auto batch_axis = 3U - class_axis - candidate_axis;
    const auto class_stop = layout.class_channel_start + layout.class_count;
    if (layout.class_channel_start < layout.box_coordinate_channels || layout.class_count <= 0
        || class_stop > contract.output_shape[class_axis]) {
        fail("Detection layout does not fit the configured raw output shape.");
    }
    const auto batches = static_cast<std::size_t>(contract.output_shape[batch_axis]);
    const auto candidates = static_cast<std::size_t>(contract.output_shape[candidate_axis]);
    std::size_t matching_classes{};
    for (std::size_t batch = 0; batch < batches; ++batch) {
        for (std::size_t location = 0; location < candidates; ++location) {
            if (winning_class(reference, contract.output_shape, layout, batch, location)
                == winning_class(candidate, contract.output_shape, layout, batch, location)) {
                ++matching_classes;
            }
        }
    }
    return {max_absolute_error, max_relative_error,
            static_cast<double>(matching_classes) / static_cast<double>(batches * candidates)};
}

void configure_session(Ort::SessionOptions& session_options) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(INFERENCE_BENCH_CUDA_RUNNER)
    int device_count{};
    const auto device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status != cudaSuccess || device_count < 1) {
        fail("CUDA device 0 is unavailable: " + std::string(cudaGetErrorString(device_count_status)));
    }
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(
        session_options, &cuda_options));
#endif
}

void synchronize_device() {
#if defined(INFERENCE_BENCH_CUDA_RUNNER)
    const auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fail("CUDA synchronization failed: " + std::string(cudaGetErrorString(status)));
    }
#endif
}

#if defined(INFERENCE_BENCH_CUDA_RUNNER)
int cuda_runtime_version() {
    int version{};
    const auto status = cudaRuntimeGetVersion(&version);
    if (status != cudaSuccess) {
        fail("Could not read the CUDA runtime version: " + std::string(cudaGetErrorString(status)));
    }
    return version;
}
#endif

double percentile(std::vector<double> values, double percentile_value) {
    std::sort(values.begin(), values.end());
    const double position = (static_cast<double>(values.size()) - 1.0) * percentile_value / 100.0;
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    if (lower_index == upper_index) {
        return values[lower_index];
    }
    return values[lower_index]
        + (values[upper_index] - values[lower_index]) * (position - static_cast<double>(lower_index));
}

LatencySummary summarize(std::vector<double> samples_ms) {
    if (samples_ms.empty()) {
        fail("At least one timed inference is required.");
    }
    for (const double sample : samples_ms) {
        if (!std::isfinite(sample) || sample <= 0.0) {
            fail("Latency samples must be finite positive numbers.");
        }
    }
    const auto [minimum, maximum] = std::minmax_element(samples_ms.begin(), samples_ms.end());
    const double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    const double mean = sum / static_cast<double>(samples_ms.size());
    const double p50 = percentile(samples_ms, 50.0);
    const double p95 = percentile(samples_ms, 95.0);
    const double p99 = percentile(samples_ms, 99.0);
    return {
        std::move(samples_ms),
        mean,
        *minimum,
        *maximum,
        p50,
        p95,
        p99,
    };
}

std::optional<std::uint64_t> process_rss_bytes() {
#if defined(__linux__)
    std::ifstream stream("/proc/self/status");
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream line_stream(line);
        std::string label;
        std::uint64_t kib{};
        std::string unit;
        line_stream >> label >> kib >> unit;
        if (label == "VmRSS:") {
            return kib * 1024U;
        }
    }
#endif
    return std::nullopt;
}

std::string json_escape(const std::string& value);

struct GpuTelemetry {
    bool available{};
    std::string name;
    std::string driver_version;
    std::optional<double> memory_used_mib;
    std::optional<double> utilization_percent;
    std::optional<double> power_watts;
    std::string reason;
};

#if defined(INFERENCE_BENCH_CUDA_RUNNER)
std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::optional<double> optional_double(const std::string& value) {
    try {
        std::size_t consumed{};
        const double parsed = std::stod(value, &consumed);
        if (consumed == value.size()) {
            return parsed;
        }
    } catch (const std::exception&) {
    }
    return std::nullopt;
}
#endif

GpuTelemetry sample_gpu_telemetry() {
#if defined(INFERENCE_BENCH_CUDA_RUNNER) && defined(__linux__)
    constexpr const char* command =
        "nvidia-smi --id=0 --query-gpu=name,driver_version,memory.used,utilization.gpu,power.draw "
        "--format=csv,noheader,nounits";
    FILE* pipe = popen(command, "r");
    if (pipe == nullptr) {
        return {false, {}, {}, std::nullopt, std::nullopt, std::nullopt,
                "nvidia-smi could not be started"};
    }
    std::array<char, 1024> buffer{};
    std::string line;
    if (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        line = buffer.data();
    }
    const int status = pclose(pipe);
    if (status != 0 || line.empty()) {
        return {false, {}, {}, std::nullopt, std::nullopt, std::nullopt,
                "nvidia-smi did not return telemetry for CUDA device 0"};
    }
    std::array<std::string, 5> fields;
    std::istringstream stream(line);
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (!std::getline(stream, fields[index], ',')) {
            return {false, {}, {}, std::nullopt, std::nullopt, std::nullopt,
                    "nvidia-smi returned an unexpected telemetry format"};
        }
        fields[index] = trim(std::move(fields[index]));
    }
    return {true, fields[0], fields[1], optional_double(fields[2]), optional_double(fields[3]),
            optional_double(fields[4]), {}};
#else
    return {false, {}, {}, std::nullopt, std::nullopt, std::nullopt, "CPU runner"};
#endif
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\b': escaped << "\\b"; break;
            case '\f': escaped << "\\f"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (character < 0x20U) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                            << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
                } else {
                    escaped << static_cast<char>(character);
                }
        }
    }
    return escaped.str();
}

void write_number(std::ostream& stream, double value) {
    stream << std::setprecision(12) << value;
}

void write_optional_number(std::ostream& stream, const std::optional<double>& value) {
    if (value) {
        write_number(stream, *value);
    } else {
        stream << "null";
    }
}

void write_gpu_telemetry(std::ostream& stream, const GpuTelemetry& telemetry) {
    if (!telemetry.available) {
        stream << "{\"status\": \"unavailable\", \"reason\": \""
               << json_escape(telemetry.reason) << "\"}";
        return;
    }
    stream << "{\"status\": \"available\", \"gpus\": [{\"name\": \""
           << json_escape(telemetry.name) << "\", \"driver_version\": \""
           << json_escape(telemetry.driver_version) << "\", \"memory_used_mib\": ";
    write_optional_number(stream, telemetry.memory_used_mib);
    stream << ", \"utilization_percent\": ";
    write_optional_number(stream, telemetry.utilization_percent);
    stream << ", \"power_watts\": ";
    write_optional_number(stream, telemetry.power_watts);
    stream << "}]}";
}

void write_latency_array(std::ostream& stream, const std::vector<double>& samples) {
    stream << '[';
    for (std::size_t index = 0; index < samples.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        write_number(stream, samples[index]);
    }
    stream << ']';
}

void write_shape(std::ostream& stream, std::span<const std::int64_t> shape) {
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) stream << ',';
        stream << shape[index];
    }
    stream << ']';
}

void write_result(
    const Options& options,
    const LatencySummary& latency,
    const std::vector<Ort::Value>& output,
    const OutputParity& parity,
    const ModelContract& contract,
    const GpuTelemetry& gpu_telemetry_before,
    const GpuTelemetry& gpu_telemetry_after) {
    const auto output_values = output.front().GetTensorData<float>();
    const double output_sum = std::accumulate(
        output_values, output_values + expected_elements(contract.output_shape), 0.0);
    const double throughput = 1000.0 / latency.mean_ms;
    const auto rss = process_rss_bytes();
    const auto artifact_size = std::filesystem::file_size(options.model_path);

    std::cout << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"runner\": {\n"
              << "    \"engine\": \"onnxruntime_cpp\",\n"
              << "    \"device\": \"" << kRunnerDevice << "\",\n"
              << "    \"active_providers\": [\"" << kPrimaryProvider << "\""
#if defined(INFERENCE_BENCH_CUDA_RUNNER)
              << ", \"CPUExecutionProvider\""
#endif
              << "],\n"
              << "    \"configuration\": {\n"
              << "      \"api\": \"onnxruntime_cxx_api\",\n"
              << "      \"language\": \"c++\",\n"
              << "      \"input_file\": \"" << json_escape(options.input_file.string()) << "\",\n"
              << "      \"reference_output_file\": \""
              << json_escape(options.reference_output_file.string()) << "\",\n"
              << "      \"output_shape\": ";
    write_shape(std::cout, contract.output_shape);
    std::cout << ",\n"
              << "      \"output_dtype\": \"float32\"";
    if (contract.task == Task::kDetection) {
        if (!contract.detection_layout) {
            fail("The detection model contract is missing raw-output layout metadata.");
        }
        const auto& layout = *contract.detection_layout;
        std::cout << ",\n      \"task\": \"detection\",\n"
                  << "      \"output\": \"" << json_escape(std::string(layout.output)) << "\",\n"
                  << "      \"box_coordinate_channels\": " << layout.box_coordinate_channels << ",\n"
                  << "      \"class_channel_axis\": " << layout.class_channel_axis << ",\n"
                  << "      \"candidate_axis\": " << layout.candidate_axis << ",\n"
                  << "      \"class_channel_start\": " << layout.class_channel_start << ",\n"
                  << "      \"class_count\": " << layout.class_count;
    }
    std::cout << ",\n"
              << "      \"output_sum\": ";
    write_number(std::cout, output_sum);
    std::cout << "\n    }\n  },\n"
              << "  \"model\": {\n"
              << "    \"name\": \"" << json_escape(options.model_name) << "\",\n"
              << "    \"input_shape\": ";
    write_shape(std::cout, contract.input_shape);
    std::cout << ",\n"
              << "    \"input_seed\": " << options.input_seed << ",\n"
              << "    \"model_seed\": ";
    if (options.model_seed) {
        std::cout << *options.model_seed;
    } else {
        std::cout << "null";
    }
    std::cout << ",\n"
              << "    \"artifact_path\": \"" << json_escape(options.model_path.string()) << "\",\n"
              << "    \"artifact_size_bytes\": " << artifact_size << "\n  },\n"
              << "  \"configuration\": {\n"
              << "    \"warmup_iterations\": " << options.warmup_iterations << ",\n"
              << "    \"timed_iterations\": " << options.timed_iterations << "\n  },\n"
              << "  \"measurement\": {\n"
              << "    \"latency_ms\": {\"mean\": ";
    write_number(std::cout, latency.mean_ms);
    std::cout << ", \"min\": ";
    write_number(std::cout, latency.min_ms);
    std::cout << ", \"max\": ";
    write_number(std::cout, latency.max_ms);
    std::cout << ", \"p50\": ";
    write_number(std::cout, latency.p50_ms);
    std::cout << ", \"p95\": ";
    write_number(std::cout, latency.p95_ms);
    std::cout << ", \"p99\": ";
    write_number(std::cout, latency.p99_ms);
    std::cout << ", \"samples\": ";
    write_latency_array(std::cout, latency.samples_ms);
    std::cout << "},\n    \"throughput_samples_per_second\": ";
    write_number(std::cout, throughput);
    std::cout << ",\n    \"process_rss\": ";
    if (rss) {
        std::cout << "{\"status\": \"available\", \"value\": " << *rss
                  << ", \"unit\": \"bytes\"}";
    } else {
        std::cout << "{\"status\": \"unavailable\", \"reason\": \"The native runner could not read process RSS.\"}";
    }
    std::cout << ",\n    \"gpu_telemetry\": {\n      \"before\": ";
    write_gpu_telemetry(std::cout, gpu_telemetry_before);
    std::cout << ",\n      \"after\": ";
    write_gpu_telemetry(std::cout, gpu_telemetry_after);
    std::cout << "\n    }\n  },\n"
              << "  \"correctness\": {\"parity\": {\"max_absolute_error\": ";
    write_number(std::cout, parity.max_absolute_error);
    std::cout << ", \"max_relative_error\": ";
    write_number(std::cout, parity.max_relative_error);
    std::cout << ", \"prediction_agreement\": ";
    write_number(std::cout, parity.prediction_agreement);
    std::cout << "}},\n"
              << "  \"environment\": {\n"
              << "    \"onnxruntime\": {\"status\": \"available\", \"version\": \"" << Ort::GetVersionString() << "\"},\n"
              << "    \"native_runner\": {\"status\": \"available\"}"
#if defined(INFERENCE_BENCH_CUDA_RUNNER)
              << ",\n    \"cuda\": {\"runtime_version\": " << cuda_runtime_version()
              << ", \"device_ordinal\": 0}"
#endif
              << "\n"
              << "  }\n"
              << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_arguments(argc, argv);
        if (!std::filesystem::is_regular_file(options.model_path)) {
            fail("ONNX model does not exist: " + options.model_path.string());
        }
        const auto& contract = model_contract(options.model_name);
        auto input_values = read_input_file(options.input_file, contract);
        const auto reference_output = read_reference_output_file(options.reference_output_file, contract);

        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "inference-bench"};
        Ort::SessionOptions session_options;
        configure_session(session_options);
        Ort::Session session{environment, options.model_path.c_str(), session_options};
        Ort::AllocatorWithDefaultOptions allocator;
        validate_interface(session, allocator, contract);
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (int index = 0; index < options.warmup_iterations; ++index) {
            validate_output(run_once(session, memory_info, input_values, contract), contract);
        }
        synchronize_device();

        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<std::size_t>(options.timed_iterations));
        std::vector<Ort::Value> final_output;
        const auto gpu_telemetry_before = sample_gpu_telemetry();
        for (int index = 0; index < options.timed_iterations; ++index) {
            const auto started_at = std::chrono::steady_clock::now();
            auto output = run_once(session, memory_info, input_values, contract);
            synchronize_device();
            const auto completed_at = std::chrono::steady_clock::now();
            validate_output(output, contract);
            samples_ms.push_back(std::chrono::duration<double, std::milli>(completed_at - started_at).count());
            final_output = std::move(output);
        }
        const auto gpu_telemetry_after = sample_gpu_telemetry();

        const auto output_values = final_output.front().GetTensorData<float>();
        const auto parity = compare_outputs(
            reference_output,
            std::span<const float>{output_values, expected_elements(contract.output_shape)},
            contract);
        write_result(
            options,
            summarize(std::move(samples_ms)),
            final_output,
            parity,
            contract,
            gpu_telemetry_before,
            gpu_telemetry_after);
        return EXIT_SUCCESS;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return EXIT_FAILURE;
}
