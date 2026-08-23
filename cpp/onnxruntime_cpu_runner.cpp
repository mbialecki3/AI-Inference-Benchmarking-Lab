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

namespace {

constexpr std::string_view kModelName = "resnet50";
constexpr std::string_view kInputName = "images";
constexpr std::string_view kOutputName = "logits";
constexpr std::int64_t kDefaultInputSeed = 69420;
constexpr int kDefaultWarmupIterations = 5;
constexpr int kDefaultTimedIterations = 20;
constexpr std::array<std::int64_t, 4> kInputShape{1, 3, 224, 224};
constexpr std::array<std::int64_t, 2> kOutputShape{1, 1000};

struct Options {
    std::filesystem::path model_path{"artifacts/resnet50.onnx"};
    std::filesystem::path input_file;
    std::int64_t input_seed{kDefaultInputSeed};
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
    stream << "Usage: onnxruntime_cpu_runner --input-file PATH [options]\n"
           << "\n"
           << "Runs the validated ResNet-50 ONNX artifact with ONNX Runtime's CPU provider.\n"
           << "The input file must be the float32 NCHW binary emitted by\n"
           << "python -m inference_bench.input_artifact.\n"
           << "\n"
           << "Options:\n"
           << "  --model-path PATH    ONNX artifact (default: artifacts/resnet50.onnx)\n"
           << "  --input-file PATH    Required deterministic float32 input binary\n"
           << "  --input-seed N       Metadata only; must match input artifact (default: 69420)\n"
           << "  --warmup N           Warm synchronous inferences (default: 5)\n"
           << "  --iterations N       Timed synchronous inferences (default: 20)\n"
           << "  --help               Show this help text\n";
}

Options parse_arguments(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--help") {
            print_usage(std::cout);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--model") {
            const auto model = require_value(index, argc, argv, argument);
            if (model != kModelName) {
                fail("The initial native runner supports only model resnet50.");
            }
        } else if (argument == "--model-path") {
            options.model_path = require_value(index, argc, argv, argument);
        } else if (argument == "--input-file") {
            options.input_file = require_value(index, argc, argv, argument);
        } else if (argument == "--input-seed") {
            options.input_seed = parse_int64(require_value(index, argc, argv, argument), argument);
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
    return options;
}

std::size_t expected_input_elements() {
    return static_cast<std::size_t>(std::accumulate(
        kInputShape.begin(), kInputShape.end(), std::int64_t{1}, std::multiplies<>{}));
}

std::vector<float> read_input_file(const std::filesystem::path& path) {
    if constexpr (std::endian::native != std::endian::little) {
        fail("The native CPU runner currently requires a little-endian host.");
    }
    if (!std::filesystem::is_regular_file(path)) {
        fail("Input file does not exist: " + path.string());
    }
    const auto expected_bytes = expected_input_elements() * sizeof(float);
    const auto actual_bytes = std::filesystem::file_size(path);
    if (actual_bytes != expected_bytes) {
        fail("Input file has " + std::to_string(actual_bytes) + " bytes; expected "
             + std::to_string(expected_bytes) + " for float32 NCHW [1,3,224,224].");
    }

    std::vector<float> values(expected_input_elements());
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(actual_bytes));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(actual_bytes)) {
        fail("Could not read the complete input file: " + path.string());
    }
    return values;
}

template <typename TensorTypeAndShapeInfo>
std::vector<std::int64_t> tensor_shape(const TensorTypeAndShapeInfo& info) {
    return info.GetShape();
}

void validate_interface(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        fail("Expected exactly one ONNX input and output.");
    }
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    if (std::string_view{input_name.get()} != kInputName
        || std::string_view{output_name.get()} != kOutputName) {
        fail("Expected ONNX interface images -> logits.");
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
    if (tensor_shape(input_info) != std::vector<std::int64_t>(kInputShape.begin(), kInputShape.end())
        || tensor_shape(output_info) != std::vector<std::int64_t>(kOutputShape.begin(), kOutputShape.end())) {
        fail("ONNX tensor shapes do not match the validated ResNet-50 contract.");
    }
}

std::vector<Ort::Value> run_once(
    Ort::Session& session,
    const Ort::MemoryInfo& memory_info,
    std::span<float> input_values) {
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_values.data(),
        input_values.size(),
        kInputShape.data(),
        kInputShape.size());
    const char* input_names[] = {kInputName.data()};
    const char* output_names[] = {kOutputName.data()};
    return session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
}

void validate_output(const std::vector<Ort::Value>& outputs) {
    if (outputs.size() != 1 || !outputs.front().IsTensor()) {
        fail("ONNX Runtime did not return one logits tensor.");
    }
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || tensor_shape(info) != std::vector<std::int64_t>(kOutputShape.begin(), kOutputShape.end())) {
        fail("ONNX Runtime output does not match float32 logits [1,1000].");
    }
}

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

void write_result(
    const Options& options,
    const LatencySummary& latency,
    const std::vector<Ort::Value>& output) {
    const auto output_values = output.front().GetTensorData<float>();
    const double output_sum = std::accumulate(output_values, output_values + kOutputShape[1], 0.0);
    const double throughput = 1000.0 / latency.mean_ms;
    const auto rss = process_rss_bytes();
    const auto artifact_size = std::filesystem::file_size(options.model_path);

    std::cout << "{\n"
              << "  \"schema_version\": 1,\n"
              << "  \"runner\": {\n"
              << "    \"engine\": \"onnxruntime_cpp\",\n"
              << "    \"device\": \"cpu\",\n"
              << "    \"active_providers\": [\"CPUExecutionProvider\"],\n"
              << "    \"configuration\": {\n"
              << "      \"api\": \"onnxruntime_cxx_api\",\n"
              << "      \"language\": \"c++\",\n"
              << "      \"input_file\": \"" << json_escape(options.input_file.string()) << "\",\n"
              << "      \"output_shape\": [1, 1000],\n"
              << "      \"output_dtype\": \"float32\",\n"
              << "      \"output_sum\": ";
    write_number(std::cout, output_sum);
    std::cout << "\n    }\n  },\n"
              << "  \"model\": {\n"
              << "    \"name\": \"resnet50\",\n"
              << "    \"input_shape\": [1, 3, 224, 224],\n"
              << "    \"input_seed\": " << options.input_seed << ",\n"
              << "    \"model_seed\": null,\n"
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
    std::cout << ",\n    \"gpu_telemetry\": {\n"
              << "      \"before\": {\"status\": \"unavailable\", \"reason\": \"CPU runner\"},\n"
              << "      \"after\": {\"status\": \"unavailable\", \"reason\": \"CPU runner\"}\n"
              << "    }\n  },\n"
              << "  \"correctness\": {\"parity\": null},\n"
              << "  \"environment\": {\n"
              << "    \"onnxruntime\": {\"status\": \"available\", \"version\": \"" << Ort::GetVersionString() << "\"},\n"
              << "    \"native_runner\": {\"status\": \"partial\", \"note\": \"Host and driver capture will be added with the native CUDA path.\"}\n"
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
        auto input_values = read_input_file(options.input_file);

        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "inference-bench"};
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session{environment, options.model_path.c_str(), session_options};
        Ort::AllocatorWithDefaultOptions allocator;
        validate_interface(session, allocator);
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (int index = 0; index < options.warmup_iterations; ++index) {
            validate_output(run_once(session, memory_info, input_values));
        }

        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<std::size_t>(options.timed_iterations));
        std::vector<Ort::Value> final_output;
        for (int index = 0; index < options.timed_iterations; ++index) {
            const auto started_at = std::chrono::steady_clock::now();
            auto output = run_once(session, memory_info, input_values);
            const auto completed_at = std::chrono::steady_clock::now();
            validate_output(output);
            samples_ms.push_back(std::chrono::duration<double, std::milli>(completed_at - started_at).count());
            final_output = std::move(output);
        }

        write_result(options, summarize(std::move(samples_ms)), final_output);
        return EXIT_SUCCESS;
    } catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
    }
    return EXIT_FAILURE;
}
