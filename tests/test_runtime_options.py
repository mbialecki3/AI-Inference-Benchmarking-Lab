"""Tests for serialized runtime optimization experiment settings."""

import unittest

from inference_bench.runtime_options import OnnxRuntimeOptions, OpenVinoOptions


class RuntimeOptionsTests(unittest.TestCase):
    def test_onnx_runtime_options_apply_session_and_cuda_provider_controls(self) -> None:
        options = OnnxRuntimeOptions(
            graph_optimization_level="extended",
            execution_mode="parallel",
            intra_op_num_threads=2,
            inter_op_num_threads=3,
            cuda_conv_algorithm="heuristic",
        )
        session = options.session_options()

        self.assertEqual(session.intra_op_num_threads, 2)
        self.assertEqual(session.inter_op_num_threads, 3)
        self.assertEqual(options.providers("cuda:0")[0][1]["cudnn_conv_algo_search"], "HEURISTIC")
        self.assertEqual(options.summary()["execution_mode"], "parallel")

    def test_openvino_options_build_latency_and_lower_precision_configurations(self) -> None:
        options = OpenVinoOptions(performance_hint="throughput", inference_precision="bf16")

        self.assertEqual(options.summary(), {"performance_hint": "throughput", "inference_precision": "bf16"})
        self.assertEqual(len(options.compile_configuration()), 2)

    def test_invalid_experimental_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "graph optimization"):
            OnnxRuntimeOptions(graph_optimization_level="invalid").session_options()
        with self.assertRaisesRegex(ValueError, "inference_precision"):
            OpenVinoOptions(inference_precision="int8").compile_configuration()


if __name__ == "__main__":
    unittest.main()
