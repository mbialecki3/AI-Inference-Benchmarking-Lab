"""Tests for the shared benchmark model contracts."""

import unittest

from inference_bench.models import available_models, build_model, get_model_spec


class ModelContractTests(unittest.TestCase):
    def test_mobilenet_v3_large_is_a_classification_contract(self) -> None:
        spec = get_model_spec("mobilenet_v3_large")

        self.assertEqual(available_models(), ("mobilenet_v3_large", "resnet50"))
        self.assertEqual(spec.task, "classification")
        self.assertEqual(spec.input_shape, (1, 3, 224, 224))
        self.assertIn("1000", spec.output_description)
        self.assertEqual(build_model(spec.name).classifier[-1].out_features, 1000)


if __name__ == "__main__":
    unittest.main()
