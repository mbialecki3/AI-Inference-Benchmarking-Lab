"""Tests for the static DeepLabV3 raw-logit segmentation contract."""

import unittest

import numpy as np
import torch

from inference_bench.segmentation import (
    DEEPLABV3_RESNET50_INPUT_SHAPE,
    DEEPLABV3_RESNET50_OUTPUT_SHAPE,
    available_segmentation_models,
    compare_segmentation_outputs,
    get_segmentation_model_spec,
    make_segmentation_input,
    raw_segmentation_tensor,
)


class SegmentationContractTests(unittest.TestCase):
    def test_deeplabv3_is_registered_with_static_raw_logits(self) -> None:
        spec = get_segmentation_model_spec("deeplabv3_resnet50")

        self.assertEqual(available_segmentation_models(), ("deeplabv3_resnet50",))
        self.assertEqual(spec.input_shape, DEEPLABV3_RESNET50_INPUT_SHAPE)
        self.assertEqual(spec.output_shape, DEEPLABV3_RESNET50_OUTPUT_SHAPE)
        self.assertEqual(spec.benchmark_metadata()["task"], "semantic_segmentation")
        self.assertEqual(spec.benchmark_metadata()["class_channel_axis"], 1)

    def test_seeded_input_and_torchvision_dictionary_output_are_normalized(self) -> None:
        first = make_segmentation_input("deeplabv3_resnet50", seed=7)
        second = make_segmentation_input("deeplabv3_resnet50", seed=7)
        logits = torch.zeros(DEEPLABV3_RESNET50_OUTPUT_SHAPE, dtype=torch.float32)

        self.assertEqual(tuple(first.shape), DEEPLABV3_RESNET50_INPUT_SHAPE)
        self.assertTrue(torch.equal(first, second))
        self.assertIs(raw_segmentation_tensor({"out": logits}), logits)

    def test_parity_uses_per_pixel_winning_class_agreement(self) -> None:
        spec = get_segmentation_model_spec("deeplabv3_resnet50")
        reference = np.zeros(spec.output_shape, dtype=np.float32)
        reference[:, 4, :, :] = 1.0
        candidate = reference.copy()
        candidate[:, 4, 0, 0] = 0.0
        candidate[:, 5, 0, 0] = 1.0

        parity = compare_segmentation_outputs(reference, candidate, spec=spec)

        self.assertEqual(parity.prediction_agreement, 1 - 1 / (224 * 224))
        self.assertEqual(parity.max_absolute_error, 1.0)

    def test_parity_rejects_a_non_static_segmentation_shape(self) -> None:
        spec = get_segmentation_model_spec("deeplabv3_resnet50")
        with self.assertRaisesRegex(ValueError, "static shape"):
            compare_segmentation_outputs(np.zeros((1, 21, 8, 8)), np.zeros((1, 21, 8, 8)), spec=spec)


if __name__ == "__main__":
    unittest.main()
