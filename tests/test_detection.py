"""Tests for the separate raw-output YOLO11n detection contract."""

import unittest

import numpy as np
import torch

from inference_bench.detection import (
    YOLO11N_INPUT_SHAPE,
    YOLO11N_OUTPUT_SHAPE,
    compare_detection_outputs,
    make_detection_input,
    raw_detection_tensor,
)


class DetectionContractTests(unittest.TestCase):
    def test_seeded_detection_input_is_repeatable_and_640_square(self) -> None:
        first = make_detection_input(seed=7)
        second = make_detection_input(seed=7)

        self.assertEqual(tuple(first.shape), YOLO11N_INPUT_SHAPE)
        self.assertEqual(YOLO11N_OUTPUT_SHAPE, (1, 84, 8400))
        self.assertTrue(torch.equal(first, second))

    def test_raw_detection_tensor_accepts_ultralytics_tuple_output(self) -> None:
        raw = torch.zeros((1, 84, 8400), dtype=torch.float32)
        self.assertIs(raw_detection_tensor((raw, [object()])), raw)

    def test_detection_parity_uses_per_candidate_class_agreement(self) -> None:
        reference = np.zeros((1, 6, 2), dtype=np.float32)
        reference[:, 4, :] = (0.9, 0.1)
        reference[:, 5, :] = (0.1, 0.9)
        candidate = reference.copy()
        candidate[:, 4:, 1] = candidate[:, 4:, 1][:, ::-1]

        parity = compare_detection_outputs(reference, candidate)

        self.assertEqual(parity.prediction_agreement, 0.5)
        self.assertGreater(parity.max_absolute_error, 0)

    def test_detection_parity_rejects_non_detection_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-3"):
            compare_detection_outputs(np.zeros((1, 1000)), np.zeros((1, 1000)))


if __name__ == "__main__":
    unittest.main()
