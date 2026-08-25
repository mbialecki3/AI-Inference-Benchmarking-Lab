"""Tests for the separate raw-output YOLO11n detection contract."""

import unittest

import numpy as np
import torch

from inference_bench.detection import (
    DetectionLayout,
    YOLO11N_INPUT_SHAPE,
    YOLO11N_LAYOUT,
    YOLO11N_OUTPUT_SHAPE,
    YOLO11S_INPUT_SHAPE,
    YOLO11S_OUTPUT_SHAPE,
    available_detection_models,
    compare_detection_outputs,
    get_detection_model_spec,
    make_detection_input,
    raw_detection_tensor,
)


class DetectionContractTests(unittest.TestCase):
    def test_yolo11n_is_registered_through_the_generic_detection_contract(self) -> None:
        spec = get_detection_model_spec("yolo11n")

        self.assertEqual(available_detection_models(), ("yolo11n", "yolo11s"))
        self.assertEqual(spec.input_shape, YOLO11N_INPUT_SHAPE)
        self.assertEqual(spec.output_shape, YOLO11N_OUTPUT_SHAPE)
        self.assertEqual(spec.layout, YOLO11N_LAYOUT)

    def test_yolo11s_reuses_the_explicit_static_raw_detection_contract(self) -> None:
        spec = get_detection_model_spec("yolo11s")

        self.assertEqual(spec.input_shape, YOLO11S_INPUT_SHAPE)
        self.assertEqual(spec.output_shape, YOLO11S_OUTPUT_SHAPE)
        self.assertEqual(spec.layout, YOLO11N_LAYOUT)

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

        parity = compare_detection_outputs(
            reference,
            candidate,
            layout=DetectionLayout(
                output="raw_predictions",
                box_coordinate_channels=4,
                class_channel_axis=1,
                candidate_axis=2,
                class_channel_start=4,
                class_count=2,
            ),
        )

        self.assertEqual(parity.prediction_agreement, 0.5)
        self.assertGreater(parity.max_absolute_error, 0)

    def test_detection_layout_metadata_and_non_default_class_axis_are_configurable(self) -> None:
        layout = DetectionLayout(
            output="raw_predictions",
            box_coordinate_channels=4,
            class_channel_axis=2,
            candidate_axis=1,
            class_channel_start=4,
            class_count=2,
        )
        reference = np.zeros((1, 2, 6), dtype=np.float32)
        reference[:, :, 4] = (0.9, 0.1)
        reference[:, :, 5] = (0.1, 0.9)
        candidate = reference.copy()
        candidate[:, 1, 4:] = candidate[:, 1, 4:][:, ::-1]

        parity = compare_detection_outputs(reference, candidate, layout=layout)

        self.assertEqual(parity.prediction_agreement, 0.5)
        self.assertEqual(layout.benchmark_metadata()["class_channel_axis"], 2)
        self.assertEqual(YOLO11N_LAYOUT.benchmark_metadata()["class_count"], 80)

    def test_detection_parity_rejects_non_detection_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-3"):
            compare_detection_outputs(np.zeros((1, 1000)), np.zeros((1, 1000)))

    def test_detection_layout_rejects_duplicate_normalized_axes(self) -> None:
        layout = DetectionLayout(
            output="raw_predictions",
            box_coordinate_channels=4,
            class_channel_axis=-1,
            candidate_axis=2,
            class_channel_start=4,
            class_count=2,
        )
        output = np.zeros((1, 2, 6), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "distinct output axes"):
            compare_detection_outputs(output, output, layout=layout)


if __name__ == "__main__":
    unittest.main()
