"""Tests for the shared binary input handoff used by native runners."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_bench.input_artifact import export_input_artifact
from inference_bench.inputs import DEFAULT_INPUT_SEED, make_input


class InputArtifactTests(unittest.TestCase):
    def test_artifact_preserves_the_seeded_float32_nchw_input_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "resnet50_input.bin"
            artifact = export_input_artifact("resnet50", path)

            expected = make_input("resnet50", seed=DEFAULT_INPUT_SEED).numpy()
            actual = np.fromfile(path, dtype="<f4").reshape(expected.shape)

            self.assertEqual(artifact.input_shape, (1, 3, 224, 224))
            self.assertEqual(artifact.input_seed, DEFAULT_INPUT_SEED)
            self.assertEqual(artifact.dtype, "float32")
            self.assertEqual(artifact.size_bytes, expected.nbytes)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
