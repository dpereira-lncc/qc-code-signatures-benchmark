from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.cfs_planted_error_experiment import (
    expected_error_weight,
    sample_exact_weight_error,
)
from qc_ldpc_cfs.parameters import DEMO


class PlantedErrorTests(unittest.TestCase):
    def test_requested_expected_weights(self):
        small = replace(DEMO, block_size=80)
        large = replace(DEMO, block_size=4096)
        self.assertEqual(expected_error_weight(small), 2)
        self.assertEqual(expected_error_weight(large), 82)

    def test_sample_has_exact_weight(self):
        error = sample_exact_weight_error(320, 2, 1234)
        self.assertEqual(error.bit_count(), 2)
        self.assertEqual(error, sample_exact_weight_error(320, 2, 1234))


if __name__ == "__main__":
    unittest.main()
