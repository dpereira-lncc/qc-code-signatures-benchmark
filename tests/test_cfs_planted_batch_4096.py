from __future__ import annotations

import unittest

from experiments.cfs_planted_batch_4096 import (
    BLOCK_SIZE,
    CSV_FIELDS,
    N,
    TARGET_WEIGHT,
)


class PlantedBatchConfigurationTests(unittest.TestCase):
    def test_requested_configuration(self):
        self.assertEqual(BLOCK_SIZE, 4096)
        self.assertEqual(N, 16384)
        self.assertEqual(TARGET_WEIGHT, 82)
        self.assertEqual(BLOCK_SIZE * 4, N)

    def test_csv_fields_are_exactly_the_requested_fields(self):
        self.assertEqual(CSV_FIELDS, (
            "key_id",
            "trial",
            "planted_weight",
            "success",
            "exact_recovery",
            "iterations",
            "final_residual_weight",
        ))


if __name__ == "__main__":
    unittest.main()
