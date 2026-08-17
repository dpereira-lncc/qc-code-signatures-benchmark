from __future__ import annotations

import unittest

from qc_ldpc_cfs.parameters import DEMO
from experiments.sensitivity_cfs_demo_weights import (
    build_cases,
    validate_values,
)


class CFSWeightSensitivityTests(unittest.TestCase):
    def test_oat_has_one_control_and_nine_nonbaseline_cases(self):
        cases = build_cases(design="oat")
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case.case_id == "control" for case in cases), 1)
        baseline = {
            "channel_error_probability": DEMO.channel_error_probability,
            "min_sum_normalization": DEMO.min_sum_normalization,
            "secret_block_weight": DEMO.secret_block_weight,
            "q_block_weight": DEMO.q_block_weight,
        }
        for case in cases[1:]:
            differences = sum(
                case.parameters[name] != value for name, value in baseline.items()
            )
            self.assertEqual(differences, 1)

    def test_factorial_contains_full_default_cartesian_product(self):
        cases = build_cases(design="factorial")
        self.assertEqual(len(cases), 3 * 3 * 4 * 3)
        self.assertEqual(len({tuple(case.parameters.values()) for case in cases}), 108)
        self.assertEqual(sum(case.case_id == "control" for case in cases), 1)

    def test_even_q_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            validate_values((0.12,), (0.8,), (3,), (2,))

    def test_decoder_ranges_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_values((0.5,), (0.8,), (3,), (3,))
        with self.assertRaises(ValueError):
            validate_values((0.12,), (0.0,), (3,), (3,))


if __name__ == "__main__":
    unittest.main()
