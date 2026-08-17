from __future__ import annotations

import unittest

from qc_ldpc_cfs.parameters import DEMO
from experiments.sensitivity_analysis import Case, _factorial_effects, build_cases


class SensitivityDesignTests(unittest.TestCase):
    def test_default_design_has_control_oat_and_full_factorial(self):
        cases = build_cases(DEMO)
        self.assertEqual(len(cases), 15)
        self.assertEqual(
            {phase: sum(case.phase == phase for case in cases) for phase in (
                "control", "A", "B", "C", "D"
            )},
            {"control": 1, "A": 2, "B": 2, "C": 2, "D": 8},
        )
        self.assertEqual(len({case.levels for case in cases if case.phase == "D"}), 8)

    def test_one_at_a_time_phases_hold_other_parameters_at_baseline(self):
        cases = build_cases(DEMO)
        for case in cases:
            if case.phase == "A":
                self.assertEqual(case.min_sum_normalization, DEMO.min_sum_normalization)
                self.assertEqual(case.max_bp_iterations, DEMO.max_bp_iterations)
            elif case.phase == "B":
                self.assertEqual(case.channel_error_probability, DEMO.channel_error_probability)
                self.assertEqual(case.max_bp_iterations, DEMO.max_bp_iterations)
            elif case.phase == "C":
                self.assertEqual(case.channel_error_probability, DEMO.channel_error_probability)
                self.assertEqual(case.min_sum_normalization, DEMO.min_sum_normalization)

    def test_factorial_main_effect_uses_high_minus_low(self):
        results = []
        for case in build_cases(DEMO, phases=("D",)):
            channel_level, _, _ = case.levels
            value = 10.0 if channel_level > 0 else 2.0
            results.append({
                "phase": "D",
                "factor_levels": case.levels,
                "sign_success_rate": value,
                "attempts_all_failures_capped": {"mean": value},
                "sign_seconds": {"mean": value},
            })
        effects = _factorial_effects(results)
        self.assertIsNotNone(effects)
        self.assertEqual(effects["sign_success_rate"]["channel"], 8.0)
        self.assertEqual(effects["sign_success_rate"]["normalization"], 0.0)
        self.assertEqual(effects["sign_success_rate"]["channel×iterations"], 0.0)

    def test_invalid_decoder_range_is_rejected(self):
        with self.assertRaises(ValueError):
            build_cases(DEMO, normalization_values=(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
