from __future__ import annotations

import unittest

from experiments.cfs_decoder_capacity_sweep import fine_ratios, locate_transition, _points


class CapacitySweepDesignTests(unittest.TestCase):
    def test_weights_use_requested_n(self):
        points = _points("coarse", (0.005, 0.01, 0.10))
        self.assertEqual([point[2] for point in points], [82, 164, 1638])

    def test_transition_uses_largest_drop_when_no_90_10_bracket(self):
        rows = [
            {"target_ratio": 0.01, "syndrome_success_rate": 1.0},
            {"target_ratio": 0.02, "syndrome_success_rate": 0.8},
            {"target_ratio": 0.03, "syndrome_success_rate": 0.2},
        ]
        self.assertEqual(locate_transition(rows), (0.02, 0.03))

    def test_fine_ratios_are_internal(self):
        self.assertEqual(fine_ratios(0.02, 0.03, 4), (0.022, 0.024, 0.026, 0.028))


if __name__ == "__main__":
    unittest.main()
