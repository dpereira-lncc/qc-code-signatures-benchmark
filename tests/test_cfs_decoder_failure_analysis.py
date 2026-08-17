from __future__ import annotations

import unittest
from dataclasses import replace

from experiments.cfs_decoder_failure_analysis import (
    default_block_sizes,
    diagnostic_decode,
    weight_limit_for,
)
from qc_ldpc_cfs.core import keygen
from qc_ldpc_cfs.hashing import hash_to_syndrome
from qc_ldpc_cfs.parameters import DEMO
from common.rng import ShakeRNG


class DecoderFailureAnalysisTests(unittest.TestCase):
    def test_block_sizes_include_requested_end_point(self):
        self.assertEqual(
            default_block_sizes(),
            (31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 80),
        )

    def test_weight_limit_is_a_diagnostic_bsc_bound(self):
        self.assertEqual(weight_limit_for(DEMO, None), None)
        limit = weight_limit_for(DEMO, 3.0)
        self.assertGreater(limit, DEMO.n * DEMO.channel_error_probability)
        self.assertLessEqual(limit, DEMO.n)

    def test_zero_syndrome_converges_with_correct_syndrome(self):
        keys = keygen(DEMO, ShakeRNG.from_int(991), warm_up_decoder=True)
        result = diagnostic_decode(
            keys.secret_key.decoder,
            0,
            weight_limit=None,
            stationary_patience=3,
        )
        self.assertEqual(result.category, 1)
        self.assertEqual(result.unsatisfied_checks, 0)

    def test_result_has_independently_checked_syndrome(self):
        parameters = replace(DEMO, max_bp_iterations=2)
        keys = keygen(parameters, ShakeRNG.from_int(992), warm_up_decoder=True)
        syndrome = hash_to_syndrome(b"diagnostic-test", 0, parameters.r)
        result = diagnostic_decode(
            keys.secret_key.decoder,
            syndrome,
            weight_limit=None,
            stationary_patience=2,
        )
        self.assertIn(result.category, (1, 2, 4, 5))
        if result.category == 1:
            self.assertEqual(result.unsatisfied_checks, 0)


if __name__ == "__main__":
    unittest.main()
