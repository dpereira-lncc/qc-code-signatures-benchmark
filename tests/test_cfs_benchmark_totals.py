from __future__ import annotations

from types import SimpleNamespace
import unittest

from benchmarks.unified import Scenario, run_scenario
from qc_ldpc_cfs.benchmark import _default_attempt_limit as cfs_attempt_limit
from qc_ldpc_cfs.parameters import DEMO as CFS_DEMO
from qc_ldpc_cfs.parameters import ESTIMATED_128 as CFS_ESTIMATED_128
from qc_ldpc_cfs.parameters import ORIGINAL_ARTICLE as CFS_ORIGINAL
from qc_ldpc_cfs_punc.benchmark import (
    _default_attempt_limit as punctured_attempt_limit,
)
from qc_ldpc_cfs_punc.parameters import DEMO as PUNCTURED_DEMO
from qc_ldpc_cfs_punc.parameters import ESTIMATED_128 as PUNCTURED_ESTIMATED_128
from qc_ldpc_cfs_punc.parameters import ORIGINAL_ARTICLE as PUNCTURED_ORIGINAL
from common.errors import SigningFailure


class CFSBenchmarkTotalsTests(unittest.TestCase):
    def test_local_benchmarks_bound_all_research_scale_profiles(self):
        self.assertEqual(cfs_attempt_limit(CFS_DEMO), CFS_DEMO.max_sign_attempts)
        self.assertEqual(cfs_attempt_limit(CFS_ORIGINAL), 100)
        self.assertEqual(cfs_attempt_limit(CFS_ESTIMATED_128), 100)
        self.assertEqual(
            punctured_attempt_limit(PUNCTURED_DEMO),
            PUNCTURED_DEMO.max_sign_attempts,
        )
        self.assertEqual(punctured_attempt_limit(PUNCTURED_ORIGINAL), 100)
        self.assertEqual(punctured_attempt_limit(PUNCTURED_ESTIMATED_128), 100)

    def test_failed_attempts_save_total_time_and_count(self):
        parameters = SimpleNamespace(
            name="cfs-test",
            max_sign_attempts=10,
        )

        def keygen(par, rng):
            return SimpleNamespace(secret_key=object(), public_key=object())

        def sign(
            message,
            secret_key,
            public_key,
            par,
            rng,
            *,
            max_attempts,
        ):
            raise SigningFailure("expected failure", max_attempts)

        def verify(message, signature, public_key, par):
            return True

        scenario = Scenario(
            "qc_ldpc_cfs",
            "test",
            None,
            parameters,
            keygen,
            sign,
            verify,
            123,
            False,
            100,
            True,
        )

        result = run_scenario(
            scenario,
            repetitions=1,
            message=b"A",
            sign_attempt_limit_override=3,
        )

        self.assertEqual(result["status"], "signing_incomplete")
        self.assertEqual(result["sign_attempt_limit"], 3)
        self.assertEqual(result["total_sign_attempts"], 3)
        self.assertGreaterEqual(result["total_sign_seconds"], 0.0)
        self.assertEqual(result["signing_runs"][0]["status"], "failed")
        self.assertEqual(result["signing_runs"][0]["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
