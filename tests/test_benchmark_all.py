from __future__ import annotations

from types import SimpleNamespace
import unittest

from benchmarks.unified import SCENARIOS, Scenario, run_scenario


class BenchmarkResultTests(unittest.TestCase):
    def test_only_published_security_schemes_are_enabled_by_default(self):
        self.assertEqual(
            {
                scenario.implementation
                for scenario in SCENARIOS
                if scenario.enabled_by_default
            },
            {"hqcs_r_signature", "lm_qcs_python", "lmqcs_python"},
        )

    def test_all_requested_cfs_profiles_are_registered(self):
        by_identifier = {scenario.identifier: scenario for scenario in SCENARIOS}
        for implementation in ("qc_ldpc_cfs", "qc_ldpc_cfs_punc"):
            for variant in (
                "original",
                "estimated_128",
                "estimated_192",
                "estimated_256",
            ):
                identifier = f"{implementation}:{variant}"
                self.assertIn(identifier, by_identifier)
                scenario = by_identifier[identifier]
                self.assertFalse(scenario.enabled_by_default)
                self.assertEqual(scenario.sign_attempt_limit, 100)
                self.assertTrue(scenario.tolerate_sign_failure)

    def test_cfs_profiles_do_not_claim_validated_security(self):
        by_identifier = {scenario.identifier: scenario for scenario in SCENARIOS}
        for implementation in ("qc_ldpc_cfs", "qc_ldpc_cfs_punc"):
            self.assertIsNone(
                by_identifier[f"{implementation}:original"].security_bits
            )
            for level in (128, 192, 256):
                self.assertIsNone(
                    by_identifier[
                        f"{implementation}:estimated_{level}"
                    ].security_bits
                )

    def test_parameter_set_without_sign_attempt_limit(self):
        parameters = SimpleNamespace(name="without-attempt-limit")

        def keygen(par, rng):
            return SimpleNamespace(secret_key=object(), public_key=object())

        def sign(message, secret_key, public_key, par, rng):
            return SimpleNamespace(attempts=1)

        def verify(message, signature, public_key, par):
            return True

        scenario = Scenario(
            "test",
            "without-attempt-limit",
            None,
            parameters,
            keygen,
            sign,
            verify,
            123,
        )

        result = run_scenario(scenario, 1, b"A")

        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["sign_attempt_limit"])


if __name__ == "__main__":
    unittest.main()
