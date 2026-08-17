from __future__ import annotations

import unittest

from experiments.cfs_sign_attempt_instrumentation import (
    BLOCK_SIZE,
    DEFAULT_ATTEMPTS,
    N,
    TARGET_WEIGHT,
)


class SignInstrumentationConfigurationTests(unittest.TestCase):
    def test_requested_configuration(self):
        self.assertEqual(BLOCK_SIZE, 4096)
        self.assertEqual(N, 16384)
        self.assertEqual(TARGET_WEIGHT, 82)
        self.assertEqual(DEFAULT_ATTEMPTS, 10_000)


if __name__ == "__main__":
    unittest.main()
