from __future__ import annotations

import unittest
from dataclasses import replace

from qc_ldpc_cfs.core import keygen as cfs_keygen
from qc_ldpc_cfs.core import sign as cfs_sign
from qc_ldpc_cfs.decoder import DecodeResult as CFSDecodeResult
from qc_ldpc_cfs.parameters import DEMO as CFS_DEMO
from qc_ldpc_cfs_punc.core import keygen as punc_keygen
from qc_ldpc_cfs_punc.core import sign as punc_sign
from qc_ldpc_cfs_punc.decoder import DecodeResult as PuncDecodeResult
from qc_ldpc_cfs_punc.parameters import DEMO as PUNC_DEMO
from common.rng import ShakeRNG
from common.errors import SigningFailure


class _AlwaysFailCFSDecoder:
    def decode(self, syndrome: int) -> CFSDecodeResult:
        return CFSDecodeResult(False, 0, 1, 0)


class _AlwaysFailPuncDecoder:
    def decode(self, syndrome: int) -> PuncDecodeResult:
        return PuncDecodeResult(False, 0, 1, 0)


class CFSStandardizationTests(unittest.TestCase):
    def test_cfs_keygen_is_reproducible_with_shake_rng(self):
        first = cfs_keygen(
            CFS_DEMO,
            ShakeRNG.from_int(7001),
            warm_up_decoder=False,
        )
        second = cfs_keygen(
            CFS_DEMO,
            ShakeRNG.from_int(7001),
            warm_up_decoder=False,
        )
        self.assertEqual(
            first.public_key.to_bytes(),
            second.public_key.to_bytes(),
        )

    def test_punctured_keygen_is_reproducible_with_shake_rng(self):
        first = punc_keygen(
            PUNC_DEMO,
            ShakeRNG.from_int(7002),
            warm_up_decoder=False,
        )
        second = punc_keygen(
            PUNC_DEMO,
            ShakeRNG.from_int(7002),
            warm_up_decoder=False,
        )
        self.assertEqual(
            first.public_key.to_bytes(),
            second.public_key.to_bytes(),
        )

    def test_cfs_signing_failure_is_bounded_and_reports_attempts(self):
        keys = cfs_keygen(
            CFS_DEMO,
            ShakeRNG.from_int(7003),
            warm_up_decoder=False,
        )
        secret_key = replace(
            keys.secret_key,
            decoder=_AlwaysFailCFSDecoder(),
        )
        with self.assertRaises(SigningFailure) as caught:
            cfs_sign(
                b"A" * 128,
                secret_key,
                keys.public_key,
                CFS_DEMO,
                ShakeRNG.from_int(7004),
                max_attempts=3,
            )
        self.assertEqual(caught.exception.attempts, 3)

    def test_punctured_signing_failure_is_bounded_and_reports_attempts(self):
        keys = punc_keygen(
            PUNC_DEMO,
            ShakeRNG.from_int(7005),
            warm_up_decoder=False,
        )
        secret_key = replace(
            keys.secret_key,
            decoder=_AlwaysFailPuncDecoder(),
        )
        with self.assertRaises(SigningFailure) as caught:
            punc_sign(
                b"A" * 128,
                secret_key,
                keys.public_key,
                PUNC_DEMO,
                ShakeRNG.from_int(7006),
                max_attempts=3,
            )
        self.assertEqual(caught.exception.attempts, 3)


if __name__ == "__main__":
    unittest.main()
