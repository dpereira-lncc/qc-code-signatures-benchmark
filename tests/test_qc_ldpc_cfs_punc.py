from __future__ import annotations

import unittest

from qc_ldpc_cfs_punc.core import PublicKey, Signature, keygen, sign, verify
from qc_ldpc_cfs_punc.parameters import DEMO
from common.rng import ShakeRNG


class ConstructionTests(unittest.TestCase):
    def test_dimensions(self):
        keys = keygen(DEMO)
        construction = keys.secret_key.construction

        self.assertEqual(len(construction.punctured_rows), DEMO.punctured_r)
        self.assertEqual(len(construction.random_rows), DEMO.puncture_count)
        self.assertEqual(len(construction.modified_rows), DEMO.r)
        self.assertEqual(len(construction.permutation), DEMO.n)


class FlowTests(unittest.TestCase):
    def test_complete_flow(self):
        message = b"punctured qc-ldpc"
        rng = ShakeRNG.from_int(60101)
        keys = keygen(DEMO, rng)
        signature = sign(message, keys.secret_key, keys.public_key, DEMO, rng)

        self.assertTrue(verify(message, signature, keys.public_key, DEMO))
        self.assertFalse(
            verify(b"modified message", signature, keys.public_key, DEMO)
        )

        encoded_pk = keys.public_key.to_bytes()
        encoded_sig = signature.to_bytes(DEMO)

        decoded_pk = PublicKey.from_bytes(encoded_pk, DEMO)
        decoded_sig = Signature.from_bytes(encoded_sig, DEMO)

        self.assertTrue(verify(message, decoded_sig, decoded_pk, DEMO))
        self.assertEqual(len(encoded_pk), DEMO.public_key_bytes)
        self.assertEqual(len(encoded_sig), DEMO.signature_bytes)


if __name__ == "__main__":
    unittest.main()
