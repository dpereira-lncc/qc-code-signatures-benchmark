from __future__ import annotations

import unittest

from qc_ldpc_cfs.core import PublicKey, Signature, keygen, sign, verify
from qc_ldpc_cfs.parameters import DEMO
from qc_ldpc_cfs.ring import cyclic_mul, poly_inverse_mod_xn_minus_one
from qc_ldpc_cfs.sampling import sample_invertible_sparse_poly
from common.rng import ShakeRNG


class RingTests(unittest.TestCase):
    def test_inverse(self):
        value, _ = sample_invertible_sparse_poly(31, 3)
        inverse = poly_inverse_mod_xn_minus_one(value, 31)
        self.assertEqual(cyclic_mul(value, inverse, 31), 1)


class DecoderTests(unittest.TestCase):
    def test_decoder_is_reused(self):
        keys = keygen(DEMO)
        self.assertIs(
            keys.secret_key.decoder,
            keys.secret_key.decoder,
        )
        self.assertGreater(
            keys.secret_key.decoder.edge_count,
            0,
        )


class FlowTests(unittest.TestCase):
    def test_complete_flow(self):
        message = b"teste qc-ldpc cfs"
        rng = ShakeRNG.from_int(50101)
        keys = keygen(DEMO, rng)
        signature = sign(
            message,
            keys.secret_key,
            keys.public_key,
            DEMO,
            rng,
        )

        self.assertTrue(
            verify(
                message,
                signature,
                keys.public_key,
                DEMO,
            )
        )
        self.assertFalse(
            verify(
                b"modified message",
                signature,
                keys.public_key,
                DEMO,
            )
        )

        encoded_signature = signature.to_bytes(DEMO)
        decoded_signature = Signature.from_bytes(
            encoded_signature,
            DEMO,
        )

        self.assertTrue(
            verify(
                message,
                decoded_signature,
                keys.public_key,
                DEMO,
            )
        )

        encoded_public_key = keys.public_key.to_bytes()
        decoded_public_key = PublicKey.from_bytes(
            encoded_public_key,
            DEMO,
        )

        self.assertEqual(
            decoded_public_key,
            keys.public_key,
        )
        self.assertEqual(
            len(keys.public_key.parity_check.blocks),
            4,
        )
        self.assertEqual(
            len(encoded_public_key),
            DEMO.public_key_bytes,
        )
        self.assertEqual(
            len(encoded_signature),
            DEMO.signature_bytes,
        )


if __name__ == "__main__":
    unittest.main()
