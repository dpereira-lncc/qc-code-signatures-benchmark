from __future__ import annotations

import unittest
import numpy as np

from lm_qcs_python.challenge import decode_challenge, encode_challenge
from lm_qcs_python.core import PublicKey, SecretKey, Signature, keygen, sign, verify
from lm_qcs_python.parameters import LMQCS_I, LMQCS_II, LMQCS_III
from lm_qcs_python.ring import cyclic_mul, poly_inverse
from common.rng import ShakeRNG


class LMQCSTests(unittest.TestCase):
    def test_reported_sizes(self):
        expected = {
            "LM-QCS-I": (1123, 519, 2106),
            "LM-QCS-II": (1766, 757, 3076),
            "LM-QCS-III": (2102, 1201, 3968),
        }
        for par in (LMQCS_I, LMQCS_II, LMQCS_III):
            self.assertEqual(
                (par.public_key_bytes, par.secret_key_bytes, par.signature_bytes),
                expected[par.name],
            )

    def test_ring_inverse_small(self):
        q = 101
        a = np.array([1, 2, 0, 1, 3], dtype=np.int64)
        inv = poly_inverse(a, q)
        product = cyclic_mul(a, inv, q)
        self.assertEqual(product[0], 1)
        self.assertTrue(np.all(product[1:] == 0))

    def test_challenge_roundtrip(self):
        par = LMQCS_I
        rng = np.random.default_rng(7)
        c = np.zeros(par.n, dtype=np.int64)
        pos = np.sort(rng.choice(par.n, par.omega_c, replace=False))
        c[pos] = rng.choice(np.array([-1, 1]), size=par.omega_c)
        value = encode_challenge(c, par.n, par.omega_c, par.challenge_bits)
        self.assertTrue(np.array_equal(c, decode_challenge(value, par.n, par.omega_c)))

    def test_complete_flow_128_and_serialization(self):
        par = LMQCS_I
        rng = ShakeRNG.from_int(42)
        message = b"LM-QCS test"
        keys = keygen(par, rng)
        sig = sign(message, keys.secret_key, keys.public_key, par, rng)
        self.assertTrue(verify(message, sig, keys.public_key, par))
        self.assertFalse(verify(message + b"!", sig, keys.public_key, par))

        pk2 = PublicKey.from_bytes(keys.public_key.to_bytes(par), par)
        sk2 = SecretKey.from_bytes(keys.secret_key.to_bytes(par), par)
        sig2 = Signature.from_bytes(sig.to_bytes(par), par)
        self.assertTrue(verify(message, sig2, pk2, par))
        self.assertTrue(np.array_equal(sk2.e1, keys.secret_key.e1))
        self.assertEqual(len(pk2.to_bytes(par)), 1123)
        self.assertEqual(len(sk2.to_bytes(par)), 519)
        self.assertEqual(len(sig2.to_bytes(par)), 2106)


if __name__ == "__main__":
    unittest.main()
