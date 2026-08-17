from __future__ import annotations

import unittest

import numpy as np

from lmqcs_python.core import PublicKey, SecretKey, Signature, keygen, sign, verify
from lmqcs_python.parameters import LMQCS128, LMQCS192, LMQCS256
from lmqcs_python.ring import cyclic_mul, poly_inverse
from common.rng import ShakeRNG


class LMQCSTests(unittest.TestCase):
    def test_parameter_sizes(self):
        expected = {
            LMQCS128.name: (2217, 370, 4434, 629),
            LMQCS192.name: (3241, 499, 6232, 730),
            LMQCS256.name: (4554, 651, 8457, 833),
        }
        for par in (LMQCS128, LMQCS192, LMQCS256):
            with self.subTest(par=par.name):
                pk, sk, sig, gamma = expected[par.name]
                self.assertEqual(par.public_key_bytes, pk)
                self.assertEqual(par.secret_key_bytes, sk)
                self.assertEqual(par.signature_bytes, sig)
                self.assertEqual(par.gamma, gamma)

    def test_ring_inverse_small(self):
        q = 101
        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        inv = poly_inverse(a, q)
        product = cyclic_mul(a, inv, q)
        self.assertEqual(product.tolist(), [1, 0, 0, 0, 0])

    def test_flow_128(self):
        par = LMQCS128
        rng = ShakeRNG.from_int(7)
        keys = keygen(par, rng)
        message = b"LMQCS unit test"
        signature = sign(message, keys.secret_key, keys.public_key, par, rng, max_attempts=100)
        self.assertTrue(verify(message, signature, keys.public_key, par))
        self.assertFalse(verify(message + b"!", signature, keys.public_key, par))

        pk_bytes = keys.public_key.to_bytes(par)
        sk_bytes = keys.secret_key.to_bytes(par)
        sig_bytes = signature.to_bytes(par)
        self.assertEqual(len(pk_bytes), par.public_key_bytes)
        self.assertEqual(len(sk_bytes), par.secret_key_bytes)
        self.assertEqual(len(sig_bytes), par.signature_bytes)
        self.assertTrue(verify(message, Signature.from_bytes(sig_bytes, par), PublicKey.from_bytes(pk_bytes, par), par))
        self.assertEqual(SecretKey.from_bytes(sk_bytes, par).parameter_name, par.name)


if __name__ == "__main__":
    unittest.main()
