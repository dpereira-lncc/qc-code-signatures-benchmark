from __future__ import annotations

import unittest

from hqcs_r_signature.challenge import Challenge
from hqcs_r_signature.core import PublicKey, SecretKey, Signature, keygen, sign, verify
from hqcs_r_signature.encoding import decode_challenge, encode_challenge
from hqcs_r_signature.parameters import HQCS_R_1, HQCS_R_2, HQCS_R_3, HQCS_R_NIST_3, HQCS_R_NIST_5, Parameters
from hqcs_r_signature.ring import cyclic_mul, invert
from hqcs_r_signature.parameter_analysis import analyze


TEST = Parameters(
    name="HQCS-R-TEST",
    security_bits=32,
    k=31,
    q=65537,
    p=521,
    ell=126,
    ell_e=1,
    omega_c=3,
    claimed_acceptance=0.9,
    experimental_acceptance=0.9,
    max_sign_attempts=100,
)


class ParameterTests(unittest.TestCase):
    def test_declared_sizes(self):
        self.assertEqual(HQCS_R_1.public_key_bytes, 5888)
        self.assertEqual(HQCS_R_1.secret_key_bytes, 567)
        self.assertEqual(HQCS_R_1.signature_bytes, 6265)

        self.assertEqual(HQCS_R_2.public_key_bytes, 5888)
        self.assertEqual(HQCS_R_2.secret_key_bytes, 567)
        self.assertEqual(HQCS_R_2.signature_bytes, 6265)

        self.assertEqual(HQCS_R_3.public_key_bytes, 7520)
        self.assertEqual(HQCS_R_3.secret_key_bytes, 810)
        self.assertEqual(HQCS_R_3.signature_bytes, 7520)

        self.assertEqual(HQCS_R_NIST_3.public_key_bytes, 9116)
        self.assertEqual(HQCS_R_NIST_3.secret_key_bytes, 851)
        self.assertEqual(HQCS_R_NIST_3.signature_bytes, 9683)

        self.assertEqual(HQCS_R_NIST_5.public_key_bytes, 12530)
        self.assertEqual(HQCS_R_NIST_5.secret_key_bytes, 1134)
        self.assertEqual(HQCS_R_NIST_5.signature_bytes, 13286)

    def test_candidate_parameter_conditions(self):
        for par in (HQCS_R_NIST_3, HQCS_R_NIST_5):
            result = analyze(par)
            self.assertTrue(result["primality"]["p_is_prime"])
            self.assertTrue(result["primality"]["q_is_prime"])
            self.assertTrue(
                all(result["algebraic_conditions"].values())
            )
            self.assertGreaterEqual(
                result["challenge_entropy_bits"],
                2 * par.security_bits,
            )
            self.assertGreaterEqual(
                result["public_key_forgery_exponent_bits"],
                par.security_bits,
            )
            self.assertGreater(
                result["acceptance"]["acceptance_lower_bound"],
                0.996,
            )


class RingTests(unittest.TestCase):
    def test_inverse(self):
        a = [1, 1, 0] + [0] * (TEST.k - 3)
        inverse = invert(a, TEST.q)
        product = cyclic_mul(a, inverse, TEST.q)
        self.assertEqual(product[0], 1)
        self.assertTrue(all(value == 0 for value in product[1:]))


class EncodingTests(unittest.TestCase):
    def test_challenge_round_trip(self):
        challenge = Challenge((1, 5, 8), (1, -1, 1))
        encoded = encode_challenge(challenge, TEST)
        self.assertEqual(decode_challenge(encoded, TEST), challenge)


class FlowTests(unittest.TestCase):
    def test_complete_flow(self):
        message = b"HQCS-R test"
        keys = keygen(TEST)
        signature = sign(message, keys.secret_key, keys.public_key, TEST)

        self.assertTrue(verify(message, signature, keys.public_key, TEST))
        self.assertFalse(
            verify(b"altered", signature, keys.public_key, TEST)
        )

        pk = PublicKey.from_bytes(keys.public_key.to_bytes(TEST), TEST)
        sk = SecretKey.from_bytes(keys.secret_key.to_bytes(TEST), TEST)
        sig = Signature.from_bytes(signature.to_bytes(TEST), TEST)

        self.assertTrue(verify(message, sig, pk, TEST))
        self.assertEqual(sk.e1, keys.secret_key.e1)


if __name__ == "__main__":
    unittest.main()
