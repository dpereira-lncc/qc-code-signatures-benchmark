from __future__ import annotations

import unittest

import numpy as np

from hqcs_r_signature.ring import cyclic_mul as hqcs_mul
from hqcs_r_signature.ring import invert as hqcs_invert
from lm_qcs_python.ring import cyclic_mul as lm_qcs_mul
from lm_qcs_python.ring import sparse_cyclic_mul as lm_qcs_sparse_mul
from lmqcs_python.ring import cyclic_mul as lmqcs_mul
from lmqcs_python.ring import sparse_cyclic_mul as lmqcs_sparse_mul
from common.rng import ShakeRNG


class ShakeRNGTests(unittest.TestCase):
    def test_reproducible_stream_and_sampling(self):
        first = ShakeRNG.from_int(1234)
        second = ShakeRNG.from_int(1234)
        self.assertEqual(first.random_bytes(257), second.random_bytes(257))
        self.assertEqual(
            first.sample_positions(100, 30),
            second.sample_positions(100, 30),
        )

    def test_sample_positions_has_exact_weight(self):
        positions = ShakeRNG.from_int(9).sample_positions(1000, 300)
        self.assertEqual(len(positions), 300)
        self.assertEqual(len(set(positions)), 300)


class SparseMultiplicationTests(unittest.TestCase):
    def _check_backend(self, dense_mul, sparse_mul):
        q = 4073
        sparse = np.array([0, -1, 0, 0, 1, 0, -1], dtype=np.int64)
        dense = np.array([11, 0, 31, 7, 0, 19, 2], dtype=np.int64)
        expected = dense_mul(sparse, dense, q)
        actual = sparse_mul(sparse, dense, q)
        self.assertTrue(np.array_equal(actual, expected))

    def test_lm_qcs_sparse_kernel(self):
        self._check_backend(lm_qcs_mul, lm_qcs_sparse_mul)

    def test_lmqcs_sparse_kernel(self):
        self._check_backend(lmqcs_mul, lmqcs_sparse_mul)


class HQCSCompiledInversionTests(unittest.TestCase):
    def test_large_modulus_without_int64_product_overflow(self):
        modulus = 32_230_149_377
        polynomial = [2, 0, 0, 0, 0]
        inverse = hqcs_invert(polynomial, modulus)
        self.assertEqual(hqcs_mul(polynomial, inverse, modulus), [1, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
