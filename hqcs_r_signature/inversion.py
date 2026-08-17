from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(cache=True)
def _mul_mod(a: int, b: int, modulus: int) -> int:
    """Modular multiplication without overflowing signed int64."""
    x = a % modulus
    y = b % modulus
    result = 0
    while y:
        if y & 1:
            result = (result + x) % modulus
        x = (x + x) % modulus
        y >>= 1
    return result


@njit(cache=True)
def _pow_mod(base: int, exponent: int, modulus: int) -> int:
    result = 1
    value = base % modulus
    while exponent:
        if exponent & 1:
            result = _mul_mod(result, value, modulus)
        exponent >>= 1
        if exponent:
            value = _mul_mod(value, value, modulus)
    return result


@njit(cache=True)
def _degree(poly: np.ndarray) -> int:
    for index in range(poly.shape[0] - 1, -1, -1):
        if poly[index] != 0:
            return index
    return -1


@njit(cache=True)
def _poly_divmod(
    dividend: np.ndarray,
    divisor: np.ndarray,
    modulus: int,
) -> tuple[np.ndarray, np.ndarray]:
    size = dividend.shape[0]
    remainder = dividend.copy()
    quotient = np.zeros(size, dtype=np.int64)
    divisor_degree = _degree(divisor)
    if divisor_degree < 0:
        return quotient, remainder
    inverse_lead = _pow_mod(int(divisor[divisor_degree]), modulus - 2, modulus)
    remainder_degree = _degree(remainder)
    while remainder_degree >= divisor_degree:
        shift = remainder_degree - divisor_degree
        coefficient = _mul_mod(
            int(remainder[remainder_degree]), inverse_lead, modulus
        )
        quotient[shift] = coefficient
        for index in range(divisor_degree + 1):
            product = _mul_mod(coefficient, int(divisor[index]), modulus)
            remainder[index + shift] = (
                int(remainder[index + shift]) - product
            ) % modulus
        remainder_degree = _degree(remainder)
    return quotient, remainder


@njit(cache=True)
def _poly_mul(
    a: np.ndarray,
    b: np.ndarray,
    modulus: int,
) -> np.ndarray:
    size = a.shape[0]
    result = np.zeros(size, dtype=np.int64)
    degree_a = _degree(a)
    degree_b = _degree(b)
    if degree_a < 0 or degree_b < 0:
        return result
    for i in range(degree_a + 1):
        if a[i] == 0:
            continue
        maximum_j = min(degree_b, size - 1 - i)
        for j in range(maximum_j + 1):
            if b[j] != 0:
                product = _mul_mod(int(a[i]), int(b[j]), modulus)
                result[i + j] = (int(result[i + j]) + product) % modulus
    return result


@njit(cache=True)
def _invert_mod_xn_minus_one(
    coefficients: np.ndarray,
    modulus: int,
) -> tuple[np.ndarray, bool]:
    n = coefficients.shape[0]
    size = 2 * n + 2
    r0 = np.zeros(size, dtype=np.int64)
    r0[0] = modulus - 1
    r0[n] = 1
    r1 = np.zeros(size, dtype=np.int64)
    r1[:n] = coefficients
    s0 = np.zeros(size, dtype=np.int64)
    s1 = np.zeros(size, dtype=np.int64)
    s1[0] = 1

    while _degree(r1) >= 0:
        quotient, remainder = _poly_divmod(r0, r1, modulus)
        product = _poly_mul(quotient, s1, modulus)
        s2 = (s0 - product) % modulus
        r0, r1 = r1, remainder
        s0, s1 = s1, s2

    if _degree(r0) != 0 or r0[0] == 0:
        return np.zeros(n, dtype=np.int64), False

    inverse_gcd = _pow_mod(int(r0[0]), modulus - 2, modulus)
    result = np.zeros(n, dtype=np.int64)
    for index in range(size):
        product = _mul_mod(int(s0[index]), inverse_gcd, modulus)
        target = index % n
        result[target] = (int(result[target]) + product) % modulus
    return result, True


def invert_compiled(coefficients: Sequence[int], modulus: int) -> list[int]:
    values = np.asarray(coefficients, dtype=np.int64) % modulus
    inverse, ok = _invert_mod_xn_minus_one(values, modulus)
    if not ok:
        raise ValueError("Polynomial is not invertible in R_q.")
    return inverse.tolist()
