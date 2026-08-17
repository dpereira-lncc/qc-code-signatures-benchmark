from __future__ import annotations

from functools import lru_cache
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
def _cyclic_mul_numba(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    n = a.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ai = int(a[i])
        if ai == 0:
            continue
        for j in range(n):
            bj = int(b[j])
            if bj != 0:
                k = i + j
                if k >= n:
                    k -= n
                out[k] = (out[k] + ai * bj) % q
    return out


def mod_q(v: Sequence[int] | np.ndarray, q: int) -> np.ndarray:
    return np.asarray(v, dtype=np.int64) % q


def symmetric(v: Sequence[int] | np.ndarray, q: int) -> np.ndarray:
    x = mod_q(v, q)
    half = (q - 1) // 2
    return np.where(x > half, x - q, x).astype(np.int64)


def cyclic_mul(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray, q: int) -> np.ndarray:
    aa = mod_q(a, q)
    bb = mod_q(b, q)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("a and b must be one-dimensional vectors of equal length.")
    return _cyclic_mul_numba(aa, bb, q)


@njit(cache=True)
def _sparse_cyclic_mul_numba(
    positions: np.ndarray,
    coefficients: np.ndarray,
    dense: np.ndarray,
    q: int,
) -> np.ndarray:
    n = dense.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for sparse_index in range(positions.shape[0]):
        position = int(positions[sparse_index])
        coefficient = int(coefficients[sparse_index])
        for dense_index in range(n):
            value = int(dense[dense_index])
            if value != 0:
                index = position + dense_index
                if index >= n:
                    index -= n
                out[index] = (out[index] + coefficient * value) % q
    return out


def sparse_cyclic_mul(
    sparse: Sequence[int] | np.ndarray,
    dense: Sequence[int] | np.ndarray,
    q: int,
) -> np.ndarray:
    sparse_array = np.asarray(sparse, dtype=np.int64)
    dense_array = mod_q(dense, q)
    if sparse_array.shape != dense_array.shape or sparse_array.ndim != 1:
        raise ValueError("sparse and dense must be one-dimensional arrays of equal size")
    positions = np.flatnonzero(sparse_array).astype(np.int64)
    coefficients = sparse_array[positions]
    return _sparse_cyclic_mul_numba(positions, coefficients, dense_array, q)


def cyclic_pow(a: Sequence[int] | np.ndarray, exponent: int, q: int) -> np.ndarray:
    if exponent < 0:
        return cyclic_pow(poly_inverse(a, q), -exponent, q)
    base = mod_q(a, q)
    result = np.zeros_like(base)
    result[0] = 1
    e = exponent
    while e:
        if e & 1:
            result = cyclic_mul(result, base, q)
        e >>= 1
        if e:
            base = cyclic_mul(base, base, q)
    return result


def poly_inverse(a: Sequence[int] | np.ndarray, q: int) -> np.ndarray:
    """Inverte a em F_q[x]/(x^n-1) por Euclides estendido compilado."""
    coeffs = mod_q(a, q)
    inverse, ok = _poly_inverse_xn1_numba(coeffs, q)
    if not ok:
        raise ValueError("Polynomial is not invertible in R_q.")
    return inverse


def is_invertible(a: Sequence[int] | np.ndarray, q: int) -> bool:
    try:
        poly_inverse(a, q)
        return True
    except ValueError:
        return False


def norm_inf(v: Sequence[int] | np.ndarray, q: int | None = None) -> int:
    x = symmetric(v, q) if q is not None else np.asarray(v, dtype=np.int64)
    return int(np.max(np.abs(x))) if len(x) else 0


def norm_l1(v: Sequence[int] | np.ndarray, q: int | None = None) -> int:
    x = symmetric(v, q) if q is not None else np.asarray(v, dtype=np.int64)
    return int(np.sum(np.abs(x), dtype=np.int64))

@njit(cache=True)
def _mod_inverse_int(a: int, p: int) -> int:
    a %= p
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r != 1:
        return -1
    return t % p


@njit(cache=True)
def _degree(poly: np.ndarray) -> int:
    for i in range(poly.shape[0] - 1, -1, -1):
        if poly[i] != 0:
            return i
    return -1


@njit(cache=True)
def _poly_divmod_dense(a: np.ndarray, b: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    size = a.shape[0]
    remainder = a.copy()
    quotient = np.zeros(size, dtype=np.int64)
    db = _degree(b)
    if db < 0:
        return quotient, remainder
    inv_lead = _mod_inverse_int(int(b[db]), p)
    dr = _degree(remainder)
    while dr >= db:
        shift = dr - db
        coefficient = (int(remainder[dr]) * inv_lead) % p
        quotient[shift] = coefficient
        for j in range(db + 1):
            remainder[j + shift] = (remainder[j + shift] - coefficient * int(b[j])) % p
        dr = _degree(remainder)
    return quotient, remainder


@njit(cache=True)
def _poly_mul_dense(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    size = a.shape[0]
    out = np.zeros(size, dtype=np.int64)
    da = _degree(a)
    db = _degree(b)
    if da < 0 or db < 0:
        return out
    for i in range(da + 1):
        ai = int(a[i])
        if ai == 0:
            continue
        max_j = min(db, size - 1 - i)
        for j in range(max_j + 1):
            bj = int(b[j])
            if bj != 0:
                out[i + j] = (out[i + j] + ai * bj) % p
    return out


@njit(cache=True)
def _poly_inverse_xn1_numba(a: np.ndarray, p: int) -> tuple[np.ndarray, bool]:
    n = a.shape[0]
    size = 2 * n + 2
    r0 = np.zeros(size, dtype=np.int64)
    r0[0] = p - 1
    r0[n] = 1
    r1 = np.zeros(size, dtype=np.int64)
    for i in range(n):
        r1[i] = int(a[i]) % p
    s0 = np.zeros(size, dtype=np.int64)
    s1 = np.zeros(size, dtype=np.int64)
    s1[0] = 1

    while _degree(r1) >= 0:
        quotient, remainder = _poly_divmod_dense(r0, r1, p)
        product = _poly_mul_dense(quotient, s1, p)
        s2 = (s0 - product) % p
        r0, r1 = r1, remainder
        s0, s1 = s1, s2

    if _degree(r0) != 0 or r0[0] == 0:
        return np.zeros(n, dtype=np.int64), False
    inv_gcd = _mod_inverse_int(int(r0[0]), p)
    out = np.zeros(n, dtype=np.int64)
    for i in range(size):
        out[i % n] = (out[i % n] + int(s0[i]) * inv_gcd) % p
    return out, True


def poly_inverse_fast(a: Sequence[int] | np.ndarray, q: int) -> np.ndarray:
    coeffs = mod_q(a, q)
    inverse, ok = _poly_inverse_xn1_numba(coeffs, q)
    if not ok:
        raise ValueError("Polynomial is not invertible in R_q.")
    return inverse
