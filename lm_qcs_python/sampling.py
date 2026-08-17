from __future__ import annotations

import numpy as np

from common.rng import ShakeRNG


def sample_box(n: int, ell: int, rng: ShakeRNG) -> np.ndarray:
    return np.fromiter((rng.randbelow(2 * ell + 1) - ell for _ in range(n)), dtype=np.int64, count=n)


def sample_sparse_ternary(n: int, weight: int, rng: ShakeRNG) -> np.ndarray:
    if not 0 <= weight <= n:
        raise ValueError("Invalid ternary weight.")
    out = np.zeros(n, dtype=np.int64)
    positions = np.asarray(rng.sample_positions(n, weight), dtype=np.int64)
    signs = np.fromiter((1 if rng.randbelow(2) else -1 for _ in range(weight)), dtype=np.int64, count=weight)
    out[positions] = signs
    return out


def circular_shift(v: np.ndarray, r: int) -> np.ndarray:
    return np.roll(np.asarray(v, dtype=np.int64), int(r))
