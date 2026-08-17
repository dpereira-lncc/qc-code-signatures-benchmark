from __future__ import annotations

from math import comb
from typing import Iterable

import numpy as np


def rank_combination(indices: Iterable[int], n: int, k: int) -> int:
    """Lexicographic rank of a k-subset of {0,...,n-1}."""
    selected = list(map(int, indices))
    if len(selected) != k or selected != sorted(selected):
        raise ValueError("Indices must be a sorted k-subset.")
    rank = 0
    previous = -1
    remaining = k
    for value in selected:
        for candidate in range(previous + 1, value):
            rank += comb(n - candidate - 1, remaining - 1)
        previous = value
        remaining -= 1
    return rank


def unrank_combination(rank: int, n: int, k: int) -> list[int]:
    total = comb(n, k)
    if not 0 <= rank < total:
        raise ValueError("Combination rank out of range.")
    result: list[int] = []
    start = 0
    remaining = k
    for _ in range(k):
        for candidate in range(start, n):
            count = comb(n - candidate - 1, remaining - 1)
            if rank < count:
                result.append(candidate)
                start = candidate + 1
                remaining -= 1
                break
            rank -= count
    return result


def encode_challenge(c: np.ndarray, n: int, k: int, out_bits: int) -> int:
    c = np.asarray(c, dtype=np.int64)
    if c.size != n or np.any((c < -1) | (c > 1)) or np.count_nonzero(c) != k:
        raise ValueError("Invalid challenge.")
    support = np.flatnonzero(c).tolist()
    support_rank = rank_combination(support, n, k)
    sign_rank = 0
    for i, pos in enumerate(support):
        if c[pos] > 0:
            sign_rank |= 1 << i
    value = support_rank * (1 << k) + sign_rank
    if value >= (1 << out_bits):
        raise ValueError("Challenge does not fit encoded width.")
    return value


def decode_challenge(value: int, n: int, k: int) -> np.ndarray:
    sign_mask = (1 << k) - 1
    sign_rank = value & sign_mask
    support_rank = value >> k
    support = unrank_combination(support_rank, n, k)
    c = np.zeros(n, dtype=np.int64)
    for i, pos in enumerate(support):
        c[pos] = 1 if ((sign_rank >> i) & 1) else -1
    return c
