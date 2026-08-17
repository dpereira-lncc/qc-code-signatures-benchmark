from __future__ import annotations
from common.rng import ShakeRNG
from .ring import is_invertible

def sample_support(
    length: int,
    weight: int,
    rng: ShakeRNG | None = None,
) -> tuple[int, ...]:
    rng = rng or ShakeRNG.from_system()
    return tuple(sorted(rng.sample_positions(length, weight)))

def support_to_poly(support: tuple[int, ...]) -> int:
    value = 0
    for p in support:
        value |= 1 << p
    return value

def sample_sparse_poly(
    length: int,
    weight: int,
    rng: ShakeRNG | None = None,
):
    support = sample_support(length, weight, rng)
    return support_to_poly(support), support

def sample_invertible_sparse_poly(
    length: int,
    weight: int,
    rng: ShakeRNG | None = None,
    max_attempts: int = 10000,
):
    for _ in range(max_attempts):
        value, support = sample_sparse_poly(length, weight, rng)
        if is_invertible(value, length):
            return value, support
    raise RuntimeError('Failed to sample an invertible sparse polynomial.')

def sample_dense_invertible_poly(
    length: int,
    rng: ShakeRNG | None = None,
    max_attempts: int = 10000,
) -> int:
    rng = rng or ShakeRNG.from_system()
    size = (length + 7)//8
    mask = (1 << length) - 1
    for _ in range(max_attempts):
        value = int.from_bytes(rng.random_bytes(size), 'little') & mask
        if value.bit_count() % 2 == 0:
            value ^= 1
        if is_invertible(value, length):
            return value
    raise RuntimeError('Failed to sample an invertible dense polynomial.')
