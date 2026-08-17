from __future__ import annotations

from common.rng import ShakeRNG


def sample_support(
    length: int,
    weight: int,
    rng: ShakeRNG | None = None,
) -> tuple[int, ...]:
    if not 0 <= weight <= length:
        raise ValueError("Invalid weight.")

    rng = rng or ShakeRNG.from_system()
    return tuple(sorted(rng.sample_positions(length, weight)))


def support_to_int(support: tuple[int, ...]) -> int:
    value = 0
    for position in support:
        value |= 1 << position
    return value


def sample_sparse_poly(
    length: int,
    weight: int,
    rng: ShakeRNG | None = None,
) -> tuple[int, tuple[int, ...]]:
    support = sample_support(length, weight, rng)
    return support_to_int(support), support


def sample_dense_bits(length: int, rng: ShakeRNG | None = None) -> int:
    rng = rng or ShakeRNG.from_system()
    byte_length = (length + 7) // 8
    return int.from_bytes(rng.random_bytes(byte_length), "little") & ((1 << length) - 1)


def sample_permutation(
    length: int,
    rng: ShakeRNG | None = None,
) -> tuple[int, ...]:
    rng = rng or ShakeRNG.from_system()
    values = list(range(length))
    for i in range(length - 1, 0, -1):
        j = rng.randbelow(i + 1)
        values[i], values[j] = values[j], values[i]
    return tuple(values)


def sample_row_xor_operations(
    row_count: int,
    count: int,
    rng: ShakeRNG | None = None,
) -> tuple[tuple[int, int], ...]:
    rng = rng or ShakeRNG.from_system()
    operations = []
    for _ in range(count):
        dst = rng.randbelow(row_count)
        src = rng.randbelow(row_count - 1)
        if src >= dst:
            src += 1
        operations.append((dst, src))
    return tuple(operations)
