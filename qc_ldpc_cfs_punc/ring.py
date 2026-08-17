from __future__ import annotations

import numpy as np


def mask_for_n(n: int) -> int:
    return (1 << n) - 1


def rotate_left(value: int, shift: int, n: int) -> int:
    shift %= n
    value &= mask_for_n(n)
    if shift == 0:
        return value
    return ((value << shift) & mask_for_n(n)) | (value >> (n - shift))


def cyclic_mul(a: int, b: int, n: int) -> int:
    a &= mask_for_n(n)
    b &= mask_for_n(n)
    if a.bit_count() > b.bit_count():
        a, b = b, a

    result = 0
    while a:
        lsb = a & -a
        shift = lsb.bit_length() - 1
        result ^= rotate_left(b, shift, n)
        a ^= lsb
    return result & mask_for_n(n)


def int_to_bits(value: int, n: int) -> np.ndarray:
    data = value.to_bytes((n + 7) // 8, "little")
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8),
        bitorder="little",
    )[:n].astype(np.uint8, copy=False)


def bits_to_int(bits: np.ndarray) -> int:
    array = np.asarray(bits, dtype=np.uint8)
    packed = np.packbits(array, bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")
