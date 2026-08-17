from __future__ import annotations

from typing import Iterable

import numpy as np


def pack_values(values: Iterable[int], bits: int) -> bytes:
    if bits <= 0:
        raise ValueError("bits must be positive.")
    out = bytearray()
    accumulator = 0
    accumulator_bits = 0
    limit = 1 << bits
    for raw in values:
        value = int(raw)
        if not 0 <= value < limit:
            raise ValueError(f"Value {value} does not fit in {bits} bits.")
        accumulator |= value << accumulator_bits
        accumulator_bits += bits
        while accumulator_bits >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            accumulator_bits -= 8
    if accumulator_bits:
        out.append(accumulator & 0xFF)
    return bytes(out)


def unpack_values(data: bytes, count: int, bits: int) -> np.ndarray:
    values = np.zeros(count, dtype=np.int64)
    accumulator = 0
    accumulator_bits = 0
    index = 0
    data_index = 0
    mask = (1 << bits) - 1
    while index < count:
        while accumulator_bits < bits:
            if data_index >= len(data):
                raise ValueError("Dados truncados.")
            accumulator |= data[data_index] << accumulator_bits
            accumulator_bits += 8
            data_index += 1
        values[index] = accumulator & mask
        accumulator >>= bits
        accumulator_bits -= bits
        index += 1
    return values


def encode_ternary(v: np.ndarray) -> np.ndarray:
    # -1 -> 0, 0 -> 1, 1 -> 2
    x = np.asarray(v, dtype=np.int64)
    if np.any((x < -1) | (x > 1)):
        raise ValueError("Non-ternary vector.")
    return x + 1


def decode_ternary(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.int64)
    if np.any(x > 2):
        raise ValueError("Invalid ternary encoding.")
    return x - 1


def pack_fields(fields: list[tuple[Iterable[int], int]]) -> bytes:
    """Pack fields of different widths without inter-field alignment."""
    out = bytearray()
    accumulator = 0
    accumulator_bits = 0
    for values, bits in fields:
        limit = 1 << bits
        for raw in values:
            value = int(raw)
            if not 0 <= value < limit:
                raise ValueError(f"Value {value} does not fit in {bits} bits.")
            accumulator |= value << accumulator_bits
            accumulator_bits += bits
            while accumulator_bits >= 8:
                out.append(accumulator & 0xFF)
                accumulator >>= 8
                accumulator_bits -= 8
    if accumulator_bits:
        out.append(accumulator & 0xFF)
    return bytes(out)


def unpack_fields(data: bytes, specifications: list[tuple[int, int]]) -> list[np.ndarray]:
    """Inverse of pack_fields; specifications contains (count, bits)."""
    accumulator = 0
    accumulator_bits = 0
    data_index = 0
    results: list[np.ndarray] = []
    for count, bits in specifications:
        mask = (1 << bits) - 1
        values = np.zeros(count, dtype=np.int64)
        for i in range(count):
            while accumulator_bits < bits:
                if data_index >= len(data):
                    raise ValueError("Dados truncados.")
                accumulator |= data[data_index] << accumulator_bits
                accumulator_bits += 8
                data_index += 1
            values[i] = accumulator & mask
            accumulator >>= bits
            accumulator_bits -= bits
        results.append(values)
    return results
