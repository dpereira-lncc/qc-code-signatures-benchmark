from __future__ import annotations

from typing import Iterable
import numpy as np


def pack_values(values: Iterable[int], bits: int) -> bytes:
    out = bytearray(); acc = 0; acc_bits = 0; limit = 1 << bits
    for raw in values:
        value = int(raw)
        if not 0 <= value < limit:
            raise ValueError(f"Value {value} does not fit in {bits} bits.")
        acc |= value << acc_bits; acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF); acc >>= 8; acc_bits -= 8
    if acc_bits: out.append(acc & 0xFF)
    return bytes(out)


def unpack_values(data: bytes, count: int, bits: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.int64); acc = 0; acc_bits = 0; di = 0; mask = (1 << bits) - 1
    for i in range(count):
        while acc_bits < bits:
            if di >= len(data): raise ValueError("Truncated data.")
            acc |= data[di] << acc_bits; acc_bits += 8; di += 1
        out[i] = acc & mask; acc >>= bits; acc_bits -= bits
    return out


def pack_fields(fields: list[tuple[Iterable[int], int]]) -> bytes:
    out = bytearray(); acc = 0; acc_bits = 0
    for values, bits in fields:
        limit = 1 << bits
        for raw in values:
            value = int(raw)
            if not 0 <= value < limit: raise ValueError("Field value out of range.")
            acc |= value << acc_bits; acc_bits += bits
            while acc_bits >= 8:
                out.append(acc & 0xFF); acc >>= 8; acc_bits -= 8
    if acc_bits: out.append(acc & 0xFF)
    return bytes(out)


def unpack_fields(data: bytes, specs: list[tuple[int, int]]) -> list[np.ndarray]:
    acc = 0; acc_bits = 0; di = 0; results=[]
    for count,bits in specs:
        vals=np.zeros(count,dtype=object if bits > 62 else np.int64); mask=(1<<bits)-1
        for i in range(count):
            while acc_bits<bits:
                if di>=len(data): raise ValueError("Truncated data.")
                acc |= data[di]<<acc_bits; acc_bits+=8; di+=1
            vals[i]=acc&mask; acc>>=bits; acc_bits-=bits
        results.append(vals)
    return results
