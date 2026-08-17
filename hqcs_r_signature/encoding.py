from __future__ import annotations

from .challenge import Challenge
from .parameters import Parameters


def pack_values(values: list[int], bits: int) -> bytes:
    accumulator = 0
    used = 0
    output = bytearray()

    for value in values:
        if value < 0 or value >= (1 << bits):
            raise ValueError("Value outside the encoding range.")
        accumulator |= value << used
        used += bits

        while used >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            used -= 8

    if used:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_values(data: bytes, count: int, bits: int) -> list[int]:
    accumulator = 0
    used = 0
    offset = 0
    output: list[int] = []

    while len(output) < count:
        while used < bits:
            if offset >= len(data):
                raise ValueError("Dados insuficientes.")
            accumulator |= data[offset] << used
            used += 8
            offset += 1

        output.append(accumulator & ((1 << bits) - 1))
        accumulator >>= bits
        used -= bits

    return output


def encode_restricted(values: list[int], par: Parameters) -> bytes:
    shifted = [value + par.ell_e for value in values]
    return pack_values(shifted, par.secret_symbol_bits)


def decode_restricted(data: bytes, par: Parameters) -> list[int]:
    values = unpack_values(data, par.k, par.secret_symbol_bits)
    result = [value - par.ell_e for value in values]
    if any(abs(value) > par.ell_e for value in result):
        raise ValueError("Invalid secret coefficient.")
    return result


def encode_challenge(challenge: Challenge, par: Parameters) -> bytes:
    dense = [0] * par.k
    for position, sign in zip(challenge.positions, challenge.signs):
        dense[position] = 1 if sign == 1 else 2
    return pack_values(dense, 2)


def decode_challenge(data: bytes, par: Parameters) -> Challenge:
    values = unpack_values(data, par.k, 2)
    positions = []
    signs = []

    for index, value in enumerate(values):
        if value == 0:
            continue
        if value == 1:
            sign = 1
        elif value == 2:
            sign = -1
        else:
            raise ValueError("Invalid challenge symbol.")
        positions.append(index)
        signs.append(sign)

    if len(positions) != par.omega_c:
        raise ValueError("Invalid challenge weight.")

    return Challenge(tuple(positions), tuple(signs))



class BitWriter:
    def __init__(self) -> None:
        self.accumulator = 0
        self.used = 0
        self.output = bytearray()

    def write(self, value: int, bits: int) -> None:
        if value < 0 or value >= (1 << bits):
            raise ValueError("Value outside the valid range.")
        self.accumulator |= value << self.used
        self.used += bits

        while self.used >= 8:
            self.output.append(self.accumulator & 0xFF)
            self.accumulator >>= 8
            self.used -= 8

    def finish(self) -> bytes:
        if self.used:
            self.output.append(self.accumulator & 0xFF)
            self.accumulator = 0
            self.used = 0
        return bytes(self.output)


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0
        self.accumulator = 0
        self.used = 0

    def read(self, bits: int) -> int:
        while self.used < bits:
            if self.offset >= len(self.data):
                raise ValueError("Dados insuficientes.")
            self.accumulator |= self.data[self.offset] << self.used
            self.used += 8
            self.offset += 1

        value = self.accumulator & ((1 << bits) - 1)
        self.accumulator >>= bits
        self.used -= bits
        return value
