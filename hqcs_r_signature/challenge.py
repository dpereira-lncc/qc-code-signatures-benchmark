from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .parameters import Parameters


@dataclass(frozen=True)
class Challenge:
    positions: tuple[int, ...]
    signs: tuple[int, ...]

    def dense(self, k: int, q: int) -> list[int]:
        result = [0] * k
        for position, sign in zip(self.positions, self.signs):
            result[position] = sign % q
        return result


def _encode_u64(value: int) -> bytes:
    return value.to_bytes(8, "little")


def hash_to_challenge(
    message: bytes,
    v_seed: bytes,
    quotient_vu: list[int],
    quotient_vuh: list[int],
    public_key_bytes: bytes,
    par: Parameters,
) -> Challenge:
    payload = bytearray()
    payload += b"HQCS-R-CHALLENGE-v1"
    payload += _encode_u64(len(message))
    payload += message
    payload += v_seed
    payload += _encode_u64(len(quotient_vu))
    for value in quotient_vu:
        payload += int(value).to_bytes(2, "little")
    for value in quotient_vuh:
        payload += int(value).to_bytes(2, "little")
    payload += public_key_bytes

    selected: set[int] = set()
    signs: dict[int, int] = {}
    counter = 0
    limit = (1 << 32) - ((1 << 32) % par.k)

    while len(selected) < par.omega_c:
        block = hashlib.shake_256(
            bytes(payload) + counter.to_bytes(8, "little")
        ).digest(1024)
        counter += 1

        for offset in range(0, len(block), 5):
            chunk = block[offset:offset + 5]
            if len(chunk) < 5:
                break
            candidate = int.from_bytes(chunk[:4], "little")
            if candidate >= limit:
                continue
            position = candidate % par.k
            if position in selected:
                continue
            selected.add(position)
            signs[position] = 1 if chunk[4] & 1 else -1
            if len(selected) == par.omega_c:
                break

    positions = tuple(sorted(selected))
    return Challenge(
        positions=positions,
        signs=tuple(signs[position] for position in positions),
    )
