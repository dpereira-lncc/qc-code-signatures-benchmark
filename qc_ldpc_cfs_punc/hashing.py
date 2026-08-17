from __future__ import annotations

import hashlib


_DOMAIN = b"PUNCTURED-QC-LDPC-SIGN-v1"


def hash_to_syndrome(message: bytes, counter: int, syndrome_bits: int) -> int:
    payload = (
        _DOMAIN
        + len(message).to_bytes(8, "little")
        + message
        + counter.to_bytes(8, "little")
    )
    size = (syndrome_bits + 7) // 8
    value = int.from_bytes(hashlib.shake_256(payload).digest(size), "little")
    return value & ((1 << syndrome_bits) - 1)
