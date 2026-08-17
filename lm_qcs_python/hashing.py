from __future__ import annotations

import hashlib
import struct

import numpy as np

from .encoding import pack_values
from .parameters import LMQCSParameters
from .ring import poly_inverse
from .challenge import decode_challenge


def _serialize_ring(v: np.ndarray, par: LMQCSParameters) -> bytes:
    return pack_values((np.asarray(v, dtype=np.int64) % par.q).tolist(), par.q_bits)


def _hash_input(message: bytes, v: np.ndarray, h: np.ndarray, par: LMQCSParameters) -> bytes:
    return (
        b"LM-QCS-H-v1"
        + struct.pack("<Q", len(message))
        + message
        + _serialize_ring(v, par)
        + _serialize_ring(h, par)
    )


def hash_to_challenge(
    message: bytes,
    v: np.ndarray,
    h: np.ndarray,
    par: LMQCSParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Hash to V_{1,omega_c} intersect R_q^*.

    The paper specifies the codomain and invertibility requirement but not the
    concrete mapping. We use SHAKE256, rejection sampling over the exact
    challenge space, and reject non-invertible polynomials.
    """
    seed = _hash_input(message, v, h, par)
    challenge_space = (1 << par.omega_c) * __import__("math").comb(par.n, par.omega_c)
    byte_len = (par.challenge_bits + 7) // 8
    for counter in range(1 << 32):
        raw = hashlib.shake_256(seed + struct.pack("<I", counter)).digest(byte_len)
        x = int.from_bytes(raw, "little")
        limit = (1 << (8 * byte_len)) - ((1 << (8 * byte_len)) % challenge_space)
        if x >= limit:
            continue
        c = decode_challenge(x % challenge_space, par.n, par.omega_c)
        try:
            return c, poly_inverse(c, par.q)
        except ValueError:
            continue
    raise RuntimeError("Failed to derive invertible challenge.")
