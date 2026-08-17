from __future__ import annotations

import hashlib
import struct

import numpy as np

from .encoding import pack_values
from .parameters import LMQCSParameters
from .ring import poly_inverse


def _serialize_ring(v: np.ndarray, par: LMQCSParameters) -> bytes:
    return pack_values((np.asarray(v, dtype=np.int64) % par.q).tolist(), par.q_bits)


def _hash_input(
    message: bytes,
    t: np.ndarray,
    u: np.ndarray,
    b_bar: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    par: LMQCSParameters,
) -> bytes:
    return b"LMQCS-H-v1" + struct.pack("<Q", len(message)) + message + b"".join(
        _serialize_ring(v, par) for v in (t, u, b_bar, h, b)
    )


def hash_to_challenge(
    message: bytes,
    t: np.ndarray,
    u: np.ndarray,
    b_bar: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    par: LMQCSParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Map data to an invertible ternary c of weight omega_c.

    The article requires these properties but does not specify the mapping.
    This implementation uses SHAKE256 with domain separation and rejection
    sampling.
    """
    seed = _hash_input(message, t, u, b_bar, h, b, par)
    for counter in range(1 << 32):
        shake = hashlib.shake_256(seed + struct.pack("<I", counter))
        raw = shake.digest(8 * par.n)
        scores = np.frombuffer(raw, dtype="<u8", count=par.n)
        positions = np.argpartition(scores, par.omega_c - 1)[: par.omega_c]
        sign_raw = shake.digest(8 * par.n + par.omega_c)[8 * par.n :]
        c = np.zeros(par.n, dtype=np.int64)
        for i, position in enumerate(positions):
            c[int(position)] = 1 if (sign_raw[i] & 1) else -1
        try:
            c_inv = poly_inverse(c, par.q)
            return c, c_inv
        except ValueError:
            continue
    raise RuntimeError("Failed to generate an invertible challenge.")
