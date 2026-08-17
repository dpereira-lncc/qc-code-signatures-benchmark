from __future__ import annotations

import hashlib

from .parameters import Parameters
from .ring import invert
from common.rng import ShakeRNG


def _shake_stream(seed: bytes, domain: bytes, length: int) -> bytes:
    return hashlib.shake_256(domain + seed).digest(length)


def vector_from_seed(
    seed: bytes,
    par: Parameters,
    domain: bytes,
) -> list[int]:
    """
    Expand a seed into an approximately uniform vector in F_q^k,
    using rejection of 64-bit words.
    """
    output: list[int] = []
    counter = 0
    limit = (1 << 64) - ((1 << 64) % par.q)

    while len(output) < par.k:
        block = hashlib.shake_256(
            domain + seed + counter.to_bytes(8, "little")
        ).digest(4096)
        counter += 1

        for offset in range(0, len(block), 8):
            value = int.from_bytes(block[offset:offset + 8], "little")
            if value >= limit:
                continue
            output.append(value % par.q)
            if len(output) == par.k:
                break

    return output


def sample_uniform_vector(par: Parameters, rng: ShakeRNG) -> list[int]:
    return [rng.randbelow(par.q) for _ in range(par.k)]


def sample_restricted_vector(par: Parameters, rng: ShakeRNG) -> list[int]:
    return [
        rng.randbelow(2 * par.ell_e + 1) - par.ell_e
        for _ in range(par.k)
    ]


def sample_seeded_invertible_vector(
    par: Parameters,
    domain: bytes,
    rng: ShakeRNG,
) -> tuple[bytes, list[int], list[int]]:
    while True:
        seed = rng.random_bytes(par.seed_bytes)
        vector = vector_from_seed(seed, par, domain)
        try:
            inverse = invert(vector, par.q)
            return seed, vector, inverse
        except Exception:
            continue
