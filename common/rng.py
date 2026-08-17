from __future__ import annotations

import hashlib
import os


class ShakeRNG:
    """Deterministic SHAKE256-based random source shared by all schemes.

    Instances are reproducible when constructed from an explicit seed.  The
    default constructors used by the schemes seed the generator from the OS.
    """

    _BLOCK_BYTES = 136

    def __init__(self, seed: bytes):
        if not isinstance(seed, bytes) or not seed:
            raise ValueError("seed must be a non-empty byte string")
        self._seed = seed
        self._counter = 0
        self._buffer = bytearray()

    @classmethod
    def from_system(cls, seed_bytes: int = 32) -> "ShakeRNG":
        return cls(os.urandom(seed_bytes))

    @classmethod
    def from_int(cls, seed: int) -> "ShakeRNG":
        if seed < 0:
            raise ValueError("seed must be non-negative")
        size = max(1, (seed.bit_length() + 7) // 8)
        return cls(seed.to_bytes(size, "little"))

    def _refill(self) -> None:
        block = hashlib.shake_256(
            b"LEE-BENCHMARK-SHAKE-RNG-v1"
            + len(self._seed).to_bytes(4, "little")
            + self._seed
            + self._counter.to_bytes(16, "little")
        ).digest(self._BLOCK_BYTES)
        self._counter += 1
        self._buffer.extend(block)

    def random_bytes(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("size must be non-negative")
        while len(self._buffer) < size:
            self._refill()
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def randbelow(self, upper_bound: int) -> int:
        """Uniform integer in range(upper_bound), using rejection sampling."""
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        bits = (upper_bound - 1).bit_length()
        size = max(1, (bits + 7) // 8)
        mask = (1 << bits) - 1
        while True:
            candidate = int.from_bytes(self.random_bytes(size), "little") & mask
            if candidate < upper_bound:
                return candidate

    def sample_positions(self, population_size: int, weight: int) -> list[int]:
        """Uniform sample without replacement using a partial Fisher-Yates map."""
        if not 0 <= weight <= population_size:
            raise ValueError("invalid sample weight")
        swaps: dict[int, int] = {}
        result: list[int] = []
        for index in range(weight):
            remaining = population_size - index
            selected = self.randbelow(remaining)
            value = swaps.get(selected, selected)
            last = remaining - 1
            swaps[selected] = swaps.get(last, last)
            result.append(value)
        return result
