from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseBinaryParityCheck:
    rows: tuple[int, ...]
    n: int

    @property
    def r(self) -> int:
        return len(self.rows)

    def syndrome(self, vector: int) -> int:
        syndrome = 0
        for i, row in enumerate(self.rows):
            syndrome |= ((row & vector).bit_count() & 1) << i
        return syndrome

    def to_bytes(self) -> bytes:
        row_size = (self.n + 7) // 8
        return b"".join(row.to_bytes(row_size, "little") for row in self.rows)

    @classmethod
    def from_bytes(cls, data: bytes, r: int, n: int) -> "DenseBinaryParityCheck":
        row_size = (n + 7) // 8
        if len(data) != r * row_size:
            raise ValueError("Invalid public matrix size.")

        mask = (1 << n) - 1
        rows = []
        for i in range(r):
            start = i * row_size
            rows.append(int.from_bytes(data[start:start + row_size], "little") & mask)
        return cls(tuple(rows), n)


def apply_row_operations_to_rows(
    rows: list[int],
    operations: tuple[tuple[int, int], ...],
) -> list[int]:
    result = list(rows)
    for dst, src in operations:
        result[dst] ^= result[src]
    return result


def apply_row_operations_to_vector(
    vector: int,
    operations: tuple[tuple[int, int], ...],
) -> int:
    result = vector
    for dst, src in operations:
        bit = (result >> src) & 1
        if bit:
            result ^= 1 << dst
    return result


def apply_inverse_row_operations_to_vector(
    vector: int,
    operations: tuple[tuple[int, int], ...],
) -> int:
    result = vector
    for dst, src in reversed(operations):
        bit = (result >> src) & 1
        if bit:
            result ^= 1 << dst
    return result


def permute_vector_secret_to_public(secret_vector: int, permutation: tuple[int, ...]) -> int:
    """
    permutation[i] = public position that receives secret bit i.
    """
    public_vector = 0
    value = secret_vector
    while value:
        lsb = value & -value
        secret_index = lsb.bit_length() - 1
        public_vector |= 1 << permutation[secret_index]
        value ^= lsb
    return public_vector


def permute_row_secret_to_public(secret_row: int, permutation: tuple[int, ...]) -> int:
    return permute_vector_secret_to_public(secret_row, permutation)
