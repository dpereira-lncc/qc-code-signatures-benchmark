from __future__ import annotations

from dataclasses import dataclass

from .ring import cyclic_mul


@dataclass(frozen=True)
class QCParityCheck:
    """
    Matriz QC formada por uma linha de blocos circulantes:

        H = [H0 | H1 | ... | H_{n0-1}]
    """

    blocks: tuple[int, ...]
    block_size: int

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def n(self) -> int:
        return self.block_count * self.block_size

    @property
    def r(self) -> int:
        return self.block_size

    def syndrome_int(self, vector: int) -> int:
        mask = (1 << self.block_size) - 1
        syndrome = 0

        for index, block in enumerate(self.blocks):
            part = (vector >> (index * self.block_size)) & mask
            syndrome ^= cyclic_mul(block, part, self.block_size)

        return syndrome

    def serialize(self) -> bytes:
        size = (self.block_size + 7) // 8
        return b"".join(
            block.to_bytes(size, "little")
            for block in self.blocks
        )

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        block_size: int,
        block_count: int,
    ) -> "QCParityCheck":
        size = (block_size + 7) // 8
        if len(data) != size * block_count:
            raise ValueError("Invalid QC matrix size.")

        mask = (1 << block_size) - 1
        blocks = []

        for index in range(block_count):
            start = index * size
            blocks.append(
                int.from_bytes(data[start:start + size], "little") & mask
            )

        return cls(tuple(blocks), block_size)


def supports_from_blocks(
    blocks: tuple[int, ...],
    block_size: int,
) -> tuple[tuple[int, ...], ...]:
    supports = []

    for block in blocks:
        support = []
        value = block

        while value:
            lsb = value & -value
            support.append(lsb.bit_length() - 1)
            value ^= lsb

        supports.append(tuple(support))

    return tuple(supports)
