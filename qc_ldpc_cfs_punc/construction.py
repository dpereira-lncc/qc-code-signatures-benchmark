from __future__ import annotations

from common.rng import ShakeRNG
from dataclasses import dataclass

from .parameters import Parameters
from .ring import rotate_left
from .sampling import sample_dense_bits, sample_permutation, sample_row_xor_operations, sample_sparse_poly
from .matrix import (
    DenseBinaryParityCheck,
    apply_row_operations_to_rows,
    permute_row_secret_to_public,
)


@dataclass(frozen=True)
class PuncturedConstruction:
    base_information_blocks: tuple[int, ...]
    retained_rows: tuple[int, ...]
    deleted_rows: tuple[int, ...]
    punctured_rows: tuple[int, ...]
    random_rows: tuple[int, ...]
    modified_rows: tuple[int, ...]
    public_matrix: DenseBinaryParityCheck
    permutation: tuple[int, ...]
    scrambling_operations: tuple[tuple[int, int], ...]


def _information_row(
    blocks: tuple[int, ...],
    row_index: int,
    block_size: int,
) -> int:
    row = 0
    for block_index, block in enumerate(blocks):
        rotated = rotate_left(block, row_index, block_size)
        row |= rotated << (block_index * block_size)
    return row


def _select_puncture_rows(
    information_rows: list[int],
    puncture_count: int,
    rng: ShakeRNG,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    The article requires selection guided by minimum-weight codewords but does
    not provide an executable algorithm. We use a reproducible substitute:
    remove lower-weight rows and break ties randomly.
    """
    candidates = list(range(len(information_rows)))
    random_tags = {
        index: int.from_bytes(rng.random_bytes(8), "little")
        for index in candidates
    }
    candidates.sort(
        key=lambda index: (
            information_rows[index].bit_count(),
            random_tags[index],
        )
    )
    deleted = tuple(sorted(candidates[:puncture_count]))
    deleted_set = set(deleted)
    retained = tuple(index for index in range(len(information_rows)) if index not in deleted_set)
    return retained, deleted


def build_construction(
    par: Parameters,
    rng: ShakeRNG | None = None,
) -> PuncturedConstruction:
    rng = rng or ShakeRNG.from_system()
    q = par.block_size
    k = par.k
    r = par.r
    p = par.puncture_count

    info_blocks = []
    for _ in range(par.information_blocks):
        block, _ = sample_sparse_poly(q, par.secret_block_weight, rng)
        info_blocks.append(block)
    info_blocks_t = tuple(info_blocks)

    information_rows = [
        _information_row(info_blocks_t, row, q)
        for row in range(r)
    ]

    retained, deleted = _select_puncture_rows(information_rows, p, rng)

    # H_D = [P_D^T | I_{r-p}]
    punctured_rows = []
    for local_row, original_row in enumerate(retained):
        row = information_rows[original_row]
        row |= 1 << (k + local_row)
        punctured_rows.append(row)

    punctured_n = par.punctured_n

    # The p random columns inserted into P_D become p random rows R, and the
    # corresponding identity occupies the final p positions.
    random_rows = []
    for inserted_index in range(p):
        row = sample_dense_bits(punctured_n, rng)
        row |= 1 << (punctured_n + inserted_index)
        random_rows.append(row)

    modified_rows = punctured_rows + random_rows

    permutation = sample_permutation(par.n, rng)
    operations = sample_row_xor_operations(r, par.scrambling_operations, rng)

    permuted_rows = [
        permute_row_secret_to_public(row, permutation)
        for row in modified_rows
    ]
    public_rows = apply_row_operations_to_rows(permuted_rows, operations)

    return PuncturedConstruction(
        base_information_blocks=info_blocks_t,
        retained_rows=retained,
        deleted_rows=deleted,
        punctured_rows=tuple(punctured_rows),
        random_rows=tuple(random_rows),
        modified_rows=tuple(modified_rows),
        public_matrix=DenseBinaryParityCheck(tuple(public_rows), par.n),
        permutation=permutation,
        scrambling_operations=operations,
    )
