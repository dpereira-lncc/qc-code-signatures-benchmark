from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np

from .ring import bits_to_int, int_to_bits

try:
    from numba import njit
    NUMBA_AVAILABLE: Final = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator


@njit(cache=True)
def _min_sum_kernel(
    syndrome_bits: np.ndarray,
    check_ptr: np.ndarray,
    check_edges: np.ndarray,
    variable_ptr: np.ndarray,
    variable_edges: np.ndarray,
    edge_vars: np.ndarray,
    prior: float,
    normalization: float,
    max_iterations: int,
) -> tuple[bool, int, np.ndarray]:
    edge_count = edge_vars.size
    variable_count = variable_ptr.size - 1
    check_count = check_ptr.size - 1

    v2c = np.empty(edge_count, dtype=np.float32)
    c2v = np.zeros(edge_count, dtype=np.float32)
    posterior = np.empty(variable_count, dtype=np.float32)
    hard = np.zeros(variable_count, dtype=np.uint8)

    for edge in range(edge_count):
        v2c[edge] = prior

    for iteration in range(1, max_iterations + 1):
        for check in range(check_count):
            start = check_ptr[check]
            end = check_ptr[check + 1]

            total_sign = -1.0 if syndrome_bits[check] else 1.0
            min1 = np.float32(1.0e30)
            min2 = np.float32(1.0e30)
            min_edge = -1

            for pos in range(start, end):
                edge = check_edges[pos]
                value = v2c[edge]

                if value < 0:
                    total_sign = -total_sign
                    magnitude = -value
                else:
                    magnitude = value

                if magnitude < min1:
                    min2 = min1
                    min1 = magnitude
                    min_edge = edge
                elif magnitude < min2:
                    min2 = magnitude

            if end - start == 1:
                min2 = min1

            for pos in range(start, end):
                edge = check_edges[pos]
                own_sign = -1.0 if v2c[edge] < 0 else 1.0
                magnitude = min2 if edge == min_edge else min1
                c2v[edge] = normalization * total_sign * own_sign * magnitude

        for variable in range(variable_count):
            total = prior
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            for pos in range(start, end):
                total += c2v[variable_edges[pos]]
            posterior[variable] = total
            hard[variable] = 1 if total < 0 else 0

        valid = True
        for check in range(check_count):
            parity = 0
            start = check_ptr[check]
            end = check_ptr[check + 1]
            for pos in range(start, end):
                edge = check_edges[pos]
                parity ^= int(hard[edge_vars[edge]])
            if parity != int(syndrome_bits[check]):
                valid = False
                break

        if valid:
            return True, iteration, hard

        for variable in range(variable_count):
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            total = posterior[variable]
            for pos in range(start, end):
                edge = variable_edges[pos]
                v2c[edge] = total - c2v[edge]

    return False, max_iterations, hard


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    error: int
    iterations: int
    weight: int


@dataclass
class SparseMinSumDecoder:
    rows: tuple[int, ...]
    n: int
    max_iterations: int
    crossover_probability: float
    normalization: float

    def __post_init__(self) -> None:
        self.prior = np.float32(
            log((1.0 - self.crossover_probability) / self.crossover_probability)
        )
        self._build_graph()

    def _build_graph(self) -> None:
        check_lists = []
        variable_lists = [[] for _ in range(self.n)]
        edge_vars = []

        for check, row in enumerate(self.rows):
            edges = []
            value = row
            while value:
                lsb = value & -value
                variable = lsb.bit_length() - 1
                edge = len(edge_vars)
                edge_vars.append(variable)
                edges.append(edge)
                variable_lists[variable].append(edge)
                value ^= lsb
            check_lists.append(edges)

        self.edge_vars = np.asarray(edge_vars, dtype=np.int32)
        self.check_ptr, self.check_edges = self._to_csr(check_lists)
        self.variable_ptr, self.variable_edges = self._to_csr(variable_lists)

    @staticmethod
    def _to_csr(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        ptr = np.empty(len(rows) + 1, dtype=np.int32)
        ptr[0] = 0
        total = 0
        for i, row in enumerate(rows):
            total += len(row)
            ptr[i + 1] = total

        values = np.empty(total, dtype=np.int32)
        offset = 0
        for row in rows:
            length = len(row)
            if length:
                values[offset:offset + length] = row
            offset += length
        return ptr, values

    def warm_up(self) -> None:
        zero = np.zeros(len(self.rows), dtype=np.uint8)
        _min_sum_kernel(
            zero,
            self.check_ptr,
            self.check_edges,
            self.variable_ptr,
            self.variable_edges,
            self.edge_vars,
            self.prior,
            np.float32(self.normalization),
            1,
        )

    def decode(self, syndrome: int) -> DecodeResult:
        syndrome_bits = int_to_bits(syndrome, len(self.rows))
        success, iterations, hard = _min_sum_kernel(
            syndrome_bits,
            self.check_ptr,
            self.check_edges,
            self.variable_ptr,
            self.variable_edges,
            self.edge_vars,
            self.prior,
            np.float32(self.normalization),
            self.max_iterations,
        )
        if not success:
            return DecodeResult(False, 0, int(iterations), 0)

        error = bits_to_int(hard)
        return DecodeResult(
            True,
            error,
            int(iterations),
            int(np.sum(hard)),
        )
