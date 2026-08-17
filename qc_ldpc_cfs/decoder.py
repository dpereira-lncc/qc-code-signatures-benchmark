from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np

from .qc import QCParityCheck
from .ring import bits_to_int, int_to_bits

try:
    from numba import njit
    NUMBA_AVAILABLE: Final = True
except ImportError:  # pragma: no cover - fallback for environments without Numba
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
    """
    Compiled normalized min-sum kernel.

    The graph is provided in CSR format:
      check_ptr/check_edges       -> arestas incidentes em cada check;
      variable_ptr/variable_edges -> edges incident to each variable.
    """
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
        # Check -> variable.
        for check in range(check_count):
            start = check_ptr[check]
            end = check_ptr[check + 1]

            total_sign = -1.0 if syndrome_bits[check] else 1.0
            min1 = np.float32(1.0e30)
            min2 = np.float32(1.0e30)
            min_edge = -1

            for position in range(start, end):
                edge = check_edges[position]
                value = v2c[edge]

                if value < 0.0:
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

            for position in range(start, end):
                edge = check_edges[position]
                own_sign = -1.0 if v2c[edge] < 0.0 else 1.0
                magnitude = min2 if edge == min_edge else min1
                c2v[edge] = normalization * total_sign * own_sign * magnitude

        # Posterior and decision.
        for variable in range(variable_count):
            total = prior
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]

            for position in range(start, end):
                total += c2v[variable_edges[position]]

            posterior[variable] = total
            hard[variable] = 1 if total < 0.0 else 0

        # Check H * hard^T = syndrome without leaving compiled code.
        valid = True
        for check in range(check_count):
            parity = 0
            start = check_ptr[check]
            end = check_ptr[check + 1]

            for position in range(start, end):
                edge = check_edges[position]
                parity ^= int(hard[edge_vars[edge]])

            if parity != int(syndrome_bits[check]):
                valid = False
                break

        if valid:
            return True, iteration, hard

        # Variable -> check.
        for variable in range(variable_count):
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            total = posterior[variable]

            for position in range(start, end):
                edge = variable_edges[position]
                v2c[edge] = total - c2v[edge]

    return False, max_iterations, hard


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    error: int
    iterations: int
    weight: int


@dataclass
class MinSumSyndromeDecoder:
    parity_check: QCParityCheck
    supports: tuple[tuple[int, ...], ...]
    max_iterations: int
    crossover_probability: float
    normalization: float

    def __post_init__(self) -> None:
        if not 0.0 < self.crossover_probability < 0.5:
            raise ValueError("The probability must lie in (0, 0.5).")
        if not 0.0 < self.normalization <= 1.0:
            raise ValueError("The normalization factor must lie in (0, 1].")

        self.prior = np.float32(
            log(
                (1.0 - self.crossover_probability)
                / self.crossover_probability
            )
        )
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Build the graph once per key and store it in CSR format.
        """
        block_size = self.parity_check.block_size
        variable_count = self.parity_check.n
        check_count = self.parity_check.r

        check_lists: list[list[int]] = [[] for _ in range(check_count)]
        variable_lists: list[list[int]] = [[] for _ in range(variable_count)]
        edge_vars: list[int] = []

        for block_index, support in enumerate(self.supports):
            block_offset = block_index * block_size

            for local_variable in range(block_size):
                variable = block_offset + local_variable

                for shift in support:
                    check = (local_variable + shift) % block_size
                    edge = len(edge_vars)

                    edge_vars.append(variable)
                    check_lists[check].append(edge)
                    variable_lists[variable].append(edge)

        self.edge_vars = np.asarray(edge_vars, dtype=np.int32)
        self.check_ptr, self.check_edges = self._to_csr(check_lists)
        self.variable_ptr, self.variable_edges = self._to_csr(variable_lists)

    @staticmethod
    def _to_csr(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        ptr = np.empty(len(rows) + 1, dtype=np.int32)
        ptr[0] = 0

        total = 0
        for index, row in enumerate(rows):
            total += len(row)
            ptr[index + 1] = total

        values = np.empty(total, dtype=np.int32)
        offset = 0

        for row in rows:
            length = len(row)
            if length:
                values[offset:offset + length] = row
            offset += length

        return ptr, values

    @property
    def edge_count(self) -> int:
        return int(self.edge_vars.size)

    @property
    def numba_enabled(self) -> bool:
        return NUMBA_AVAILABLE

    def warm_up(self) -> None:
        """
        Compile the Numba kernel outside signature measurements.

        The call uses the zero syndrome and a single iteration. Its result is
        discarded; the purpose is to materialize the JIT specialization.
        """
        zero = np.zeros(self.parity_check.r, dtype=np.uint8)
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
        syndrome_bits = int_to_bits(
            syndrome,
            self.parity_check.r,
        ).astype(np.uint8, copy=False)

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
            return DecodeResult(
                success=False,
                error=0,
                iterations=int(iterations),
                weight=0,
            )

        error = bits_to_int(hard)

        return DecodeResult(
            success=True,
            error=error,
            iterations=int(iterations),
            weight=int(np.sum(hard)),
        )
