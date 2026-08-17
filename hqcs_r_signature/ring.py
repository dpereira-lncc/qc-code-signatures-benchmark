from __future__ import annotations

from math import ceil, log2

from .inversion import invert_compiled


def centered(value: int, q: int) -> int:
    value %= q
    return value - q if value > q // 2 else value


def to_residue(value: int, q: int) -> int:
    return value % q


def add(a: list[int], b: list[int], q: int) -> list[int]:
    return [(x + y) % q for x, y in zip(a, b)]


def sub(a: list[int], b: list[int], q: int) -> list[int]:
    return [(x - y) % q for x, y in zip(a, b)]


def scalar_mul(a: list[int], scalar: int, q: int) -> list[int]:
    return [(scalar * x) % q for x in a]


def _packing_bits(k: int, q: int) -> int:
    # Ordinary convolution contains sums of up to k products.
    return ceil(log2(k * (q - 1) * (q - 1) + 1))


def _pack_coefficients(a: list[int], bits: int) -> int:
    value = 0
    shift = 0
    for coefficient in a:
        value |= int(coefficient) << shift
        shift += bits
    return value


def cyclic_mul(a: list[int], b: list[int], q: int) -> list[int]:
    """
    Produto em F_q[x]/(x^k-1) via Kronecker substitution.

    The base is chosen large enough to prevent carry propagation between
    coefficients of the ordinary convolution.
    """
    k = len(a)
    if len(b) != k:
        raise ValueError("The polynomials must have the same length.")

    bits = _packing_bits(k, q)
    mask = (1 << bits) - 1

    packed_a = _pack_coefficients([x % q for x in a], bits)
    packed_b = _pack_coefficients([x % q for x in b], bits)
    product = packed_a * packed_b

    convolution = [0] * (2 * k - 1)
    for index in range(2 * k - 1):
        convolution[index] = (product >> (index * bits)) & mask

    result = [0] * k
    for index, value in enumerate(convolution):
        result[index % k] = (result[index % k] + value) % q
    return result


def sparse_cyclic_mul(
    sparse_positions: tuple[int, ...],
    sparse_signs: tuple[int, ...],
    dense: list[int],
    q: int,
) -> list[int]:
    k = len(dense)
    result = [0] * k

    for position, sign in zip(sparse_positions, sparse_signs):
        for index, value in enumerate(dense):
            result[(index + position) % k] += sign * value

    return [value % q for value in result]


def invert(a: list[int], q: int) -> list[int]:
    """Inverte a em F_q[x]/(x^k-1) com Euclides estendido compilado por Numba."""
    return invert_compiled(a, q)


def is_invertible(a: list[int], q: int) -> bool:
    try:
        invert(a, q)
        return True
    except Exception:
        return False


def quotient_vector(a: list[int], p: int) -> list[int]:
    return [coefficient // p for coefficient in a]
