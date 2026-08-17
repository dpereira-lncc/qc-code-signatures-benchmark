from __future__ import annotations

def mask_for_n(n: int) -> int:
    return (1 << n) - 1

def rotate_left(value: int, shift: int, n: int) -> int:
    shift %= n
    value &= mask_for_n(n)
    if shift == 0:
        return value
    return ((value << shift) & mask_for_n(n)) | (value >> (n - shift))

def reduce_mod_xn_minus_one(poly: int, n: int) -> int:
    mask = mask_for_n(n)
    while poly.bit_length() > n:
        poly = (poly & mask) ^ (poly >> n)
    return poly & mask

def cyclic_mul(a: int, b: int, n: int) -> int:
    a &= mask_for_n(n); b &= mask_for_n(n)
    if a.bit_count() > b.bit_count():
        a, b = b, a
    result = 0
    while a:
        lsb = a & -a
        result ^= rotate_left(b, lsb.bit_length() - 1, n)
        a ^= lsb
    return result & mask_for_n(n)

def poly_inverse_mod_xn_minus_one(a: int, n: int) -> int:
    a &= mask_for_n(n)
    if a == 0:
        raise ValueError('Zero is not invertible.')
    modulus = (1 << n) | 1
    u, v = a, modulus
    g1, g2 = 1, 0
    while u != 1:
        if u == 0:
            raise ValueError('Polynomial is not invertible.')
        j = u.bit_length() - v.bit_length()
        if j < 0:
            u, v = v, u
            g1, g2 = g2, g1
            j = -j
        u ^= v << j
        g1 ^= g2 << j
    return reduce_mod_xn_minus_one(g1, n)

def is_invertible(a: int, n: int) -> bool:
    try:
        poly_inverse_mod_xn_minus_one(a, n)
        return True
    except ValueError:
        return False

def int_to_bits(value: int, n: int):
    import numpy as np
    data = value.to_bytes((n + 7)//8, 'little')
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little')[:n].astype(np.uint8, copy=False)

def bits_to_int(bits) -> int:
    import numpy as np
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder='little')
    return int.from_bytes(packed.tobytes(), 'little')
