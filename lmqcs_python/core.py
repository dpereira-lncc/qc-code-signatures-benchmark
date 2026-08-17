from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import decode_ternary, encode_ternary, pack_fields, pack_values, unpack_fields, unpack_values
from .hashing import hash_to_challenge
from .parameters import LMQCS128, LMQCSParameters
from .ring import cyclic_mul, norm_inf, norm_l1, poly_inverse, sparse_cyclic_mul, symmetric
from common.rng import ShakeRNG
from .sampling import sample_box, sample_sparse_ternary, sample_uniform_ring


@dataclass(frozen=True)
class PublicKey:
    h: np.ndarray
    b: np.ndarray
    parameter_name: str

    def to_bytes(self, par: LMQCSParameters = LMQCS128) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Incompatible parameters.")
        return pack_values(np.concatenate((self.h % par.q, self.b % par.q)), par.q_bits)

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS128) -> "PublicKey":
        if len(data) != par.public_key_bytes:
            raise ValueError("Invalid public-key size.")
        values = unpack_values(data, 2 * par.n, par.q_bits)
        if np.any(values >= par.q):
            raise ValueError("Coefficient outside F_q.")
        return cls(values[:par.n], values[par.n:], par.name)


@dataclass(frozen=True)
class SecretKey:
    e1: np.ndarray
    e2: np.ndarray
    parameter_name: str

    def to_bytes(self, par: LMQCSParameters = LMQCS128) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Incompatible parameters.")
        return pack_values(encode_ternary(np.concatenate((self.e1, self.e2))), 2)

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS128) -> "SecretKey":
        if len(data) != par.secret_key_bytes:
            raise ValueError("Invalid private-key size.")
        values = decode_ternary(unpack_values(data, 2 * par.n, 2))
        return cls(values[:par.n], values[par.n:], par.name)


@dataclass(frozen=True)
class Signature:
    c: np.ndarray
    b_bar: np.ndarray
    u: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    parameter_name: str
    attempts: int = 1
    elapsed_seconds: float = 0.0

    def to_bytes(self, par: LMQCSParameters = LMQCS128) -> bytes:
        if self.parameter_name != par.name:
            raise ValueError("Incompatible parameters.")
        out = pack_fields([
            (encode_ternary(self.c), 2),
            (self.b_bar % par.q, par.q_bits),
            (self.u % par.q, par.q_bits),
            ((self.s1 + par.gamma).tolist(), par.s_bits),
            ((self.s2 + par.gamma).tolist(), par.s_bits),
        ])
        if len(out) != par.signature_bytes:
            raise RuntimeError(
                f"Unexpected serialized size: {len(out)} != {par.signature_bytes}."
            )
        return out

    @classmethod
    def from_bytes(cls, data: bytes, par: LMQCSParameters = LMQCS128) -> "Signature":
        if len(data) != par.signature_bytes:
            raise ValueError("Invalid signature size.")
        c_enc, b_bar, u, s1_enc, s2_enc = unpack_fields(data, [
            (par.n, 2),
            (par.n, par.q_bits),
            (par.n, par.q_bits),
            (par.n, par.s_bits),
            (par.n, par.s_bits),
        ])
        c = decode_ternary(c_enc)
        s1 = s1_enc - par.gamma
        s2 = s2_enc - par.gamma
        if np.any(b_bar >= par.q) or np.any(u >= par.q):
            raise ValueError("Coefficient outside F_q.")
        return cls(c, b_bar, u, s1, s2, par.name)


@dataclass(frozen=True)
class KeyPair:
    public_key: PublicKey
    secret_key: SecretKey
    h_inverse: np.ndarray
    elapsed_seconds: float = 0.0
    attempts: int = 1


def keygen(
    par: LMQCSParameters = LMQCS128,
    rng: ShakeRNG | None = None,
) -> KeyPair:
    rng = rng or ShakeRNG.from_system()
    attempts = 0
    while True:
        attempts += 1
        h = sample_uniform_ring(par.n, par.q, rng)
        try:
            h_inverse = poly_inverse(h, par.q)
            break
        except ValueError:
            continue
    e1 = sample_sparse_ternary(par.n, par.omega_e, rng)
    e2 = sample_sparse_ternary(par.n, par.omega_e, rng)
    b = (sparse_cyclic_mul(e1, h, par.q) + sparse_cyclic_mul(e2, h_inverse, par.q)) % par.q
    return KeyPair(
        public_key=PublicKey(h, b, par.name),
        secret_key=SecretKey(e1, e2, par.name),
        h_inverse=h_inverse,
        attempts=attempts,
    )


def _public_precomputation(public_key: PublicKey, par: LMQCSParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_inv = poly_inverse(public_key.h, par.q)
    h2 = cyclic_mul(public_key.h, public_key.h, par.q)
    h_inv2 = cyclic_mul(h_inv, h_inv, par.q)
    return h_inv, h2, h_inv2


def sign(
    message: bytes,
    secret_key: SecretKey,
    public_key: PublicKey,
    par: LMQCSParameters = LMQCS128,
    rng: ShakeRNG | None = None,
    max_attempts: int = 10_000,
) -> Signature:
    if secret_key.parameter_name != par.name or public_key.parameter_name != par.name:
        raise ValueError("Incompatible parameters.")
    rng = rng or ShakeRNG.from_system()
    h_inv, h2, h_inv2 = _public_precomputation(public_key, par)

    for attempt in range(1, max_attempts + 1):
        e_bar1 = sample_box(par.n, par.ell, rng)
        e_bar2 = sample_box(par.n, par.ell, rng)
        u1 = sample_box(par.n, par.ell, rng)
        u2 = sample_box(par.n, par.ell, rng)

        b_bar = (cyclic_mul(e_bar1, public_key.h, par.q) + cyclic_mul(e_bar2, h_inv, par.q)) % par.q
        if not np.any(b_bar):
            continue
        u = (cyclic_mul(u1, h2, par.q) + cyclic_mul(u2, h_inv2, par.q)) % par.q
        t = (sparse_cyclic_mul(secret_key.e1, e_bar2, par.q) + sparse_cyclic_mul(secret_key.e2, e_bar1, par.q)) % par.q
        c, c_inv = hash_to_challenge(message, t, u, b_bar, public_key.h, public_key.b, par)

        s1 = symmetric(sparse_cyclic_mul(secret_key.e1, e_bar1, par.q) + sparse_cyclic_mul(c, u1, par.q), par.q)
        s2 = symmetric(sparse_cyclic_mul(secret_key.e2, e_bar2, par.q) + sparse_cyclic_mul(c, u2, par.q), par.q)

        w1 = symmetric(cyclic_mul(c_inv, s1, par.q), par.q)
        w2 = symmetric(cyclic_mul(c_inv, s2, par.q), par.q)
        t_sym = symmetric(t, par.q)

        if (
            norm_inf(s1) <= par.gamma
            and norm_inf(s2) <= par.gamma
            and norm_l1(s1) <= par.rho
            and norm_l1(s2) <= par.rho
            and norm_inf(w1) > par.gamma
            and norm_inf(w2) > par.gamma
            and norm_inf(t_sym) <= par.gamma
        ):
            return Signature(c, b_bar, u, s1, s2, par.name, attempts=attempt)

    raise RuntimeError(f"Could not generate a signature in {max_attempts} attempts.")


def verify(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: LMQCSParameters = LMQCS128,
) -> bool:
    try:
        if signature.parameter_name != par.name or public_key.parameter_name != par.name:
            return False
        if not np.any(signature.b_bar):
            return False
        if np.count_nonzero(signature.c) != par.omega_c:
            return False
        if np.any((signature.c < -1) | (signature.c > 1)):
            return False

        h_inv, h2, h_inv2 = _public_precomputation(public_key, par)
        c_inv = poly_inverse(signature.c, par.q)

        bb = cyclic_mul(signature.b_bar, public_key.b, par.q)
        cu = sparse_cyclic_mul(signature.c, signature.u, par.q)
        s_terms = (cyclic_mul(signature.s1, h2, par.q) + cyclic_mul(signature.s2, h_inv2, par.q)) % par.q
        t = (bb + cu - s_terms) % par.q

        expected_c, _ = hash_to_challenge(
            message, t, signature.u, signature.b_bar, public_key.h, public_key.b, par
        )
        if not np.array_equal(expected_c, signature.c):
            return False

        if norm_inf(signature.s1) > par.gamma or norm_inf(signature.s2) > par.gamma:
            return False
        if norm_l1(signature.s1) > par.rho or norm_l1(signature.s2) > par.rho:
            return False

        w1 = symmetric(cyclic_mul(c_inv, signature.s1, par.q), par.q)
        w2 = symmetric(cyclic_mul(c_inv, signature.s2, par.q), par.q)
        if norm_inf(w1) <= par.gamma or norm_inf(w2) <= par.gamma:
            return False
        if norm_inf(t, par.q) > par.gamma:
            return False
        return True
    except (ValueError, RuntimeError, OverflowError):
        return False
