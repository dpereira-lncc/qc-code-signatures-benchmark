from __future__ import annotations

from dataclasses import dataclass

from .challenge import Challenge, hash_to_challenge
from .encoding import (
    BitReader,
    BitWriter,
    decode_restricted,
    encode_restricted,
    pack_values,
    unpack_values,
)
from .parameters import Parameters, HQCS_R_1
from .ring import add, cyclic_mul, quotient_vector, sparse_cyclic_mul
from common.rng import ShakeRNG
from .sampling import (
    sample_restricted_vector,
    sample_seeded_invertible_vector,
    sample_uniform_vector,
    vector_from_seed,
)


@dataclass(frozen=True)
class PublicKey:
    h_seed: bytes
    b: list[int]

    def h(self, par: Parameters) -> list[int]:
        return vector_from_seed(self.h_seed, par, b"HQCS-R-H-v1")

    def to_bytes(self, par: Parameters) -> bytes:
        raw = self.h_seed + pack_values(self.b, par.q_bits)
        if len(raw) > par.public_key_bytes:
            raise RuntimeError("The public key exceeds its declared size.")
        return raw.ljust(par.public_key_bytes, b"\x00")

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "PublicKey":
        if len(data) != par.public_key_bytes:
            raise ValueError("Invalid public-key size.")

        seed = data[:par.seed_bytes]
        required_b_bytes = (par.k * par.q_bits + 7) // 8
        b_bytes = data[par.seed_bytes:par.seed_bytes + required_b_bytes]
        b = unpack_values(b_bytes, par.k, par.q_bits)

        if any(value >= par.q for value in b):
            raise ValueError("Invalid public coefficient.")
        return cls(seed, b)


@dataclass(frozen=True)
class SecretKey:
    e1: list[int]

    def to_bytes(self, par: Parameters) -> bytes:
        return encode_restricted(self.e1, par)[:par.secret_key_bytes]

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "SecretKey":
        if len(data) != par.secret_key_bytes:
            raise ValueError("Invalid secret-key size.")
        return cls(decode_restricted(data, par))


@dataclass(frozen=True)
class KeyPair:
    public_key: PublicKey
    secret_key: SecretKey
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class Signature:
    challenge: Challenge
    s: list[int]
    v_seed: bytes
    attempts: int
    elapsed_seconds: float = 0.0

    def to_bytes(self, par: Parameters) -> bytes:
        writer = BitWriter()

        for value in self.s:
            writer.write(value, par.q_bits)

        for byte in self.v_seed:
            writer.write(byte, 8)

        challenge_map = {
            position: sign
            for position, sign in zip(
                self.challenge.positions,
                self.challenge.signs,
            )
        }
        for index in range(par.k):
            sign = challenge_map.get(index, 0)
            symbol = 0 if sign == 0 else (1 if sign == 1 else 2)
            writer.write(symbol, 2)

        encoded = writer.finish()
        if len(encoded) != par.signature_bytes:
            raise RuntimeError("Inconsistent internal signature size.")
        return encoded

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "Signature":
        if len(data) != par.signature_bytes:
            raise ValueError("Invalid signature size.")

        reader = BitReader(data)

        s = [reader.read(par.q_bits) for _ in range(par.k)]
        if any(value >= par.q for value in s):
            raise ValueError("Invalid signature coefficient.")

        v_seed = bytes(reader.read(8) for _ in range(par.seed_bytes))

        positions = []
        signs = []
        for index in range(par.k):
            symbol = reader.read(2)
            if symbol == 0:
                continue
            if symbol == 1:
                sign = 1
            elif symbol == 2:
                sign = -1
            else:
                raise ValueError("Invalid challenge symbol.")
            positions.append(index)
            signs.append(sign)

        if len(positions) != par.omega_c:
            raise ValueError("Invalid challenge weight.")

        challenge = Challenge(tuple(positions), tuple(signs))
        return cls(challenge, s, v_seed, 0, 0.0)


def keygen(
    par: Parameters = HQCS_R_1,
    rng: ShakeRNG | None = None,
) -> KeyPair:
    rng = rng or ShakeRNG.from_system()

    h_seed = rng.random_bytes(par.seed_bytes)
    h = vector_from_seed(h_seed, par, b"HQCS-R-H-v1")

    e1 = sample_restricted_vector(par, rng)
    e2 = sample_restricted_vector(par, rng)

    b = add(
        cyclic_mul([value % par.q for value in e1], h, par.q),
        [value % par.q for value in e2],
        par.q,
    )

    return KeyPair(
        public_key=PublicKey(h_seed, b),
        secret_key=SecretKey(e1),
    )


def sign(
    message: bytes,
    secret_key: SecretKey,
    public_key: PublicKey,
    par: Parameters = HQCS_R_1,
    rng: ShakeRNG | None = None,
) -> Signature:
    rng = rng or ShakeRNG.from_system()
    h = public_key.h(par)
    pk_bytes = public_key.to_bytes(par)

    for attempt in range(1, par.max_sign_attempts + 1):
        u = sample_uniform_vector(par, rng)
        v_seed, v, v_inverse = sample_seeded_invertible_vector(
            par,
            b"HQCS-R-V-v1",
            rng,
        )

        vu = cyclic_mul(v, u, par.q)
        vuh = cyclic_mul(vu, h, par.q)

        quotient_vu = quotient_vector(vu, par.p)
        quotient_vuh = quotient_vector(vuh, par.p)

        challenge = hash_to_challenge(
            message,
            v_seed,
            quotient_vu,
            quotient_vuh,
            pk_bytes,
            par,
        )

        ce1 = sparse_cyclic_mul(
            challenge.positions,
            challenge.signs,
            [value % par.q for value in secret_key.e1],
            par.q,
        )
        correction = cyclic_mul(v_inverse, ce1, par.q)
        s = add(u, correction, par.q)

        vs = cyclic_mul(v, s, par.q)
        cb = sparse_cyclic_mul(
            challenge.positions,
            challenge.signs,
            public_key.b,
            par.q,
        )
        t = [
            (x - y) % par.q
            for x, y in zip(cyclic_mul(vs, h, par.q), cb)
        ]

        if quotient_vector(vs, par.p) != quotient_vu:
            continue
        if quotient_vector(t, par.p) != quotient_vuh:
            continue

        return Signature(
            challenge=challenge,
            s=s,
            v_seed=v_seed,
            attempts=attempt,
        )

    raise RuntimeError(
        f"Failed after {par.max_sign_attempts} signing attempts."
    )


def verify(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: Parameters = HQCS_R_1,
) -> bool:
    h = public_key.h(par)
    v = vector_from_seed(signature.v_seed, par, b"HQCS-R-V-v1")

    vs = cyclic_mul(v, signature.s, par.q)
    cb = sparse_cyclic_mul(
        signature.challenge.positions,
        signature.challenge.signs,
        public_key.b,
        par.q,
    )
    t = [
        (x - y) % par.q
        for x, y in zip(cyclic_mul(vs, h, par.q), cb)
    ]

    expected = hash_to_challenge(
        message,
        signature.v_seed,
        quotient_vector(vs, par.p),
        quotient_vector(t, par.p),
        public_key.to_bytes(par),
        par,
    )
    return expected == signature.challenge
