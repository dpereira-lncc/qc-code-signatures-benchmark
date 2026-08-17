from __future__ import annotations

from dataclasses import dataclass, field

from common.rng import ShakeRNG
from common.errors import SigningFailure

from .construction import PuncturedConstruction, build_construction
from .decoder import SparseMinSumDecoder
from .hashing import hash_to_syndrome
from .matrix import (
    DenseBinaryParityCheck,
    apply_inverse_row_operations_to_vector,
    permute_vector_secret_to_public,
)
from .parameters import Parameters, DEMO


@dataclass(frozen=True)
class PublicKey:
    parity_check: DenseBinaryParityCheck

    def to_bytes(self) -> bytes:
        return self.parity_check.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "PublicKey":
        return cls(DenseBinaryParityCheck.from_bytes(data, par.r, par.n))


@dataclass(frozen=True)
class SecretKey:
    construction: PuncturedConstruction
    decoder: SparseMinSumDecoder = field(repr=False, compare=False)


@dataclass(frozen=True)
class KeyPair:
    public_key: PublicKey
    secret_key: SecretKey
    elapsed_seconds: float = 0.0
    graph_build_seconds: float = 0.0
    jit_warmup_seconds: float = 0.0


@dataclass(frozen=True)
class Signature:
    error: int
    counter: int
    attempts: int
    bp_iterations: int
    elapsed_seconds: float
    weight: int

    def to_bytes(self, par: Parameters) -> bytes:
        size = (par.n + 7) // 8
        return self.error.to_bytes(size, "little") + self.counter.to_bytes(4, "little")

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "Signature":
        size = (par.n + 7) // 8
        if len(data) != size + 4:
            raise ValueError("Invalid signature size.")

        error = int.from_bytes(data[:size], "little")
        if error >> par.n:
            raise ValueError("Bits outside the expected dimension.")

        return cls(
            error=error,
            counter=int.from_bytes(data[size:], "little"),
            attempts=0,
            bp_iterations=0,
            elapsed_seconds=0.0,
            weight=error.bit_count(),
        )


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    elapsed_seconds: float
    reason: str
    weight: int


def keygen(
    par: Parameters = DEMO,
    rng: ShakeRNG | None = None,
    *,
    warm_up_decoder: bool = False,
) -> KeyPair:
    rng = rng or ShakeRNG.from_system()
    construction = build_construction(par, rng)

    decoder = SparseMinSumDecoder(
        rows=construction.punctured_rows,
        n=par.punctured_n,
        max_iterations=par.max_bp_iterations,
        crossover_probability=par.channel_error_probability,
        normalization=par.min_sum_normalization,
    )
    if warm_up_decoder:
        decoder.warm_up()

    return KeyPair(
        public_key=PublicKey(construction.public_matrix),
        secret_key=SecretKey(construction, decoder),
    )


def _complete_inserted_bits(
    punctured_error: int,
    transformed_syndrome: int,
    construction: PuncturedConstruction,
    par: Parameters,
) -> int:
    p = par.puncture_count
    syndrome_tail = transformed_syndrome >> par.punctured_r

    inserted = 0
    for index, row in enumerate(construction.random_rows):
        random_part = row & ((1 << par.punctured_n) - 1)
        parity = (random_part & punctured_error).bit_count() & 1
        bit = ((syndrome_tail >> index) & 1) ^ parity
        inserted |= bit << index

    return punctured_error | (inserted << par.punctured_n)


def sign(
    message: bytes,
    secret_key: SecretKey,
    public_key: PublicKey,
    par: Parameters = DEMO,
    rng: ShakeRNG | None = None,
    *,
    max_attempts: int | None = None,
) -> Signature:
    del rng  # Deterministic signature for a key, message, and counter.
    attempt_limit = par.max_sign_attempts if max_attempts is None else max_attempts
    if attempt_limit <= 0:
        raise ValueError("max_attempts must be positive")
    construction = secret_key.construction

    for counter in range(attempt_limit):
        syndrome = hash_to_syndrome(message, counter, par.r)

        transformed = apply_inverse_row_operations_to_vector(
            syndrome,
            construction.scrambling_operations,
        )

        syndrome_head = transformed & ((1 << par.punctured_r) - 1)
        decoded = secret_key.decoder.decode(syndrome_head)

        if not decoded.success:
            continue

        secret_error = _complete_inserted_bits(
            decoded.error,
            transformed,
            construction,
            par,
        )

        weight = secret_error.bit_count()
        if weight > par.max_error_weight:
            continue

        public_error = permute_vector_secret_to_public(
            secret_error,
            construction.permutation,
        )

        if public_key.parity_check.syndrome(public_error) != syndrome:
            continue

        return Signature(
            error=public_error,
            counter=counter,
            attempts=counter + 1,
            bp_iterations=decoded.iterations,
            elapsed_seconds=0.0,
            weight=weight,
        )

    raise SigningFailure(
        f"Failed after {attempt_limit} attempts.",
        attempts=attempt_limit,
    )


def verify_detailed(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: Parameters = DEMO,
) -> VerificationResult:
    weight = signature.error.bit_count()

    if signature.error >> par.n:
        return VerificationResult(
            False,
            0.0,
            "Vector outside the expected dimension.",
            weight,
        )

    if weight > par.max_error_weight:
        return VerificationResult(
            False,
            0.0,
            "Peso de Hamming acima do limiar.",
            weight,
        )

    expected = hash_to_syndrome(message, signature.counter, par.r)
    actual = public_key.parity_check.syndrome(signature.error)

    if actual != expected:
        return VerificationResult(
            False,
            0.0,
            "Syndrome mismatch.",
            weight,
        )

    return VerificationResult(
        True,
        0.0,
        "Valid signature.",
        weight,
    )


def verify(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: Parameters = DEMO,
) -> bool:
    return verify_detailed(message, signature, public_key, par).accepted
