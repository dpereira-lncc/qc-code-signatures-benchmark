from __future__ import annotations

from dataclasses import dataclass, field

from common.rng import ShakeRNG
from common.errors import SigningFailure

from .decoder import MinSumSyndromeDecoder
from .hashing import hash_to_syndrome
from .parameters import Parameters, DEMO
from .qc import QCParityCheck
from .ring import cyclic_mul, poly_inverse_mod_xn_minus_one
from .sampling import (
    sample_dense_invertible_poly,
    sample_invertible_sparse_poly,
    sample_sparse_poly,
)


@dataclass(frozen=True)
class PublicKey:
    parity_check: QCParityCheck

    def to_bytes(self) -> bytes:
        return self.parity_check.serialize()

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "PublicKey":
        return cls(
            QCParityCheck.deserialize(
                data,
                block_size=par.block_size,
                block_count=par.block_count,
            )
        )


@dataclass(frozen=True)
class SecretKey:
    secret_parity_check: QCParityCheck
    s_inverse: int
    q_inverses: tuple[int, ...]
    secret_supports: tuple[tuple[int, ...], ...]
    decoder: MinSumSyndromeDecoder = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class KeyPair:
    public_key: PublicKey
    secret_key: SecretKey
    elapsed_seconds: float = 0.0
    graph_build_seconds: float = 0.0
    jit_warmup_seconds: float = 0.0


@dataclass(frozen=True)
class Signature:
    z: int
    counter: int
    attempts: int
    bp_iterations: int
    elapsed_seconds: float = 0.0

    def to_bytes(self, par: Parameters) -> bytes:
        vector_size = (par.n + 7) // 8
        return (
            self.z.to_bytes(vector_size, "little")
            + self.counter.to_bytes(4, "little")
        )

    @classmethod
    def from_bytes(cls, data: bytes, par: Parameters) -> "Signature":
        vector_size = (par.n + 7) // 8

        if len(data) != vector_size + 4:
            raise ValueError("Invalid signature size.")

        z = int.from_bytes(data[:vector_size], "little")

        if z >> par.n:
            raise ValueError("Bits outside the signature dimension.")

        return cls(
            z=z,
            counter=int.from_bytes(data[vector_size:], "little"),
            attempts=0,
            bp_iterations=0,
            elapsed_seconds=0.0,
        )


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    elapsed_seconds: float
    reason: str


def keygen(
    par: Parameters = DEMO,
    rng: ShakeRNG | None = None,
    *,
    warm_up_decoder: bool = False,
) -> KeyPair:
    """
    Generate H_public = S H_secret Q and prepare the key's decoder.

    The Tanner graph is built only once. When ``warm_up_decoder=True``, the
    Numba kernel is also compiled during KeyGen so that the first signature
    does not incur the JIT cost.
    """
    rng = rng or ShakeRNG.from_system()
    block_size = par.block_size

    secret_blocks: list[int] = []
    secret_supports: list[tuple[int, ...]] = []

    for _ in range(par.block_count):
        block, support = sample_sparse_poly(
            block_size,
            par.secret_block_weight,
            rng,
        )
        secret_blocks.append(block)
        secret_supports.append(support)

    s = sample_dense_invertible_poly(block_size, rng)
    s_inverse = poly_inverse_mod_xn_minus_one(s, block_size)

    q_blocks: list[int] = []
    q_inverses: list[int] = []

    for _ in range(par.block_count):
        block, _ = sample_invertible_sparse_poly(
            block_size,
            par.q_block_weight,
            rng,
        )
        q_blocks.append(block)
        q_inverses.append(
            poly_inverse_mod_xn_minus_one(
                block,
                block_size,
            )
        )

    public_blocks = tuple(
        cyclic_mul(
            cyclic_mul(
                s,
                secret_blocks[index],
                block_size,
            ),
            q_blocks[index],
            block_size,
        )
        for index in range(par.block_count)
    )

    secret_h = QCParityCheck(
        blocks=tuple(secret_blocks),
        block_size=block_size,
    )

    decoder = MinSumSyndromeDecoder(
        parity_check=secret_h,
        supports=tuple(secret_supports),
        max_iterations=par.max_bp_iterations,
        crossover_probability=par.channel_error_probability,
        normalization=par.min_sum_normalization,
    )
    if warm_up_decoder:
        decoder.warm_up()

    return KeyPair(
        public_key=PublicKey(
            QCParityCheck(
                blocks=public_blocks,
                block_size=block_size,
            )
        ),
        secret_key=SecretKey(
            secret_parity_check=secret_h,
            s_inverse=s_inverse,
            q_inverses=tuple(q_inverses),
            secret_supports=tuple(secret_supports),
            decoder=decoder,
        ),
    )


def _apply_block_diagonal(
    vector: int,
    blocks: tuple[int, ...],
    block_size: int,
) -> int:
    mask = (1 << block_size) - 1
    result = 0

    for index, block in enumerate(blocks):
        part = (vector >> (index * block_size)) & mask
        transformed = cyclic_mul(
            block,
            part,
            block_size,
        )
        result |= transformed << (index * block_size)

    return result


def sign(
    message: bytes,
    secret_key: SecretKey,
    public_key: PublicKey,
    par: Parameters = DEMO,
    rng: ShakeRNG | None = None,
    *,
    max_attempts: int | None = None,
) -> Signature:
    """
    Sign by reusing the decoder and graph prepared during KeyGen.
    """
    del rng  # Deterministic signature for a key, message, and counter.
    attempt_limit = par.max_sign_attempts if max_attempts is None else max_attempts
    if attempt_limit <= 0:
        raise ValueError("max_attempts must be positive")
    decoder = secret_key.decoder

    for counter in range(attempt_limit):
        syndrome = hash_to_syndrome(
            message,
            counter,
            par.r,
        )

        transformed_syndrome = cyclic_mul(
            secret_key.s_inverse,
            syndrome,
            par.block_size,
        )

        decoded = decoder.decode(transformed_syndrome)

        if not decoded.success:
            continue

        z = _apply_block_diagonal(
            decoded.error,
            secret_key.q_inverses,
            par.block_size,
        )

        if public_key.parity_check.syndrome_int(z) != syndrome:
            continue

        return Signature(
            z=z,
            counter=counter,
            attempts=counter + 1,
            bp_iterations=decoded.iterations,
        )

    raise SigningFailure(
        f"Failed after {attempt_limit} signing attempts.",
        attempts=attempt_limit,
    )


def verify_detailed(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: Parameters = DEMO,
) -> VerificationResult:
    if signature.z >> par.n:
        return VerificationResult(
            accepted=False,
            elapsed_seconds=0.0,
            reason="Signature vector outside the expected dimension.",
        )

    expected = hash_to_syndrome(
        message,
        signature.counter,
        par.r,
    )
    actual = public_key.parity_check.syndrome_int(
        signature.z
    )

    if actual != expected:
        return VerificationResult(
            accepted=False,
            elapsed_seconds=0.0,
            reason="Syndrome mismatch.",
        )

    return VerificationResult(
        accepted=True,
        elapsed_seconds=0.0,
        reason="Valid signature.",
    )


def verify(
    message: bytes,
    signature: Signature,
    public_key: PublicKey,
    par: Parameters = DEMO,
) -> bool:
    return verify_detailed(
        message,
        signature,
        public_key,
        par,
    ).accepted
