from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Parameters:
    name: str
    security_label: str
    block_size: int
    block_count: int
    secret_block_weight: int
    q_block_weight: int
    max_bp_iterations: int
    channel_error_probability: float
    min_sum_normalization: float
    nonce_bytes: int = 32
    max_sign_attempts: int = 10000

    @property
    def n(self) -> int:
        return self.block_size * self.block_count

    @property
    def r(self) -> int:
        return self.block_size

    @property
    def k(self) -> int:
        return self.n - self.r

    @property
    def stored_public_blocks(self) -> int:
        return self.block_count

    @property
    def public_key_bytes(self) -> int:
        return self.stored_public_blocks * ((self.block_size + 7) // 8)

    @property
    def signature_bytes(self) -> int:
        return (self.n + 7) // 8 + 4

    @property
    def code_rate(self) -> float:
        return self.k / self.n

# ============================================================
# Toy parameters
# Used only to validate the implementation workflow.
# These parameters do not provide cryptographic security.
# ============================================================

DEMO = Parameters(
    name="QC-LDPC-CFS-DEMO",
    security_label="demo",
    block_size=31,
    block_count=4,
    secret_block_weight=3,
    q_block_weight=3,
    max_bp_iterations=100,
    channel_error_probability=0.12,
    min_sum_normalization=0.80,
    max_sign_attempts=10000,
)



# ============================================================
# Parameters reported in the original QC-LDPC article
#
# n = 4 * 4096 = 16384
#
# The BP implementation parameters below are NOT specified
# in the original article. They are implementation choices
# adopted in this work.
# ============================================================

ORIGINAL_ARTICLE = Parameters(
    name="QC-LDPC-CFS-ORIGINAL",
    security_label="original",
    block_size=4096,
    block_count=4,
    secret_block_weight=13,
    q_block_weight=3,

    # Implementation-specific BP parameters
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,

    max_sign_attempts=10000,
)


# ============================================================
# Parameter sets estimated in our work
#
# n0 = 4
# R = 3/4
# w = n * 0.005
#
# The same BP configuration is used for all parameter sets
# to maintain a consistent experimental methodology.
# ============================================================

ESTIMATED_128 = Parameters(
    name="QC-LDPC-CFS-ESTIMATED-128",
    security_label="128-bit",
    block_size=3000,          # n = 12000
    block_count=4,
    secret_block_weight=13,
    q_block_weight=3,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_sign_attempts=10000,
)


ESTIMATED_192 = Parameters(
    name="QC-LDPC-CFS-ESTIMATED-192",
    security_label="192-bit",
    block_size=4690,          # n = 18760
    block_count=4,
    secret_block_weight=13,
    q_block_weight=3,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_sign_attempts=10000,
)


ESTIMATED_256 = Parameters(
    name="QC-LDPC-CFS-ESTIMATED-256",
    security_label="256-bit",
    block_size=6400,          # n = 25600
    block_count=4,
    secret_block_weight=13,
    q_block_weight=3,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_sign_attempts=10000,
)


PARAMETER_SETS = {
    "demo": DEMO,
    "original": ORIGINAL_ARTICLE,
    "estimated_128": ESTIMATED_128,
    "estimated_192": ESTIMATED_192,
    "estimated_256": ESTIMATED_256,
}
