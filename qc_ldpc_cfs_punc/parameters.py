from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Parameters:
    name: str
    block_size: int
    information_blocks: int
    puncture_count: int
    secret_block_weight: int
    max_bp_iterations: int
    channel_error_probability: float
    min_sum_normalization: float
    max_error_weight: int
    scrambling_operations: int
    max_sign_attempts: int = 10000

    @property
    def r(self) -> int:
        return self.block_size

    @property
    def k(self) -> int:
        return self.information_blocks * self.block_size

    @property
    def n(self) -> int:
        return self.k + self.r

    @property
    def punctured_r(self) -> int:
        return self.r - self.puncture_count

    @property
    def punctured_n(self) -> int:
        return self.n - self.puncture_count

    @property
    def public_key_bytes(self) -> int:
        # H_pub is stored as a dense r x n binary matrix.
        return self.r * ceil(self.n / 8)

    @property
    def signature_bytes(self) -> int:
        return ceil(self.n / 8) + 4


DEMO = Parameters(
    name="Punctured-QC-LDPC-DEMO",
    block_size=31,
    information_blocks=3,
    puncture_count=4,
    secret_block_weight=3,
    max_bp_iterations=40,
    channel_error_probability=0.10,
    min_sum_normalization=0.80,
    max_error_weight=28,
    scrambling_operations=62,
    max_sign_attempts=2000,
)

ORIGINAL_ARTICLE = Parameters(
    name="Punctured-QC-LDPC-ARTICLE",
    block_size=4096,
    information_blocks=3,
    puncture_count=64,
    secret_block_weight=13,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_error_weight=128,
    scrambling_operations=8192,
    max_sign_attempts=10000,
)

ESTIMATED_128 = Parameters(
    name="PUNCTURED-QC-LDPC-CFS-ESTIMATED-128",
    block_size=3000,          # n = 12000
    information_blocks=3,
    puncture_count=64,
    secret_block_weight=13,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_error_weight=128,
    scrambling_operations=8192,
    max_sign_attempts=10000,
)


ESTIMATED_192 = Parameters(
    name="PUNCTURED-QC-LDPC-CFS-ESTIMATED-192",
    block_size=4690,          # n = 18760
    information_blocks=3,
    puncture_count=64,
    secret_block_weight=13,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_error_weight=128,
    scrambling_operations=8192,
    max_sign_attempts=10000,
)


ESTIMATED_256 = Parameters(
    name="PUNCTURED-QC-LDPC-CFS-ESTIMATED-256",
    block_size=6400,          # n = 25600
    information_blocks=3,
    puncture_count=64,
    secret_block_weight=13,
    max_bp_iterations=80,
    channel_error_probability=0.003,
    min_sum_normalization=0.75,
    max_error_weight=128,
    scrambling_operations=8192,
    max_sign_attempts=10000,
)


PARAMETER_SETS = {
    "demo": DEMO,
    "original": ORIGINAL_ARTICLE,
    "estimated_128": ESTIMATED_128,
    "estimated_192": ESTIMATED_192,
    "estimated_256": ESTIMATED_256,
}
