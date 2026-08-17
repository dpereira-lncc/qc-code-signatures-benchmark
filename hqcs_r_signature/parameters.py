from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2


@dataclass(frozen=True)
class Parameters:
    name: str
    security_bits: int
    k: int
    q: int
    p: int
    ell: int
    ell_e: int
    omega_c: int
    claimed_acceptance: float
    experimental_acceptance: float
    max_sign_attempts: int = 1000
    public_key_size_override: int | None = None

    @property
    def q_bits(self) -> int:
        return ceil(log2(self.q))

    @property
    def secret_symbol_bits(self) -> int:
        return ceil(log2(2 * self.ell_e + 1))

    @property
    def challenge_symbol_bits(self) -> int:
        # {-1,0,1}
        return 2

    @property
    def seed_bytes(self) -> int:
        return (2 * self.security_bits + 7) // 8

    @property
    def computed_public_key_bytes(self) -> int:
        return ceil((self.k * self.q_bits + 2 * self.security_bits) / 8)

    @property
    def public_key_bytes(self) -> int:
        if self.public_key_size_override is not None:
            return self.public_key_size_override
        return self.computed_public_key_bytes

    @property
    def secret_key_bytes(self) -> int:
        return ceil(self.k * self.secret_symbol_bits / 8)

    @property
    def signature_bytes(self) -> int:
        return ceil(
            (
                self.k * self.q_bits
                + 2 * self.security_bits
                + self.k * self.challenge_symbol_bits
            )
            / 8
        )


HQCS_R_1 = Parameters(
    name="HQCS-R-1",
    security_bits=128,
    k=1511,
    q=2_131_128_193,
    p=16_780_537,
    ell=126,
    ell_e=2,
    omega_c=73,
    claimed_acceptance=0.99683,
    experimental_acceptance=0.99823,
)

HQCS_R_2 = Parameters(
    name="HQCS-R-2",
    security_bits=128,
    k=1511,
    q=2_147_446_991,
    p=16_518_823,
    ell=130,
    ell_e=3,
    omega_c=71,
    claimed_acceptance=0.99566,
    experimental_acceptance=0.99763,
)

HQCS_R_3 = Parameters(
    name="HQCS-R-3",
    security_bits=128,
    k=1619,
    q=32_230_149_377,
    p=125_899_021,
    ell=256,
    ell_e=4,
    omega_c=91,
    claimed_acceptance=0.99910,
    experimental_acceptance=0.99955,
    public_key_size_override=7520,
)


HQCS_R_NIST_1 = Parameters(
    name="HQCS-R-NIST-1",
    security_bits=128,
    k=1511,
    q=2_131_128_193,
    p=16_780_537,
    ell=126,
    ell_e=2,
    omega_c=73,
    claimed_acceptance=0.99683,
    experimental_acceptance=0.99823,
)

HQCS_R_NIST_3 = Parameters(
    name="HQCS-R-NIST-3-CANDIDATE",
    security_bits=192,
    k=2267,
    q=3_777_613_439,
    p=29_981_059,
    ell=126,
    ell_e=2,
    omega_c=110,
    claimed_acceptance=0.9968622263,
    experimental_acceptance=0.0,
)

HQCS_R_NIST_5 = Parameters(
    name="HQCS-R-NIST-5-CANDIDATE",
    security_bits=256,
    k=3022,
    q=5_964_784_949,
    p=47_339_563,
    ell=126,
    ell_e=2,
    omega_c=146,
    claimed_acceptance=0.9968622250,
    experimental_acceptance=0.0,
)

PARAMETER_SETS = {
    1: HQCS_R_1,
    2: HQCS_R_2,
    3: HQCS_R_3,
    "HQCS-R-1": HQCS_R_1,
    "HQCS-R-2": HQCS_R_2,
    "HQCS-R-3": HQCS_R_3,
    "nist1": HQCS_R_NIST_1,
    "nist3": HQCS_R_NIST_3,
    "nist5": HQCS_R_NIST_5,
    "HQCS-R-NIST-1": HQCS_R_NIST_1,
    "HQCS-R-NIST-3": HQCS_R_NIST_3,
    "HQCS-R-NIST-5": HQCS_R_NIST_5,
}
