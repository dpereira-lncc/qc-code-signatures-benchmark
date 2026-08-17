from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log2, sqrt


@dataclass(frozen=True)
class LMQCSParameters:
    name: str
    security_level: int
    n: int
    q: int
    ell_e: int
    ell_u: int
    omega_e_bar: int
    omega_c: int
    t: float
    public_key_bytes_reported: int
    secret_key_bytes_reported: int
    signature_bytes_reported: int

    @property
    def sigma_s(self) -> float:
        return sqrt(
            self.omega_e_bar * self.ell_e * (self.ell_e + 1) / 3
            + self.omega_c * self.ell_u * (self.ell_u + 1) / 3
        )

    @property
    def gamma(self) -> int:
        # The paper lists gamma=t*sigma_s with decimals. The verifier needs an
        # integer bound, so use ceil, matching the size formula 2*ceil(gamma)+1.
        return ceil(self.t * self.sigma_s)

    @property
    def q_bits(self) -> int:
        return ceil(log2(self.q))

    @property
    def secret_coeff_bits(self) -> int:
        return ceil(log2(2 * self.ell_e + 1))

    @property
    def s_bits(self) -> int:
        return ceil(log2(2 * self.gamma + 1))

    @property
    def challenge_bits(self) -> int:
        return ceil(log2((1 << self.omega_c) * comb(self.n, self.omega_c)))

    @property
    def public_key_bytes(self) -> int:
        return ceil(self.n * self.q_bits / 8)

    @property
    def secret_key_bytes(self) -> int:
        return ceil(2 * self.n * self.secret_coeff_bits / 8)

    @property
    def signature_bytes(self) -> int:
        return ceil((2 * self.n * self.s_bits + self.challenge_bits) / 8)

    def validate_reported_sizes(self) -> None:
        actual = (self.public_key_bytes, self.secret_key_bytes, self.signature_bytes)
        reported = (
            self.public_key_bytes_reported,
            self.secret_key_bytes_reported,
            self.signature_bytes_reported,
        )
        if actual != reported:
            raise ValueError(f"Computed sizes {actual} differ from paper {reported}.")


LMQCS_I = LMQCSParameters(
    name="LM-QCS-I", security_level=128, n=691, q=5479,
    ell_e=2, ell_u=28, omega_e_bar=345, omega_c=41, t=10.1,
    public_key_bytes_reported=1123, secret_key_bytes_reported=519,
    signature_bytes_reported=2106,
)

LMQCS_II = LMQCSParameters(
    name="LM-QCS-II", security_level=192, n=1009, q=8623,
    ell_e=3, ell_u=35, omega_e_bar=455, omega_c=61, t=11.0,
    public_key_bytes_reported=1766, secret_key_bytes_reported=757,
    signature_bytes_reported=3076,
)

LMQCS_III = LMQCSParameters(
    name="LM-QCS-III", security_level=256, n=1201, q=12569,
    ell_e=4, ell_u=41, omega_e_bar=705, omega_c=83, t=10.1,
    public_key_bytes_reported=2102, secret_key_bytes_reported=1201,
    signature_bytes_reported=3968,
)

PARAMETER_SETS = {
    128: LMQCS_I,
    192: LMQCS_II,
    256: LMQCS_III,
    1: LMQCS_I,
    2: LMQCS_II,
    3: LMQCS_III,
}

for _p in (LMQCS_I, LMQCS_II, LMQCS_III):
    _p.validate_reported_sizes()
