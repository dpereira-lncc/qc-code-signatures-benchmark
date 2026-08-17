from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2, sqrt


@dataclass(frozen=True)
class LMQCSParameters:
    name: str
    security_level: int
    n: int
    q: int
    ell: int
    eta: int
    omega_e: int
    omega_c: int

    def __post_init__(self) -> None:
        if self.n <= 0 or self.q <= 2:
            raise ValueError("n and q must be positive, with q > 2.")
        if self.omega_e != 2 * (self.n // 3) + 1:
            raise ValueError("omega_e must equal 2*floor(n/3)+1.")
        if self.omega_c != self.omega_e:
            raise ValueError("The article defines omega_c = omega_e.")

    @property
    def sigma(self) -> float:
        return sqrt(2 * self.omega_c * self.ell * (self.ell + 1) / 3)

    @property
    def gamma(self) -> int:
        return ceil(self.eta * self.sigma)

    @property
    def rho(self) -> float:
        return self.n * self.sigma

    @property
    def q_bits(self) -> int:
        return ceil(log2(self.q))

    @property
    def s_bits(self) -> int:
        return ceil(log2(2 * self.gamma + 1))

    @property
    def public_key_bytes(self) -> int:
        # h and b; h^{-1} can be recomputed from h.
        return ceil(2 * self.n * self.q_bits / 8)

    @property
    def secret_key_bytes(self) -> int:
        # Ternary e1 and e2: 2 bits per coefficient.
        return ceil(4 * self.n / 8)

    @property
    def signature_bytes(self) -> int:
        # c: 2 bits/coef.; b_bar,u: q_bits; s1,s2: s_bits.
        return ceil(2 * self.n * (1 + self.q_bits + self.s_bits) / 8)


LMQCS128 = LMQCSParameters(
    name="LMQCS-128",
    security_level=128,
    n=739,
    q=4073,
    ell=3,
    eta=10,
    omega_e=493,
    omega_c=493,
)

LMQCS192 = LMQCSParameters(
    name="LMQCS-192",
    security_level=192,
    n=997,
    q=8191,
    ell=3,
    eta=10,
    omega_e=665,
    omega_c=665,
)

LMQCS256 = LMQCSParameters(
    name="LMQCS-256",
    security_level=256,
    n=1301,
    q=16369,
    ell=3,
    eta=10,
    omega_e=867,
    omega_c=867,
)

PARAMETER_SETS = {
    128: LMQCS128,
    192: LMQCS192,
    256: LMQCS256,
}
