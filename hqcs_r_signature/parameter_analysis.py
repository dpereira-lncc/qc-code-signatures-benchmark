from __future__ import annotations

from dataclasses import asdict
from math import ceil, comb, log2, sqrt

from sympy import isprime

from .parameters import (
    Parameters,
    HQCS_R_NIST_1,
    HQCS_R_NIST_3,
    HQCS_R_NIST_5,
)


def challenge_entropy_bits(par: Parameters) -> float:
    return par.omega_c + log2(comb(par.k, par.omega_c))


def public_key_forgery_exponent(par: Parameters) -> float:
    """
    -log2((p/q)^k) = k*log2(q/p).
    """
    return par.k * log2(par.q / par.p)


def acceptance_lower_bound(par: Parameters) -> dict[str, float | int]:
    """
    Corollary 1 of the article.

    For sparse ternary c of weight omega_c and uniform e in
    {-ell_e,...,ell_e}, temos:
        sigma_c^2 = omega_c/k
        sigma_e^2 = ell_e(ell_e+1)/3
        sigma^2 = k*sigma_c^2*sigma_e^2
                = omega_c*ell_e(ell_e+1)/3.
    """
    sigma = sqrt(
        par.omega_c
        * par.ell_e
        * (par.ell_e + 1)
        / 3
    )

    n0 = ceil(sigma)
    n1 = ceil(2 * sigma) - ceil(sigma)
    n2 = ceil(3 * sigma) - ceil(2 * sigma)
    n3 = par.omega_c - n0 - n1 - n2

    rho = (
        0.5 * n0
        + 0.15865 * n1
        + 0.02275 * n2
        + 0.00135 * max(n3, 0)
    )

    ell = par.q // par.p
    single_failure_bound = 2 * (ell + 1) * rho / par.q
    tau = (1.0 - single_failure_bound) ** (2 * par.k)

    return {
        "sigma": sigma,
        "rho": rho,
        "ell_from_q_p": ell,
        "n0": n0,
        "n1": n1,
        "n2": n2,
        "n3": n3,
        "acceptance_lower_bound": tau,
    }


def analyze(par: Parameters) -> dict:
    acceptance = acceptance_lower_bound(par)

    return {
        "name": par.name,
        "security_target_bits": par.security_bits,
        "parameters": {
            "k": par.k,
            "q": par.q,
            "p": par.p,
            "ell": par.ell,
            "ell_e": par.ell_e,
            "omega_c": par.omega_c,
        },
        "primality": {
            "p_is_prime": bool(isprime(par.p)),
            "q_is_prime": bool(isprime(par.q)),
        },
        "algebraic_conditions": {
            "two_p_lt_q_minus_one": 2 * par.p < par.q - 1,
            # The condition printed in the article, q-1 <= ell*p, is inconsistent
            # com ell=floor(q/p), salvo no caso excepcional q=ell*p+1.
            # The parameters in Table 1 themselves do not satisfy it.
            "q_minus_one_lt_ell_plus_one_p": (
                par.q - 1 < (par.ell + 1) * par.p
            ),
            "floor_q_over_p_equals_ell": par.q // par.p == par.ell,
        },
        "challenge_entropy_bits": challenge_entropy_bits(par),
        "required_challenge_entropy_bits": 2 * par.security_bits,
        "public_key_forgery_exponent_bits": public_key_forgery_exponent(par),
        "acceptance": acceptance,
        "sizes_bytes": {
            "public_key": par.public_key_bytes,
            "secret_key": par.secret_key_bytes,
            "signature": par.signature_bytes,
        },
    }


def all_candidate_analyses() -> list[dict]:
    return [
        analyze(HQCS_R_NIST_1),
        analyze(HQCS_R_NIST_3),
        analyze(HQCS_R_NIST_5),
    ]


if __name__ == "__main__":
    import json
    print(json.dumps(all_candidate_analyses(), indent=2))
