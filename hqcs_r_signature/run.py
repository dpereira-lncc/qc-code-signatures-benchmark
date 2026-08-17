from __future__ import annotations

import argparse
from time import perf_counter

from common.rng import ShakeRNG

from .core import keygen, sign, verify
from .parameters import PARAMETER_SETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        choices=("1", "2", "3", "nist1", "nist3", "nist5"),
        default="nist1",
    )
    parser.add_argument("--message", default="HQCS-R test message")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    key = int(args.set) if args.set in {"1", "2", "3"} else args.set
    par = PARAMETER_SETS[key]
    message = args.message.encode()

    print(f"Conjunto: {par.name}")
    print(
        f"k={par.k}, q={par.q}, p={par.p}, ell={par.ell}, "
        f"ell_e={par.ell_e}, omega_c={par.omega_c}"
    )
    print(f"Target security: {par.security_bits} bits")
    print(f"Estimated acceptance rate: {par.claimed_acceptance:.8f}")

    rng = ShakeRNG.from_int(args.seed)
    start = perf_counter()
    keys = keygen(par, rng)
    keygen_seconds = perf_counter() - start
    print(f"KeyGen: {keygen_seconds:.6f} s")

    start = perf_counter()
    signature = sign(message, keys.secret_key, keys.public_key, par, rng)
    sign_seconds = perf_counter() - start
    print(f"Sign: {sign_seconds:.6f} s")
    print(f"Tentativas: {signature.attempts}")

    valid = verify(message, signature, keys.public_key, par)
    print(f"Verify: {valid}")

    print(f"PK: {len(keys.public_key.to_bytes(par))} bytes")
    print(f"SK: {len(keys.secret_key.to_bytes(par))} bytes")
    print(f"SIG: {len(signature.to_bytes(par))} bytes")


if __name__ == "__main__":
    main()
