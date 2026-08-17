from __future__ import annotations

import argparse
from time import perf_counter

from common.rng import ShakeRNG
from common.errors import SigningFailure

from .core import keygen, sign, verify_detailed
from .parameters import PARAMETER_SETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=tuple(PARAMETER_SETS),
        default="demo",
    )
    parser.add_argument("--message-bits", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-sign-attempts", type=int, default=None)
    args = parser.parse_args()
    if args.message_bits <= 0 or args.message_bits % 8:
        parser.error("--message-bits must be positive and divisible by 8")

    par = PARAMETER_SETS[args.profile]
    message = b"A" * (args.message_bits // 8)
    attempt_limit = args.max_sign_attempts
    if attempt_limit is None:
        attempt_limit = 100 if args.profile != "demo" else par.max_sign_attempts
    rng = ShakeRNG.from_int(args.seed)

    print(f"Perfil: {par.name}")
    print(f"n={par.n}, k={par.k}, r={par.r}, taxa={par.code_rate:.4f}")
    start = perf_counter()
    keys = keygen(par, rng, warm_up_decoder=True)
    print(f"KeyGen: {perf_counter() - start:.6f} s")

    start = perf_counter()
    try:
        signature = sign(
            message,
            keys.secret_key,
            keys.public_key,
            par,
            rng,
            max_attempts=attempt_limit,
        )
    except SigningFailure as error:
        print(
            f"Sign: falhou em {perf_counter() - start:.6f} s "
            f"after {error.attempts} attempts"
        )
        return
    print(
        f"Sign: {perf_counter() - start:.6f} s; "
        f"attempts={signature.attempts}; BP={signature.bp_iterations}"
    )

    start = perf_counter()
    result = verify_detailed(message, signature, keys.public_key, par)
    print(
        f"Verify: {result.accepted}; time={perf_counter() - start:.6f} s; "
        f"{result.reason}"
    )
    print(
        f"PK={len(keys.public_key.to_bytes())} B; "
        f"SIG={len(signature.to_bytes(par))} B"
    )


if __name__ == "__main__":
    main()
