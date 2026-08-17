from __future__ import annotations

import argparse

from time import perf_counter

from common.rng import ShakeRNG

from .core import keygen, sign, verify
from .parameters import PARAMETER_SETS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LMQCS KeyGen -> Sign -> Verify.")
    parser.add_argument("--level", type=int, choices=(128, 192, 256), default=128)
    parser.add_argument("--message", default="Complete LMQCS test")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    par = PARAMETER_SETS[args.level]
    rng = ShakeRNG.from_int(args.seed)
    message = args.message.encode()

    start = perf_counter()
    keys = keygen(par, rng)
    keygen_seconds = perf_counter() - start
    start = perf_counter()
    signature = sign(message, keys.secret_key, keys.public_key, par, rng)
    sign_seconds = perf_counter() - start
    valid = verify(message, signature, keys.public_key, par)

    print(f"Conjunto: {par.name}")
    print(f"KeyGen: {keygen_seconds:.6f} s ({keys.attempts} tentativa(s))")
    print(f"Sign: {sign_seconds:.6f} s ({signature.attempts} tentativa(s))")
    print(f"Verify: {valid}")
    print(f"PK: {len(keys.public_key.to_bytes(par))} bytes")
    print(f"SK: {len(keys.secret_key.to_bytes(par))} bytes")
    print(f"SIG: {len(signature.to_bytes(par))} bytes")


if __name__ == "__main__":
    main()
