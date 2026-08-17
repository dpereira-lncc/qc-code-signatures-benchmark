from __future__ import annotations

from statistics import mean, median, stdev
from time import perf_counter

from .core import keygen, sign, verify
from .parameters import Parameters, HQCS_R_1
from common.rng import ShakeRNG


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def benchmark(
    par: Parameters = HQCS_R_1,
    repetitions: int = 5,
    message: bytes = b"HQCS-R benchmark",
    seed: int = 12345,
) -> dict:
    rng = ShakeRNG.from_int(seed)
    warm_keys = keygen(par, rng)
    warm_signature = sign(message, warm_keys.secret_key, warm_keys.public_key, par, rng)
    if not verify(message, warm_signature, warm_keys.public_key, par):
        raise RuntimeError("Warm-up failed.")

    keygen_times = []
    sign_times = []
    verify_times = []
    attempts = []

    for _ in range(repetitions):
        start = perf_counter()
        keys = keygen(par, rng)
        keygen_times.append(perf_counter() - start)

        start = perf_counter()
        signature = sign(message, keys.secret_key, keys.public_key, par, rng)
        sign_times.append(perf_counter() - start)
        attempts.append(signature.attempts)

        start = perf_counter()
        valid = verify(message, signature, keys.public_key, par)
        verify_times.append(perf_counter() - start)

        if not valid:
            raise RuntimeError("A freshly generated signature was rejected.")

    return {
        "scheme": par.name,
        "repetitions": repetitions,
        "keygen_seconds": _summary(keygen_times),
        "sign_seconds": _summary(sign_times),
        "verify_seconds": _summary(verify_times),
        "mean_attempts": mean(attempts),
        "public_key_bytes": par.public_key_bytes,
        "secret_key_bytes": par.secret_key_bytes,
        "signature_bytes": par.signature_bytes,
    }
