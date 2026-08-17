from __future__ import annotations

from statistics import mean, median, stdev
from time import perf_counter

from common.rng import ShakeRNG

from .core import keygen, sign, verify
from .parameters import LMQCS128, LMQCSParameters


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": mean(values),
        "median_seconds": median(values),
        "std_seconds": stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def benchmark(
    par: LMQCSParameters = LMQCS128,
    repetitions: int = 10,
    message: bytes = b"LMQCS benchmark",
    seed: int = 12345,
) -> dict:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    rng = ShakeRNG.from_int(seed)

    # Warm up Numba and SymPy outside the measurements.
    warm_keys = keygen(par, rng)
    warm_sig = sign(b"warmup", warm_keys.secret_key, warm_keys.public_key, par, rng)
    if not verify(b"warmup", warm_sig, warm_keys.public_key, par):
        raise RuntimeError("Falha no warm-up.")

    keygen_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []
    attempts: list[int] = []

    for i in range(repetitions):
        start = perf_counter()
        keys = keygen(par, rng)
        keygen_times.append(perf_counter() - start)

        start = perf_counter()
        sig = sign(message + i.to_bytes(4, "little"), keys.secret_key, keys.public_key, par, rng)
        sign_times.append(perf_counter() - start)
        attempts.append(sig.attempts)

        start = perf_counter()
        valid = verify(message + i.to_bytes(4, "little"), sig, keys.public_key, par)
        verify_times.append(perf_counter() - start)
        if not valid:
            raise RuntimeError("A valid signature was rejected during the benchmark.")

    return {
        "parameter_set": par.name,
        "repetitions": repetitions,
        "keygen": _stats(keygen_times),
        "sign": _stats(sign_times),
        "verify": _stats(verify_times),
        "mean_attempts": mean(attempts),
        "public_key_bytes": par.public_key_bytes,
        "secret_key_bytes": par.secret_key_bytes,
        "signature_bytes": par.signature_bytes,
    }
