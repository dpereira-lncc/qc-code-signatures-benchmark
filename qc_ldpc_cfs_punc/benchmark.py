from __future__ import annotations

from statistics import mean, median, stdev
from time import perf_counter

from common.rng import ShakeRNG
from common.errors import SigningFailure

from .core import keygen, sign, verify
from .parameters import DEMO, Parameters


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _default_attempt_limit(par: Parameters) -> int:
    """Keep research-scale profiles bounded unless explicitly overridden."""
    return par.max_sign_attempts if par is DEMO else 100


def benchmark(
    par: Parameters = DEMO,
    repetitions: int = 10,
    message: bytes = b"A" * 128,
    seed: int = 12345,
    max_sign_attempts: int | None = None,
) -> dict:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    attempt_limit = max_sign_attempts
    if attempt_limit is None:
        attempt_limit = _default_attempt_limit(par)

    rng = ShakeRNG.from_int(seed)
    warm_keys = keygen(par, rng, warm_up_decoder=True)

    keygen_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []
    attempts: list[int] = []
    weights: list[int] = []
    successful = 0
    failed = 0

    for _ in range(repetitions):
        start = perf_counter()
        keys = keygen(par, rng, warm_up_decoder=False)
        keygen_times.append(perf_counter() - start)

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
            sign_times.append(perf_counter() - start)
            attempts.append(error.attempts)
            failed += 1
            continue

        sign_times.append(perf_counter() - start)
        attempts.append(signature.attempts)
        weights.append(signature.weight)
        successful += 1

        start = perf_counter()
        accepted = verify(message, signature, keys.public_key, par)
        verify_times.append(perf_counter() - start)
        if not accepted:
            raise RuntimeError("A freshly generated signature was rejected.")

    return {
        "scheme": par.name,
        "status": "ok" if failed == 0 else "signing_incomplete",
        "repetitions": repetitions,
        "rng": "ShakeRNG-SHAKE256-v1",
        "keygen_seconds": _summary(keygen_times),
        "sign_seconds": _summary(sign_times),
        "verify_seconds": _summary(verify_times) if verify_times else None,
        "successful_signatures": successful,
        "failed_signatures": failed,
        "sign_success_rate": successful / repetitions,
        "mean_attempts": mean(attempts),
        "mean_signature_weight": mean(weights) if weights else None,
        "sign_attempt_limit": attempt_limit,
        "public_key_bytes": par.public_key_bytes,
        "signature_bytes": par.signature_bytes,
    }
