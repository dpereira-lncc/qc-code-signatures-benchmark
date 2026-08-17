from __future__ import annotations

import argparse
import json
import platform
import random
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any, Callable

from hqcs_r_signature.core import keygen as hqcs_keygen
from hqcs_r_signature.core import sign as hqcs_sign
from hqcs_r_signature.core import verify as hqcs_verify
from hqcs_r_signature.parameters import HQCS_R_NIST_1, HQCS_R_NIST_3, HQCS_R_NIST_5
from lm_qcs_python.core import keygen as lm_qcs_keygen
from lm_qcs_python.core import sign as lm_qcs_sign
from lm_qcs_python.core import verify as lm_qcs_verify
from lm_qcs_python.parameters import LMQCS_I, LMQCS_II, LMQCS_III
from lmqcs_python.core import keygen as lmqcs_keygen
from lmqcs_python.core import sign as lmqcs_sign
from lmqcs_python.core import verify as lmqcs_verify
from lmqcs_python.parameters import LMQCS128, LMQCS192, LMQCS256
from qc_ldpc_cfs.core import keygen as cfs_keygen
from qc_ldpc_cfs.core import sign as cfs_sign
from qc_ldpc_cfs.core import verify as cfs_verify
from qc_ldpc_cfs.parameters import DEMO as CFS_DEMO
from qc_ldpc_cfs.parameters import ESTIMATED_128 as CFS_128
from qc_ldpc_cfs.parameters import ESTIMATED_192 as CFS_192
from qc_ldpc_cfs.parameters import ESTIMATED_256 as CFS_256
from qc_ldpc_cfs.parameters import ORIGINAL_ARTICLE as CFS_ORIGINAL
from qc_ldpc_cfs_punc.core import keygen as punc_keygen
from qc_ldpc_cfs_punc.core import sign as punc_sign
from qc_ldpc_cfs_punc.core import verify as punc_verify
from qc_ldpc_cfs_punc.parameters import DEMO as PUNC_DEMO
from qc_ldpc_cfs_punc.parameters import ESTIMATED_128 as PUNC_128
from qc_ldpc_cfs_punc.parameters import ESTIMATED_192 as PUNC_192
from qc_ldpc_cfs_punc.parameters import ESTIMATED_256 as PUNC_256
from qc_ldpc_cfs_punc.parameters import ORIGINAL_ARTICLE as PUNC_ORIGINAL
from common.rng import ShakeRNG
from common.errors import SigningFailure


DEFAULT_MESSAGE_BITS = 1024
DEFAULT_MESSAGE = b"A" * (DEFAULT_MESSAGE_BITS // 8)

assert len(DEFAULT_MESSAGE) * 8 == DEFAULT_MESSAGE_BITS


@dataclass(frozen=True)
class Scenario:
    implementation: str
    variant: str
    security_bits: int | None
    parameters: Any
    keygen: Callable[..., Any]
    sign: Callable[..., Any]
    verify: Callable[..., bool]
    seed: int
    enabled_by_default: bool = True
    sign_attempt_limit: int | None = None
    tolerate_sign_failure: bool = False

    @property
    def identifier(self) -> str:
        return f"{self.implementation}:{self.variant}"


SCENARIOS = (
    Scenario("lm_qcs_python", "128", 128, LMQCS_I, lm_qcs_keygen, lm_qcs_sign, lm_qcs_verify, 10128),
    Scenario("lm_qcs_python", "192", 192, LMQCS_II, lm_qcs_keygen, lm_qcs_sign, lm_qcs_verify, 10192),
    Scenario("lm_qcs_python", "256", 256, LMQCS_III, lm_qcs_keygen, lm_qcs_sign, lm_qcs_verify, 10256),
    Scenario("lmqcs_python", "128", 128, LMQCS128, lmqcs_keygen, lmqcs_sign, lmqcs_verify, 20128),
    Scenario("lmqcs_python", "192", 192, LMQCS192, lmqcs_keygen, lmqcs_sign, lmqcs_verify, 20192),
    Scenario("lmqcs_python", "256", 256, LMQCS256, lmqcs_keygen, lmqcs_sign, lmqcs_verify, 20256),
    # HQCS-R 192/256 are experimental candidates, not paper parameter sets.
    Scenario("hqcs_r_signature", "128", 128, HQCS_R_NIST_1, hqcs_keygen, hqcs_sign, hqcs_verify, 40128),
    Scenario("hqcs_r_signature", "192", 192, HQCS_R_NIST_3, hqcs_keygen, hqcs_sign, hqcs_verify, 40192),
    Scenario("hqcs_r_signature", "256", 256, HQCS_R_NIST_5, hqcs_keygen, hqcs_sign, hqcs_verify, 40256),
    # The CFS profiles do not have validated 128/192/256-bit equivalence and
    # therefore are opt-in. Article signing is bounded and failures are data.
    Scenario("qc_ldpc_cfs", "demo", None, CFS_DEMO, cfs_keygen, cfs_sign, cfs_verify, 50101, False),
    Scenario("qc_ldpc_cfs", "original", None, CFS_ORIGINAL, cfs_keygen, cfs_sign, cfs_verify, 50102, False, 100, True),
    Scenario("qc_ldpc_cfs", "estimated_128", None, CFS_128, cfs_keygen, cfs_sign, cfs_verify, 50128, False, 100, True),
    Scenario("qc_ldpc_cfs", "estimated_192", None, CFS_192, cfs_keygen, cfs_sign, cfs_verify, 50192, False, 100, True),
    Scenario("qc_ldpc_cfs", "estimated_256", None, CFS_256, cfs_keygen, cfs_sign, cfs_verify, 50256, False, 100, True),
    Scenario("qc_ldpc_cfs_punc", "demo", None, PUNC_DEMO, punc_keygen, punc_sign, punc_verify, 60101, False),
    Scenario("qc_ldpc_cfs_punc", "original", None, PUNC_ORIGINAL, punc_keygen, punc_sign, punc_verify, 60102, False, 100, True),
    Scenario("qc_ldpc_cfs_punc", "estimated_128", None, PUNC_128, punc_keygen, punc_sign, punc_verify, 60128, False, 100, True),
    Scenario("qc_ldpc_cfs_punc", "estimated_192", None, PUNC_192, punc_keygen, punc_sign, punc_verify, 60192, False, 100, True),
    Scenario("qc_ldpc_cfs_punc", "estimated_256", None, PUNC_256, punc_keygen, punc_sign, punc_verify, 60256, False, 100, True),
)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_or_none(values: list[float]) -> dict[str, float] | None:
    return summarize(values) if values else None


def save_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timed(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    start = perf_counter()
    result = function(*args)
    return result, perf_counter() - start


def _sign(
    scenario: Scenario,
    message: bytes,
    keys: Any,
    rng: ShakeRNG,
    *,
    warm_up: bool = False,
    attempt_limit_override: int | None = None,
) -> Any:
    attempt_limit = (
        attempt_limit_override
        if (
            attempt_limit_override is not None
            and hasattr(scenario.parameters, "max_sign_attempts")
        )
        else scenario.sign_attempt_limit
    )
    if warm_up and scenario.tolerate_sign_failure:
        attempt_limit = 1
    arguments = (
        message,
        keys.secret_key,
        keys.public_key,
        scenario.parameters,
        rng,
    )
    if attempt_limit is None:
        return scenario.sign(*arguments)
    return scenario.sign(*arguments, max_attempts=attempt_limit)


def run_scenario(
    scenario: Scenario,
    repetitions: int,
    message: bytes,
    sign_attempt_limit_override: int | None = None,
) -> dict[str, Any]:
    par = scenario.parameters
    rng = ShakeRNG.from_int(scenario.seed)

    # Warm up lazy paths outside all reported measurements. A bounded CFS
    # decoding failure is expected for the full article profiles.
    warm_keys = scenario.keygen(par, rng)
    try:
        warm_signature = _sign(
            scenario,
            message,
            warm_keys,
            rng,
            warm_up=True,
            attempt_limit_override=sign_attempt_limit_override,
        )
    except SigningFailure:
        if not scenario.tolerate_sign_failure:
            raise
    else:
        if not scenario.verify(message, warm_signature, warm_keys.public_key, par):
            raise RuntimeError(f"Warm-up failed for {scenario.identifier}.")

    keygen_times: list[float] = []
    sign_times: list[float] = []
    verify_times: list[float] = []
    attempts: list[int] = []
    signing_runs: list[dict[str, Any]] = []
    signing_errors: list[str] = []
    successful_signatures = 0
    failed_signatures = 0
    wall_start = perf_counter()

    for repetition in range(1, repetitions + 1):
        keys, keygen_time = _timed(scenario.keygen, par, rng)
        keygen_times.append(keygen_time)

        sign_start = perf_counter()
        try:
            signature = _sign(
                scenario,
                message,
                keys,
                rng,
                attempt_limit_override=sign_attempt_limit_override,
            )
        except SigningFailure as error:
            elapsed = perf_counter() - sign_start
            sign_times.append(elapsed)
            attempts.append(error.attempts)
            signing_runs.append(
                {
                    "repetition": repetition,
                    "status": "failed",
                    "attempts": error.attempts,
                    "elapsed_seconds": elapsed,
                }
            )
            failed_signatures += 1
            signing_errors.append(str(error))
            if not scenario.tolerate_sign_failure:
                raise
            print(
                f"  {scenario.identifier}: {repetition}/{repetitions} "
                f"(bounded failure after {error.attempts} attempts)",
                flush=True,
            )
            continue

        elapsed = perf_counter() - sign_start
        sign_times.append(elapsed)
        attempts.append(signature.attempts)
        signing_runs.append(
            {
                "repetition": repetition,
                "status": "success",
                "attempts": signature.attempts,
                "elapsed_seconds": elapsed,
            }
        )
        successful_signatures += 1

        valid, verify_time = _timed(
            scenario.verify, message, signature, keys.public_key, par
        )
        verify_times.append(verify_time)
        if not valid:
            raise RuntimeError(
                f"Signature rejected for {scenario.identifier}, "
                f"repetition {repetition}."
            )
        print(
            f"  {scenario.identifier}: {repetition}/{repetitions}",
            flush=True,
        )

    return {
        "implementation": scenario.implementation,
        "variant": scenario.variant,
        "parameter_set": par.name,
        "security_bits": scenario.security_bits,
        "repetitions": repetitions,
        "process_isolated": True,
        "rng": "ShakeRNG-SHAKE256-v1",
        "status": "ok" if failed_signatures == 0 else "signing_incomplete",
        "wall_seconds": perf_counter() - wall_start,
        "keygen_seconds": summarize(keygen_times),
        "sign_seconds": summarize(sign_times),
        "verify_seconds": summarize_or_none(verify_times),
        "successful_signatures": successful_signatures,
        "failed_signatures": failed_signatures,
        "sign_success_rate": successful_signatures / repetitions,
        "mean_attempts": mean(attempts),
        "total_sign_attempts": sum(attempts),
        "total_sign_seconds": sum(sign_times),
        "seconds_per_sign_attempt": sum(sign_times) / sum(attempts),
        "signing_runs": signing_runs,
        "sign_attempt_limit": (
            sign_attempt_limit_override
            if (
                sign_attempt_limit_override is not None
                and hasattr(par, "max_sign_attempts")
            )
            else scenario.sign_attempt_limit
            if scenario.sign_attempt_limit is not None
            else getattr(par, "max_sign_attempts", None)
        ),
        "signing_errors": sorted(set(signing_errors)),
        "public_key_bytes": getattr(par, "public_key_bytes", None),
        "secret_key_bytes": getattr(par, "secret_key_bytes", None),
        "signature_bytes": getattr(par, "signature_bytes", None),
    }


def _worker(
    scenario_index: int,
    repetitions: int,
    message: bytes,
    sign_attempt_limit_override: int | None,
    connection: Any,
) -> None:
    try:
        result = run_scenario(
            SCENARIOS[scenario_index],
            repetitions,
            message,
            sign_attempt_limit_override,
        )
        connection.send(("ok", result))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _timeout_result(
    scenario: Scenario,
    repetitions: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    par = scenario.parameters
    return {
        "implementation": scenario.implementation,
        "variant": scenario.variant,
        "parameter_set": par.name,
        "security_bits": scenario.security_bits,
        "repetitions": repetitions,
        "process_isolated": True,
        "rng": "ShakeRNG-SHAKE256-v1",
        "status": "timeout",
        "timeout_seconds": timeout_seconds,
        "keygen_seconds": None,
        "sign_seconds": None,
        "verify_seconds": None,
        "successful_signatures": 0,
        "failed_signatures": repetitions,
        "sign_success_rate": 0.0,
        "public_key_bytes": getattr(par, "public_key_bytes", None),
        "secret_key_bytes": getattr(par, "secret_key_bytes", None),
        "signature_bytes": getattr(par, "signature_bytes", None),
    }


def run_isolated(
    scenario_index: int,
    repetitions: int,
    message: bytes,
    timeout_seconds: float = 600.0,
    sign_attempt_limit_override: int | None = None,
) -> dict[str, Any]:
    context = get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(
            scenario_index,
            repetitions,
            message,
            sign_attempt_limit_override,
            send,
        ),
    )
    process.start()
    send.close()
    if not receive.poll(timeout_seconds):
        process.terminate()
        process.join()
        receive.close()
        return _timeout_result(
            SCENARIOS[scenario_index],
            repetitions,
            timeout_seconds,
        )
    try:
        status, payload = receive.recv()
    except EOFError as error:
        process.join()
        raise RuntimeError(
            f"Benchmark process ended without a result (exit={process.exitcode})."
        ) from error
    finally:
        receive.close()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark process failed (exit={process.exitcode}).")
    if status == "error":
        raise RuntimeError(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated signature benchmark. The three schemes with 128/192/256 "
            "parameter sets run by default; CFS profiles are opt-in."
        )
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--message-bits",
        type=int,
        default=DEFAULT_MESSAGE_BITS,
        help="Message length in bits; must be positive and divisible by 8.",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_10.json"))
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.identifier for scenario in SCENARIOS),
        help="Run only the specified scenario; may be repeated.",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=20260722,
        help="Seed used to shuffle scenario order.",
    )
    parser.add_argument(
        "--scenario-timeout",
        type=float,
        default=600.0,
        help="Timeout for each isolated scenario, in seconds.",
    )
    parser.add_argument(
        "--max-sign-attempts",
        type=int,
        default=None,
        help=(
            "Override the signing-attempt limit for CFS scenarios. "
            "Use 10000 to exhaust the limit of research-scale profiles."
        ),
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.message_bits <= 0:
        parser.error("--message-bits must be positive")
    if args.message_bits % 8 != 0:
        parser.error("--message-bits must be divisible by 8")
    if args.scenario_timeout <= 0:
        parser.error("--scenario-timeout must be positive")
    if args.max_sign_attempts is not None and args.max_sign_attempts <= 0:
        parser.error("--max-sign-attempts must be positive")

    message = b"A" * (args.message_bits // 8)
    selected = set(args.scenario or ())
    order = [
        index
        for index, scenario in enumerate(SCENARIOS)
        if (
            (selected and scenario.identifier in selected)
            or (not selected and scenario.enabled_by_default)
        )
    ]
    random.Random(args.order_seed).shuffle(order)
    execution_order = [SCENARIOS[index].identifier for index in order]
    document: dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repetitions_per_scenario": args.repetitions,
        "message_bits": args.message_bits,
        "message_utf8": message.decode("ascii"),
        "message_hex": message.hex(),
        "order_seed": args.order_seed,
        "scenario_timeout_seconds": args.scenario_timeout,
        "max_sign_attempts_override": args.max_sign_attempts,
        "execution_order": execution_order,
        "results": [],
    }
    save_json(args.output, document)

    for scenario_index in order:
        scenario = SCENARIOS[scenario_index]
        print(f"Starting {scenario.identifier}...", flush=True)
        document["results"].append(
            run_isolated(
                scenario_index,
                args.repetitions,
                message,
                args.scenario_timeout,
                args.max_sign_attempts,
            )
        )
        save_json(args.output, document)

    document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, document)
    print(f"Benchmark completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
