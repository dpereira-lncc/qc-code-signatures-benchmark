from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from itertools import product
from math import sqrt
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any, Sequence

from qc_ldpc_cfs.core import keygen, sign, verify
from qc_ldpc_cfs.parameters import DEMO
from common.rng import ShakeRNG
from common.errors import SigningFailure


@dataclass(frozen=True)
class Case:
    case_id: str
    design: str
    channel_error_probability: float
    min_sum_normalization: float
    secret_block_weight: int
    q_block_weight: int

    @property
    def parameters(self) -> dict[str, float | int]:
        return {
            "channel_error_probability": self.channel_error_probability,
            "min_sum_normalization": self.min_sum_normalization,
            "secret_block_weight": self.secret_block_weight,
            "q_block_weight": self.q_block_weight,
        }


DEFAULT_CHANNEL_VALUES = (0.06, 0.12, 0.18)
DEFAULT_NORMALIZATION_VALUES = (0.60, 0.80, 0.90)
DEFAULT_SECRET_WEIGHTS = (2, 3, 4, 5)
# Even weights are never invertible modulo x^31 - 1 because q(1) = 0.
DEFAULT_Q_WEIGHTS = (1, 3, 5)


def _float_values(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("provide at least one value")
    return values


def _int_values(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("provide at least one value")
    return values


def validate_values(
    channel_values: Sequence[float],
    normalization_values: Sequence[float],
    secret_weights: Sequence[int],
    q_weights: Sequence[int],
) -> None:
    if not all(0.0 < value < 0.5 for value in channel_values):
        raise ValueError("channel_error_probability must lie in (0, 0.5)")
    if not all(0.0 < value <= 1.0 for value in normalization_values):
        raise ValueError("min_sum_normalization must lie in (0, 1]")
    if not all(1 <= value <= DEMO.block_size for value in secret_weights):
        raise ValueError(f"secret_block_weight must be between 1 and {DEMO.block_size}")
    if not all(1 <= value <= DEMO.block_size for value in q_weights):
        raise ValueError(f"q_block_weight must be between 1 and {DEMO.block_size}")
    if any(value % 2 == 0 for value in q_weights):
        raise ValueError(
            "q_block_weight must be odd to be invertible modulo x^31 - 1"
        )


def _case_id(prefix: str, values: dict[str, float | int]) -> str:
    encoded = "_".join(
        f"{name}-{str(value).replace('.', 'p')}" for name, value in values.items()
    )
    return f"{prefix}_{encoded}"


def build_cases(
    *,
    design: str,
    channel_values: Sequence[float] = DEFAULT_CHANNEL_VALUES,
    normalization_values: Sequence[float] = DEFAULT_NORMALIZATION_VALUES,
    secret_weights: Sequence[int] = DEFAULT_SECRET_WEIGHTS,
    q_weights: Sequence[int] = DEFAULT_Q_WEIGHTS,
) -> list[Case]:
    validate_values(
        channel_values, normalization_values, secret_weights, q_weights
    )
    base = {
        "channel_error_probability": DEMO.channel_error_probability,
        "min_sum_normalization": DEMO.min_sum_normalization,
        "secret_block_weight": DEMO.secret_block_weight,
        "q_block_weight": DEMO.q_block_weight,
    }
    cases = [Case("control", "control", **base)]

    if design == "oat":
        factors: tuple[tuple[str, Sequence[float | int]], ...] = (
            ("channel_error_probability", channel_values),
            ("min_sum_normalization", normalization_values),
            ("secret_block_weight", secret_weights),
            ("q_block_weight", q_weights),
        )
        for factor, values in factors:
            for value in values:
                if value == base[factor]:
                    continue
                parameters = dict(base)
                parameters[factor] = value
                cases.append(Case(
                    _case_id("oat", {factor: value}),
                    "oat",
                    **parameters,
                ))
        return cases

    if design != "factorial":
        raise ValueError("design must be 'oat' or 'factorial'")

    cases = []
    for channel, normalization, secret_weight, q_weight in product(
        channel_values, normalization_values, secret_weights, q_weights
    ):
        parameters = {
            "channel_error_probability": channel,
            "min_sum_normalization": normalization,
            "secret_block_weight": secret_weight,
            "q_block_weight": q_weight,
        }
        case_id = "control" if parameters == base else _case_id("grid", parameters)
        cases.append(Case(case_id, "factorial", **parameters))
    return cases


def _summary(values: Sequence[float | int]) -> dict[str, float] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    return {
        "mean": mean(numeric),
        "median": median(numeric),
        "std": stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def _wilson(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def _aggregate(case: Case, runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run["status"] == "ok"]
    attempts = [run["attempts"] for run in runs]
    return {
        "case_id": case.case_id,
        "design": case.design,
        "parameters": case.parameters,
        "repetitions": len(runs),
        "successful_signatures": len(successful),
        "failed_signatures": len(runs) - len(successful),
        "sign_success_rate": len(successful) / len(runs),
        "sign_success_rate_wilson_95": _wilson(len(successful), len(runs)),
        "attempts_all_failures_capped": _summary(attempts),
        "attempts_successful_only": _summary([
            run["attempts"] for run in successful
        ]),
        "accepted_attempt_rate": len(successful) / sum(attempts),
        "bp_iterations_successful_only": _summary([
            run["bp_iterations"] for run in successful
        ]),
        "keygen_seconds": _summary([run["keygen_seconds"] for run in runs]),
        "sign_seconds": _summary([run["sign_seconds"] for run in runs]),
        "runs": sorted(runs, key=lambda run: run["repetition"]),
    }


def run_experiment(
    cases: Sequence[Case],
    *,
    repetitions: int,
    seed: int,
    max_sign_attempts: int,
) -> list[dict[str, Any]]:
    runs_by_case: dict[str, list[dict[str, Any]]] = {
        case.case_id: [] for case in cases
    }
    keygen(
        DEMO,
        ShakeRNG.from_int(seed + 10_000_000),
        warm_up_decoder=True,
    )

    for repetition in range(repetitions):
        paired_seed = seed + repetition
        message = f"LEE-CFS-weight-sensitivity-v1:{seed}:{repetition}".encode("ascii")
        ordered_cases = list(cases)
        random.Random(seed + repetition).shuffle(ordered_cases)

        for case in ordered_cases:
            parameters = replace(
                DEMO,
                name=f"{DEMO.name}-WEIGHTS-{case.case_id}",
                **case.parameters,
            )
            start = perf_counter()
            keys = keygen(
                parameters,
                ShakeRNG.from_int(paired_seed),
                warm_up_decoder=False,
            )
            keygen_seconds = perf_counter() - start

            start = perf_counter()
            try:
                signature = sign(
                    message,
                    keys.secret_key,
                    keys.public_key,
                    parameters,
                    ShakeRNG.from_int(paired_seed),
                    max_attempts=max_sign_attempts,
                )
                sign_seconds = perf_counter() - start
                if not verify(message, signature, keys.public_key, parameters):
                    raise RuntimeError("a freshly generated signature was rejected")
                run = {
                    "repetition": repetition,
                    "paired_seed": paired_seed,
                    "status": "ok",
                    "attempts": signature.attempts,
                    "bp_iterations": signature.bp_iterations,
                    "keygen_seconds": keygen_seconds,
                    "sign_seconds": sign_seconds,
                }
            except SigningFailure as error:
                run = {
                    "repetition": repetition,
                    "paired_seed": paired_seed,
                    "status": "failed",
                    "attempts": error.attempts,
                    "bp_iterations": None,
                    "keygen_seconds": keygen_seconds,
                    "sign_seconds": perf_counter() - start,
                }
            runs_by_case[case.case_id].append(run)

    return [_aggregate(case, runs_by_case[case.case_id]) for case in cases]


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fields = (
        "case_id", "design", "channel_error_probability",
        "min_sum_normalization", "secret_block_weight", "q_block_weight",
        "repetitions", "successful_signatures", "failed_signatures",
        "sign_success_rate", "success_ci95_low", "success_ci95_high",
        "mean_attempts_failures_capped", "mean_attempts_successful_only",
        "accepted_attempt_rate", "mean_bp_iterations_successful_only",
        "mean_keygen_seconds", "mean_sign_seconds",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "case_id": result["case_id"],
                "design": result["design"],
                **result["parameters"],
                "repetitions": result["repetitions"],
                "successful_signatures": result["successful_signatures"],
                "failed_signatures": result["failed_signatures"],
                "sign_success_rate": result["sign_success_rate"],
                "success_ci95_low": result["sign_success_rate_wilson_95"]["low"],
                "success_ci95_high": result["sign_success_rate_wilson_95"]["high"],
                "mean_attempts_failures_capped": result[
                    "attempts_all_failures_capped"
                ]["mean"],
                "mean_attempts_successful_only": (
                    result["attempts_successful_only"] or {}
                ).get("mean"),
                "accepted_attempt_rate": result["accepted_attempt_rate"],
                "mean_bp_iterations_successful_only": (
                    result["bp_iterations_successful_only"] or {}
                ).get("mean"),
                "mean_keygen_seconds": result["keygen_seconds"]["mean"],
                "mean_sign_seconds": result["sign_seconds"]["mean"],
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity of QC-LDPC-CFS DEMO to decoder parameters and weights."
        )
    )
    parser.add_argument("--design", choices=("oat", "factorial"), default="oat")
    parser.add_argument("--channel-values", type=_float_values, default=DEFAULT_CHANNEL_VALUES)
    parser.add_argument(
        "--normalization-values",
        type=_float_values,
        default=DEFAULT_NORMALIZATION_VALUES,
    )
    parser.add_argument("--secret-weights", type=_int_values, default=DEFAULT_SECRET_WEIGHTS)
    parser.add_argument("--q-weights", type=_int_values, default=DEFAULT_Q_WEIGHTS)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-sign-attempts", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("sensitivity_cfs_demo_weights.json")
    )
    parser.add_argument(
        "--csv-output", type=Path, default=Path("sensitivity_cfs_demo_weights.csv")
    )
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.max_sign_attempts <= 0:
        parser.error("--max-sign-attempts must be positive")
    try:
        validate_values(
            args.channel_values,
            args.normalization_values,
            args.secret_weights,
            args.q_weights,
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = build_cases(
        design=args.design,
        channel_values=args.channel_values,
        normalization_values=args.normalization_values,
        secret_weights=args.secret_weights,
        q_weights=args.q_weights,
    )
    results = run_experiment(
        cases,
        repetitions=args.repetitions,
        seed=args.seed,
        max_sign_attempts=args.max_sign_attempts,
    )
    document = {
        "experiment": "QC-LDPC-CFS DEMO decoder and weight sensitivity v1",
        "scheme": "qc_ldpc_cfs",
        "base_parameter_set": DEMO.name,
        "design": args.design,
        "number_of_configurations": len(cases),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "max_sign_attempts": args.max_sign_attempts,
        "paired_seed_and_message": True,
        "case_order_randomized_per_repetition": True,
        "grids": {
            "channel_error_probability": args.channel_values,
            "min_sum_normalization": args.normalization_values,
            "secret_block_weight": args.secret_weights,
            "q_block_weight": args.q_weights,
        },
        "security_scope": (
            "secret_block_weight and q_block_weight change the key structure and "
            "would require a new security assessment in a cryptographic parameter "
            "set. DEMO does not provide cryptographic security."
        ),
        "failure_handling": (
            "Failures are included in the success rate and censored at the limit in "
            "attempts_all_failures_capped."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.csv_output, results)
    print(f"Configurations: {len(cases)}")
    print(f"Runs: {len(cases) * args.repetitions}")
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
