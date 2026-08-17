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
from typing import Any, Callable, Sequence

from common.rng import ShakeRNG
from common.errors import SigningFailure

from qc_ldpc_cfs.core import (
    keygen as cfs_keygen,
    sign as cfs_sign,
    verify as cfs_verify,
)
from qc_ldpc_cfs.parameters import DEMO as CFS_DEMO
from qc_ldpc_cfs_punc.core import (
    keygen as punc_keygen,
    sign as punc_sign,
    verify as punc_verify,
)
from qc_ldpc_cfs_punc.parameters import DEMO as PUNC_DEMO


PHASES = ("control", "A", "B", "C", "D")


@dataclass(frozen=True)
class Scheme:
    name: str
    base_parameters: Any
    keygen: Callable[..., Any]
    sign: Callable[..., Any]
    verify: Callable[..., bool]


@dataclass(frozen=True)
class Case:
    phase: str
    case_id: str
    channel_error_probability: float
    min_sum_normalization: float
    max_bp_iterations: int
    levels: tuple[int, int, int] | None = None


SCHEMES = {
    "qc_ldpc_cfs": Scheme(
        "qc_ldpc_cfs", CFS_DEMO, cfs_keygen, cfs_sign, cfs_verify
    ),
    "qc_ldpc_cfs_punc": Scheme(
        "qc_ldpc_cfs_punc",
        PUNC_DEMO,
        punc_keygen,
        punc_sign,
        punc_verify,
    ),
}


def _two_floats(text: str) -> tuple[float, float]:
    try:
        values = tuple(float(item.strip()) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use two comma-separated numbers") from error
    if len(values) != 2:
        raise argparse.ArgumentTypeError("use exactly two numbers")
    return values  # type: ignore[return-value]


def _phases(text: str) -> tuple[str, ...]:
    normalized = tuple(item.strip() for item in text.split(")") if item.strip())
    # Accept the natural comma-separated spelling; the split above also makes
    # an unmatched ')' fail with a useful message below.
    if len(normalized) == 1:
        normalized = tuple(item.strip() for item in text.split(",") if item.strip())
    invalid = [item for item in normalized if item not in PHASES]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid phases: {', '.join(invalid)}")
    return normalized


def build_cases(
    base: Any,
    *,
    channel_scales: tuple[float, float] = (0.5, 1.5),
    normalization_values: tuple[float, float] = (0.6, 1.0),
    iteration_scales: tuple[float, float] = (0.5, 2.0),
    phases: Sequence[str] = PHASES,
) -> list[Case]:
    channel_values = tuple(base.channel_error_probability * x for x in channel_scales)
    iteration_values = tuple(
        max(1, round(base.max_bp_iterations * x)) for x in iteration_scales
    )
    if not all(0.0 < value < 0.5 for value in channel_values):
        raise ValueError("channel_error_probability must remain in (0, 0.5)")
    if not all(0.0 < value <= 1.0 for value in normalization_values):
        raise ValueError("min_sum_normalization must remain in (0, 1]")
    if channel_values[0] == channel_values[1]:
        raise ValueError("the two channel_error_probability levels must differ")
    if normalization_values[0] == normalization_values[1]:
        raise ValueError("the two min_sum_normalization levels must differ")
    if iteration_values[0] == iteration_values[1]:
        raise ValueError("the two max_bp_iterations levels must differ")

    cases: list[Case] = []
    baseline = (
        base.channel_error_probability,
        base.min_sum_normalization,
        base.max_bp_iterations,
    )
    if "control" in phases:
        cases.append(Case("control", "control", *baseline))
    if "A" in phases:
        for label, value in zip(("low", "high"), channel_values):
            cases.append(Case("A", f"A_channel_{label}", value, baseline[1], baseline[2]))
    if "B" in phases:
        for label, value in zip(("low", "high"), normalization_values):
            cases.append(Case("B", f"B_normalization_{label}", baseline[0], value, baseline[2]))
    if "C" in phases:
        for label, value in zip(("low", "high"), iteration_values):
            cases.append(Case("C", f"C_iterations_{label}", baseline[0], baseline[1], value))
    if "D" in phases:
        for channel_level, norm_level, iteration_level in product((-1, 1), repeat=3):
            channel = channel_values[0 if channel_level < 0 else 1]
            normalization = normalization_values[0 if norm_level < 0 else 1]
            iterations = iteration_values[0 if iteration_level < 0 else 1]
            suffix = "".join("L" if level < 0 else "H" for level in (
                channel_level, norm_level, iteration_level
            ))
            cases.append(Case(
                "D",
                f"D_{suffix}",
                channel,
                normalization,
                iterations,
                (channel_level, norm_level, iteration_level),
            ))
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
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def _aggregate(case: Case, runs: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [run for run in runs if run["status"] == "ok"]
    attempts = [run["attempts"] for run in runs]
    result: dict[str, Any] = {
        "phase": case.phase,
        "case_id": case.case_id,
        "parameters": {
            "channel_error_probability": case.channel_error_probability,
            "min_sum_normalization": case.min_sum_normalization,
            "max_bp_iterations": case.max_bp_iterations,
        },
        "factor_levels": case.levels,
        "repetitions": len(runs),
        "successful_signatures": len(successful),
        "failed_signatures": len(runs) - len(successful),
        "sign_success_rate": len(successful) / len(runs),
        "sign_success_rate_wilson_95": _wilson(len(successful), len(runs)),
        # Failure observations equal the attempt limit and are right-censored.
        "attempts_all_failures_capped": _summary(attempts),
        "attempts_successful_only": _summary([run["attempts"] for run in successful]),
        "accepted_attempt_rate": len(successful) / sum(attempts),
        "bp_iterations_successful_only": _summary([
            run["bp_iterations"] for run in successful
        ]),
        "keygen_seconds": _summary([run["keygen_seconds"] for run in runs]),
        "sign_seconds": _summary([run["sign_seconds"] for run in runs]),
        "signature_weight_successful_only": _summary([
            run["signature_weight"]
            for run in successful
            if run["signature_weight"] is not None
        ]),
        "runs": sorted(runs, key=lambda run: run["repetition"]),
    }
    return result


def _factorial_effects(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    factorial = [result for result in results if result["phase"] == "D"]
    if len(factorial) != 8:
        return None
    factor_names = ("channel", "normalization", "iterations")
    responses = {
        "sign_success_rate": lambda item: item["sign_success_rate"],
        "mean_attempts_failures_capped": lambda item: item[
            "attempts_all_failures_capped"
        ]["mean"],
        "mean_sign_seconds": lambda item: item["sign_seconds"]["mean"],
    }
    effects: dict[str, Any] = {}
    terms = (
        (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
    )
    for response_name, getter in responses.items():
        response_effects: dict[str, float] = {}
        for term in terms:
            positive: list[float] = []
            negative: list[float] = []
            for item in factorial:
                levels = item["factor_levels"]
                sign = 1
                for index in term:
                    sign *= levels[index]
                (positive if sign > 0 else negative).append(float(getter(item)))
            label = "×".join(factor_names[index] for index in term)
            response_effects[label] = mean(positive) - mean(negative)
        effects[response_name] = response_effects
    return effects


def run_scheme(
    scheme: Scheme,
    cases: Sequence[Case],
    *,
    repetitions: int,
    seed: int,
    max_sign_attempts: int,
) -> dict[str, Any]:
    runs_by_case: dict[str, list[dict[str, Any]]] = {
        case.case_id: [] for case in cases
    }

    # Compile the decoder kernel before collecting timing observations.
    scheme.keygen(
        scheme.base_parameters,
        ShakeRNG.from_int(seed + 10_000_000),
        warm_up_decoder=True,
    )

    for repetition in range(repetitions):
        paired_seed = seed + repetition
        message = f"LEE-sensitivity-v1:{seed}:{repetition}".encode("ascii")
        ordered_cases = list(cases)
        random.Random(seed + repetition).shuffle(ordered_cases)

        for case in ordered_cases:
            parameters = replace(
                scheme.base_parameters,
                name=f"{scheme.base_parameters.name}-SENS-{case.case_id}",
                channel_error_probability=case.channel_error_probability,
                min_sum_normalization=case.min_sum_normalization,
                max_bp_iterations=case.max_bp_iterations,
            )
            start = perf_counter()
            keys = scheme.keygen(
                parameters,
                ShakeRNG.from_int(paired_seed),
                warm_up_decoder=False,
            )
            keygen_seconds = perf_counter() - start

            start = perf_counter()
            try:
                signature = scheme.sign(
                    message,
                    keys.secret_key,
                    keys.public_key,
                    parameters,
                    ShakeRNG.from_int(paired_seed),
                    max_attempts=max_sign_attempts,
                )
                sign_seconds = perf_counter() - start
                if not scheme.verify(message, signature, keys.public_key, parameters):
                    raise RuntimeError("a freshly generated signature was rejected")
                run = {
                    "repetition": repetition,
                    "paired_seed": paired_seed,
                    "status": "ok",
                    "attempts": signature.attempts,
                    "bp_iterations": signature.bp_iterations,
                    "signature_weight": getattr(signature, "weight", None),
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
                    "signature_weight": None,
                    "keygen_seconds": keygen_seconds,
                    "sign_seconds": perf_counter() - start,
                }
            runs_by_case[case.case_id].append(run)

    results = [_aggregate(case, runs_by_case[case.case_id]) for case in cases]
    return {
        "scheme": scheme.name,
        "base_parameter_set": scheme.base_parameters.name,
        "results": results,
        "factorial_effects_phase_D": _factorial_effects(results),
    }


def _write_csv(path: Path, analyses: Sequence[dict[str, Any]]) -> None:
    fields = (
        "scheme", "phase", "case_id", "channel_error_probability",
        "min_sum_normalization", "max_bp_iterations", "repetitions",
        "successful_signatures", "failed_signatures", "sign_success_rate",
        "success_ci95_low", "success_ci95_high", "mean_attempts_failures_capped",
        "mean_attempts_successful_only", "accepted_attempt_rate",
        "mean_bp_iterations_successful_only", "mean_sign_seconds",
        "mean_signature_weight_successful_only",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for analysis in analyses:
            for result in analysis["results"]:
                parameters = result["parameters"]
                writer.writerow({
                    "scheme": analysis["scheme"],
                    "phase": result["phase"],
                    "case_id": result["case_id"],
                    **parameters,
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
                    "mean_sign_seconds": result["sign_seconds"]["mean"],
                    "mean_signature_weight_successful_only": (
                        result["signature_weight_successful_only"] or {}
                    ).get("mean"),
                })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired sensitivity analysis of the QC-LDPC-CFS DEMO decoders."
    )
    parser.add_argument(
        "--schemes", nargs="+", choices=tuple(SCHEMES), default=list(SCHEMES)
    )
    parser.add_argument("--phases", type=_phases, default=PHASES)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-sign-attempts", type=int, default=1000)
    parser.add_argument("--channel-scales", type=_two_floats, default=(0.5, 1.5))
    parser.add_argument(
        "--normalization-values", type=_two_floats, default=(0.6, 1.0)
    )
    parser.add_argument("--iteration-scales", type=_two_floats, default=(0.5, 2.0))
    parser.add_argument("--output", type=Path, default=Path("sensitivity_results.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("sensitivity_results.csv"))
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if args.max_sign_attempts <= 0:
        parser.error("--max-sign-attempts must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    analyses = []
    for scheme_name in args.schemes:
        scheme = SCHEMES[scheme_name]
        cases = build_cases(
            scheme.base_parameters,
            channel_scales=args.channel_scales,
            normalization_values=args.normalization_values,
            iteration_scales=args.iteration_scales,
            phases=args.phases,
        )
        analyses.append(run_scheme(
            scheme,
            cases,
            repetitions=args.repetitions,
            seed=args.seed,
            max_sign_attempts=args.max_sign_attempts,
        ))

    document = {
        "experiment": "QC-LDPC-CFS decoder sensitivity v1",
        "design": {
            "parameter_set": "DEMO",
            "paired_by_key_seed_and_message": True,
            "case_order_randomized_per_repetition": True,
            "phases": {
                "control": "unchanged DEMO configuration",
                "A": "channel_error_probability isolada",
                "B": "min_sum_normalization isolada",
                "C": "max_bp_iterations isolado",
                "D": "full 2^3 factorial design for effects and interactions",
            },
            "security_scope": (
                "Only decoder hyperparameters were varied. DEMO does not provide "
                "cryptographic security, and the results do not support a security "
                "claim."
            ),
            "repetitions": args.repetitions,
            "seed": args.seed,
            "max_sign_attempts": args.max_sign_attempts,
            "channel_scales": args.channel_scales,
            "normalization_values": args.normalization_values,
            "iteration_scales": args.iteration_scales,
            "failure_handling": (
                "Failures are included in the success rate; in "
                "attempts_all_failures_capped, they are observations censored at "
                "the attempt limit."
            ),
        },
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.csv_output, analyses)
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
