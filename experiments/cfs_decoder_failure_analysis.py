from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from math import ceil, sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

from qc_ldpc_cfs.core import keygen
from qc_ldpc_cfs.hashing import hash_to_syndrome
from qc_ldpc_cfs.parameters import DEMO
from qc_ldpc_cfs.ring import bits_to_int, int_to_bits
from experiments.sensitivity_cfs_demo_weights import (
    DEFAULT_CHANNEL_VALUES,
    DEFAULT_NORMALIZATION_VALUES,
    DEFAULT_Q_WEIGHTS,
    DEFAULT_SECRET_WEIGHTS,
    Case,
    _float_values,
    _int_values,
    build_cases,
)
from common.rng import ShakeRNG


CATEGORY_NAMES = {
    1: "converged_correct_syndrome",
    2: "reported_converged_but_wrong_syndrome",
    3: "correct_syndrome_but_invalid_weight",
    4: "max_iterations_without_convergence",
    5: "stationary_no_decision_change",
}


@njit(cache=True)
def _diagnostic_min_sum_kernel(
    syndrome_bits: np.ndarray,
    check_ptr: np.ndarray,
    check_edges: np.ndarray,
    variable_ptr: np.ndarray,
    variable_edges: np.ndarray,
    edge_vars: np.ndarray,
    prior: float,
    normalization: float,
    max_iterations: int,
) -> tuple[bool, int, np.ndarray, int, int, int, int]:
    """Instrument Min-Sum without changing the production decoder.

    Return success, iterations, final decision, total changes, longest unchanged
    sequence, final unchanged sequence, and final unsatisfied checks.
    """
    edge_count = edge_vars.size
    variable_count = variable_ptr.size - 1
    check_count = check_ptr.size - 1
    v2c = np.empty(edge_count, dtype=np.float32)
    c2v = np.zeros(edge_count, dtype=np.float32)
    posterior = np.empty(variable_count, dtype=np.float32)
    hard = np.zeros(variable_count, dtype=np.uint8)
    previous = np.zeros(variable_count, dtype=np.uint8)

    for edge in range(edge_count):
        v2c[edge] = prior

    total_changes = 0
    unchanged_streak = 0
    max_unchanged_streak = 0

    for iteration in range(1, max_iterations + 1):
        for check in range(check_count):
            start = check_ptr[check]
            end = check_ptr[check + 1]
            total_sign = -1.0 if syndrome_bits[check] else 1.0
            min1 = np.float32(1.0e30)
            min2 = np.float32(1.0e30)
            min_edge = -1

            for position in range(start, end):
                edge = check_edges[position]
                value = v2c[edge]
                if value < 0.0:
                    total_sign = -total_sign
                    magnitude = -value
                else:
                    magnitude = value
                if magnitude < min1:
                    min2 = min1
                    min1 = magnitude
                    min_edge = edge
                elif magnitude < min2:
                    min2 = magnitude

            if end - start == 1:
                min2 = min1
            for position in range(start, end):
                edge = check_edges[position]
                own_sign = -1.0 if v2c[edge] < 0.0 else 1.0
                magnitude = min2 if edge == min_edge else min1
                c2v[edge] = normalization * total_sign * own_sign * magnitude

        changes = 0
        for variable in range(variable_count):
            total = prior
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            for position in range(start, end):
                total += c2v[variable_edges[position]]
            posterior[variable] = total
            hard[variable] = 1 if total < 0.0 else 0
            if hard[variable] != previous[variable]:
                changes += 1

        total_changes += changes
        if changes == 0:
            unchanged_streak += 1
            if unchanged_streak > max_unchanged_streak:
                max_unchanged_streak = unchanged_streak
        else:
            unchanged_streak = 0
        previous[:] = hard

        unsatisfied = 0
        for check in range(check_count):
            parity = 0
            start = check_ptr[check]
            end = check_ptr[check + 1]
            for position in range(start, end):
                edge = check_edges[position]
                parity ^= int(hard[edge_vars[edge]])
            if parity != int(syndrome_bits[check]):
                unsatisfied += 1
        if unsatisfied == 0:
            return (
                True, iteration, hard, total_changes, max_unchanged_streak,
                unchanged_streak, 0,
            )

        for variable in range(variable_count):
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            total = posterior[variable]
            for position in range(start, end):
                edge = variable_edges[position]
                v2c[edge] = total - c2v[edge]

    final_unsatisfied = 0
    for check in range(check_count):
        parity = 0
        for position in range(check_ptr[check], check_ptr[check + 1]):
            edge = check_edges[position]
            parity ^= int(hard[edge_vars[edge]])
        if parity != int(syndrome_bits[check]):
            final_unsatisfied += 1
    return (
        False, max_iterations, hard, total_changes, max_unchanged_streak,
        unchanged_streak, final_unsatisfied,
    )


@dataclass(frozen=True)
class DiagnosticResult:
    category: int
    iterations: int
    error_weight: int
    weight_limit: int | None
    unsatisfied_checks: int
    total_decision_changes: int
    max_unchanged_streak: int
    final_unchanged_streak: int


def diagnostic_decode(
    decoder: Any,
    syndrome: int,
    *,
    weight_limit: int | None,
    stationary_patience: int,
) -> DiagnosticResult:
    syndrome_bits = int_to_bits(
        syndrome, decoder.parity_check.r
    ).astype(np.uint8, copy=False)
    (
        reported_success,
        iterations,
        hard,
        total_changes,
        max_unchanged_streak,
        final_unchanged_streak,
        kernel_unsatisfied,
    ) = _diagnostic_min_sum_kernel(
        syndrome_bits,
        decoder.check_ptr,
        decoder.check_edges,
        decoder.variable_ptr,
        decoder.variable_edges,
        decoder.edge_vars,
        decoder.prior,
        np.float32(decoder.normalization),
        decoder.max_iterations,
    )
    error = bits_to_int(hard)
    actual = decoder.parity_check.syndrome_int(error)
    independent_correct = actual == syndrome
    independent_unsatisfied = (actual ^ syndrome).bit_count()
    if independent_unsatisfied != kernel_unsatisfied:
        raise RuntimeError("contagens de checks insatisfeitos divergiram")

    error_weight = error.bit_count()
    weight_valid = weight_limit is None or error_weight <= weight_limit
    if reported_success and independent_correct:
        category = 1 if weight_valid else 3
    elif reported_success and not independent_correct:
        category = 2
    elif final_unchanged_streak >= stationary_patience:
        category = 5
    else:
        category = 4
    return DiagnosticResult(
        category=category,
        iterations=int(iterations),
        error_weight=error_weight,
        weight_limit=weight_limit,
        unsatisfied_checks=independent_unsatisfied,
        total_decision_changes=int(total_changes),
        max_unchanged_streak=int(max_unchanged_streak),
        final_unchanged_streak=int(final_unchanged_streak),
    )


def default_block_sizes() -> tuple[int, ...]:
    # 31 + 5k produces 31,...,76; 80 is included as the requested endpoint.
    return tuple(range(31, 77, 5)) + (80,)


def weight_limit_for(parameters: Any, sigma_multiplier: float | None) -> int | None:
    if sigma_multiplier is None:
        return None
    n = parameters.n
    p = parameters.channel_error_probability
    expected = n * p
    deviation = sqrt(n * p * (1.0 - p))
    return min(n, ceil(expected + sigma_multiplier * deviation))


def _matrix_hash(secret_h: Any) -> str:
    return hashlib.sha256(secret_h.serialize()).hexdigest()[:16]


def _optional_float(text: str) -> float | None:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a number or 'none'") from error


def analyze(
    cases: Sequence[Case],
    *,
    block_sizes: Sequence[int],
    matrices_per_case: int,
    syndromes_per_matrix: int,
    seed: int,
    stationary_patience: int,
    weight_sigma: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_rows: list[dict[str, Any]] = []
    ordered_jobs = [
        (block_size, case, matrix_index)
        for block_size in block_sizes
        for case in cases
        for matrix_index in range(matrices_per_case)
    ]
    random.Random(seed).shuffle(ordered_jobs)

    for block_size, case, matrix_index in ordered_jobs:
        matrix_seed = (
            seed
            + block_size * 1_000_000
            + matrix_index * 10_000
        )
        parameters = replace(
            DEMO,
            name=f"{DEMO.name}-DIAG-b{block_size}-{case.case_id}",
            block_size=block_size,
            **case.parameters,
        )
        keys = keygen(
            parameters,
            ShakeRNG.from_int(matrix_seed),
            warm_up_decoder=False,
        )
        decoder = keys.secret_key.decoder
        limit = weight_limit_for(parameters, weight_sigma)
        outcomes: list[DiagnosticResult] = []
        for syndrome_index in range(syndromes_per_matrix):
            message = (
                f"LEE-CFS-decoder-diagnostic-v1:{seed}:{block_size}:"
                f"{matrix_index}:{syndrome_index}"
            ).encode("ascii")
            syndrome = hash_to_syndrome(message, 0, parameters.r)
            outcomes.append(diagnostic_decode(
                decoder,
                syndrome,
                weight_limit=limit,
                stationary_patience=stationary_patience,
            ))

        counts = Counter(outcome.category for outcome in outcomes)
        matrix_rows.append({
            "block_size": block_size,
            "n": parameters.n,
            "case_id": case.case_id,
            "parameters": case.parameters,
            "matrix_index": matrix_index,
            "matrix_seed": matrix_seed,
            "matrix_hash": _matrix_hash(keys.secret_key.secret_parity_check),
            "edge_count": decoder.edge_count,
            "weight_limit": limit,
            "syndromes": syndromes_per_matrix,
            "category_counts": {
                CATEGORY_NAMES[number]: counts[number] for number in CATEGORY_NAMES
            },
            "mean_iterations": mean(outcome.iterations for outcome in outcomes),
            "mean_error_weight": mean(outcome.error_weight for outcome in outcomes),
            "mean_unsatisfied_checks": mean(
                outcome.unsatisfied_checks for outcome in outcomes
            ),
            "mean_total_decision_changes": mean(
                outcome.total_decision_changes for outcome in outcomes
            ),
            "mean_max_unchanged_streak": mean(
                outcome.max_unchanged_streak for outcome in outcomes
            ),
            "mean_final_unchanged_streak": mean(
                outcome.final_unchanged_streak for outcome in outcomes
            ),
        })

    summaries: list[dict[str, Any]] = []
    for block_size in block_sizes:
        for case in cases:
            selected = [
                row for row in matrix_rows
                if row["block_size"] == block_size and row["case_id"] == case.case_id
            ]
            totals = Counter()
            for row in selected:
                totals.update(row["category_counts"])
            total_decodes = sum(totals.values())
            summaries.append({
                "block_size": block_size,
                "n": block_size * DEMO.block_count,
                "case_id": case.case_id,
                "parameters": case.parameters,
                "matrices": len(selected),
                "unique_matrices": len({row["matrix_hash"] for row in selected}),
                "decodes": total_decodes,
                "category_counts": dict(totals),
                "category_rates": {
                    name: totals[name] / total_decodes for name in CATEGORY_NAMES.values()
                },
                "mean_iterations": mean(row["mean_iterations"] for row in selected),
                "mean_error_weight": mean(row["mean_error_weight"] for row in selected),
                "mean_unsatisfied_checks": mean(
                    row["mean_unsatisfied_checks"] for row in selected
                ),
                "mean_total_decision_changes": mean(
                    row["mean_total_decision_changes"] for row in selected
                ),
                "mean_final_unchanged_streak": mean(
                    row["mean_final_unchanged_streak"] for row in selected
                ),
            })
    return summaries, matrix_rows


def _write_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    category_fields = tuple(f"rate_{name}" for name in CATEGORY_NAMES.values())
    fields = (
        "block_size", "n", "case_id", "channel_error_probability",
        "min_sum_normalization", "secret_block_weight", "q_block_weight",
        "matrices", "unique_matrices", "decodes", *category_fields,
        "mean_iterations", "mean_error_weight", "mean_unsatisfied_checks",
        "mean_total_decision_changes", "mean_final_unchanged_streak",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {
                key: summary[key]
                for key in (
                    "block_size", "n", "case_id", "matrices", "unique_matrices",
                    "decodes", "mean_iterations", "mean_error_weight",
                    "mean_unsatisfied_checks", "mean_total_decision_changes",
                    "mean_final_unchanged_streak",
                )
            }
            row.update(summary["parameters"])
            row.update({
                f"rate_{name}": summary["category_rates"][name]
                for name in CATEGORY_NAMES.values()
            })
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics for the five Min-Sum outcomes in QC-LDPC-CFS DEMO."
    )
    parser.add_argument("--design", choices=("oat", "factorial"), default="oat")
    parser.add_argument("--block-sizes", type=_int_values, default=default_block_sizes())
    parser.add_argument("--matrices", type=int, default=20)
    parser.add_argument("--syndromes-per-matrix", type=int, default=20)
    parser.add_argument("--stationary-patience", type=int, default=3)
    parser.add_argument(
        "--weight-sigma",
        type=_optional_float,
        default=3.0,
        help="heuristic limit np + sigma*sqrt(np(1-p)); use 'none' to disable",
    )
    parser.add_argument("--channel-values", type=_float_values, default=DEFAULT_CHANNEL_VALUES)
    parser.add_argument(
        "--normalization-values", type=_float_values,
        default=DEFAULT_NORMALIZATION_VALUES,
    )
    parser.add_argument("--secret-weights", type=_int_values, default=DEFAULT_SECRET_WEIGHTS)
    parser.add_argument("--q-weights", type=_int_values, default=DEFAULT_Q_WEIGHTS)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--output", type=Path, default=Path("cfs_decoder_failure_analysis.json")
    )
    parser.add_argument(
        "--csv-output", type=Path,
        default=Path("cfs_decoder_failure_analysis.csv"),
    )
    args = parser.parse_args(argv)
    if args.matrices <= 0 or args.syndromes_per_matrix <= 0:
        parser.error("--matrices and --syndromes-per-matrix must be positive")
    if args.stationary_patience <= 0:
        parser.error("--stationary-patience must be positive")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if any(value <= 0 for value in args.block_sizes):
        parser.error("all block sizes must be positive")
    if args.weight_sigma is not None and args.weight_sigma < 0:
        parser.error("--weight-sigma cannot be negative")
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
    summaries, matrices = analyze(
        cases,
        block_sizes=args.block_sizes,
        matrices_per_case=args.matrices,
        syndromes_per_matrix=args.syndromes_per_matrix,
        seed=args.seed,
        stationary_patience=args.stationary_patience,
        weight_sigma=args.weight_sigma,
    )
    document = {
        "experiment": "QC-LDPC-CFS DEMO decoder failure diagnostics v1",
        "block_sizes": args.block_sizes,
        "note_block_size_sequence": (
            "31..76 in steps of 5, plus 80 as an explicit endpoint"
        ),
        "design": args.design,
        "matrices_per_block_size_and_case": args.matrices,
        "syndromes_per_matrix": args.syndromes_per_matrix,
        "configurations": len(cases),
        "stationary_definition": (
            f"category 5 when the final {args.stationary_patience} iterations "
            "end without a change in the hard decision"
        ),
        "weight_rule": (
            "no weight limit; category 3 disabled"
            if args.weight_sigma is None
            else (
                "The implemented CFS has no protocol-level weight limit. "
                "Category 3 uses only the diagnostic limit "
                f"np + {args.weight_sigma}*sqrt(np(1-p))."
            )
        ),
        "category_definitions": CATEGORY_NAMES,
        "q_weight_scope": (
            "q_block_weight varies during generation but is not passed to the "
            "decoder; therefore, it should not change categories for the same H."
        ),
        "summaries": summaries,
        "matrices": matrices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.csv_output, summaries)
    print(f"Configurations: {len(cases)}")
    print(f"Tamanhos de bloco: {len(args.block_sizes)}")
    print(f"Matrizes: {len(matrices)}")
    print(f"Decodings: {len(matrices) * args.syndromes_per_matrix}")
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
