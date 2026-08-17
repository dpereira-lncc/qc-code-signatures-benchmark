from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from math import ceil, sqrt
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

from experiments.cfs_decoder_failure_analysis import (
    CATEGORY_NAMES,
)
from qc_ldpc_cfs.core import keygen
from qc_ldpc_cfs.parameters import DEMO
from qc_ldpc_cfs.ring import bits_to_int, int_to_bits
from common.rng import ShakeRNG


DEFAULT_BLOCK_SIZES = (80, 4096)
DEFAULT_TARGET_WEIGHT_RATIO = 0.005


@njit(cache=True)
def _min_sum_with_residual_trace(
    syndrome_bits: np.ndarray,
    check_ptr: np.ndarray,
    check_edges: np.ndarray,
    variable_ptr: np.ndarray,
    variable_edges: np.ndarray,
    edge_vars: np.ndarray,
    prior: float,
    normalization: float,
    max_iterations: int,
) -> tuple[bool, int, np.ndarray, np.ndarray, np.ndarray]:
    """Run the same Min-Sum kernel and preserve residual weight per iteration."""
    edge_count = edge_vars.size
    variable_count = variable_ptr.size - 1
    check_count = check_ptr.size - 1
    v2c = np.empty(edge_count, dtype=np.float32)
    c2v = np.zeros(edge_count, dtype=np.float32)
    posterior = np.empty(variable_count, dtype=np.float32)
    hard = np.zeros(variable_count, dtype=np.uint8)
    previous = np.zeros(variable_count, dtype=np.uint8)
    residual_trace = np.empty(max_iterations, dtype=np.int32)
    decision_change_trace = np.empty(max_iterations, dtype=np.int32)
    for edge in range(edge_count):
        v2c[edge] = prior

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
        decision_change_trace[iteration - 1] = changes
        previous[:] = hard

        residual_weight = 0
        for check in range(check_count):
            parity = 0
            for position in range(check_ptr[check], check_ptr[check + 1]):
                edge = check_edges[position]
                parity ^= int(hard[edge_vars[edge]])
            if parity != int(syndrome_bits[check]):
                residual_weight += 1
        residual_trace[iteration - 1] = residual_weight
        if residual_weight == 0:
            return (
                True, iteration, hard, residual_trace[:iteration],
                decision_change_trace[:iteration],
            )

        for variable in range(variable_count):
            start = variable_ptr[variable]
            end = variable_ptr[variable + 1]
            total = posterior[variable]
            for position in range(start, end):
                edge = variable_edges[position]
                v2c[edge] = total - c2v[edge]

    return (
        False, max_iterations, hard, residual_trace, decision_change_trace,
    )


def expected_error_weight(
    parameters: Any,
    target_weight_ratio: float = DEFAULT_TARGET_WEIGHT_RATIO,
) -> int:
    """Return the integer weight nearest to the target defined as a ratio of n."""
    return round(parameters.n * target_weight_ratio)


def sample_exact_weight_error(n: int, weight: int, seed: int) -> int:
    positions = ShakeRNG.from_int(seed).sample_positions(n, weight)
    error = 0
    for position in positions:
        error |= 1 << position
    if error.bit_count() != weight:
        raise RuntimeError("the planted error does not have the requested weight")
    return error


def _zero_streaks(changes: Sequence[int]) -> tuple[int, int]:
    maximum = 0
    current = 0
    for value in changes:
        if value == 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum, current


def run_planted_test(
    block_size: int,
    *,
    seed: int,
    target_weight_ratio: float = DEFAULT_TARGET_WEIGHT_RATIO,
) -> dict[str, Any]:
    parameters = replace(
        DEMO,
        name=f"{DEMO.name}-PLANTED-b{block_size}",
        block_size=block_size,
    )
    matrix_seed = seed + block_size * 1_000_000
    error_seed = seed + block_size * 1_000_000 + 1

    start = perf_counter()
    keys = keygen(
        parameters,
        ShakeRNG.from_int(matrix_seed),
        warm_up_decoder=True,
    )
    keygen_seconds = perf_counter() - start
    secret_h = keys.secret_key.secret_parity_check
    decoder = keys.secret_key.decoder

    planted_weight = expected_error_weight(parameters, target_weight_ratio)
    planted_error = sample_exact_weight_error(
        parameters.n, planted_weight, error_seed
    )
    syndrome = secret_h.syndrome_int(planted_error)
    if secret_h.syndrome_int(planted_error) != syndrome:
        raise RuntimeError("failed to validate the planted syndrome")

    syndrome_bits = int_to_bits(
        syndrome, parameters.r
    ).astype(np.uint8, copy=False)
    start = perf_counter()
    (
        reported_success,
        iterations,
        hard,
        residual_trace_array,
        decision_change_trace_array,
    ) = _min_sum_with_residual_trace(
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
    decode_seconds = perf_counter() - start
    residual_trace = [int(value) for value in residual_trace_array]
    decision_change_trace = [
        int(value) for value in decision_change_trace_array
    ]
    total_changes = sum(decision_change_trace)
    max_unchanged_streak, final_unchanged_streak = _zero_streaks(
        decision_change_trace
    )
    kernel_unsatisfied = residual_trace[-1]
    decoded_error = bits_to_int(hard)
    decoded_syndrome = secret_h.syndrome_int(decoded_error)
    syndrome_correct = decoded_syndrome == syndrome
    unsatisfied_checks = (decoded_syndrome ^ syndrome).bit_count()
    if unsatisfied_checks != kernel_unsatisfied:
        raise RuntimeError("independent syndrome check diverged from the kernel")

    weight_limit = planted_weight
    decoded_weight = decoded_error.bit_count()
    weight_valid = decoded_weight <= weight_limit
    if reported_success and syndrome_correct:
        category = 1 if weight_valid else 3
    elif reported_success:
        category = 2
    elif final_unchanged_streak >= 3:
        category = 5
    else:
        category = 4

    # Confirm that the decoder's public path agrees with the instrumented
    # kernel. On failure, DecodeResult deliberately discards the final vector,
    # so only success and iteration count can be compared.
    production = decoder.decode(syndrome)
    production_consistent = (
        production.success == bool(reported_success)
        and production.iterations == int(iterations)
        and (
            not production.success
            or production.error == decoded_error
        )
    )
    if not production_consistent:
        raise RuntimeError("production decoder diverged from the instrumented kernel")

    matrix_bytes = secret_h.serialize()
    return {
        "block_size": block_size,
        "n": parameters.n,
        "parameters": asdict(parameters),
        "matrix_seed": matrix_seed,
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "matrix_bytes": len(matrix_bytes),
        "edge_count": decoder.edge_count,
        "planted": {
            "error_seed": error_seed,
            "target_weight_ratio": target_weight_ratio,
            "expected_real_weight": parameters.n * target_weight_ratio,
            "error_weight": planted_weight,
            "syndrome_has_known_solution": True,
            "syndrome_hex": hex(syndrome),
        },
        "decoded": {
            "reported_success": bool(reported_success),
            "category": category,
            "category_name": CATEGORY_NAMES[category],
            "iterations": int(iterations),
            "syndrome_correct": syndrome_correct,
            "unsatisfied_checks": unsatisfied_checks,
            "error_weight": decoded_weight,
            "target_weight_limit": weight_limit,
            "weight_valid": weight_valid,
            "exact_planted_error_recovered": decoded_error == planted_error,
            "hamming_distance_to_planted_error": (
                decoded_error ^ planted_error
            ).bit_count(),
            "total_decision_changes": int(total_changes),
            "max_unchanged_streak": int(max_unchanged_streak),
            "final_unchanged_streak": int(final_unchanged_streak),
            "production_decoder_consistent": production_consistent,
        },
        "iteration_trace": [
            {
                "iteration": index + 1,
                "residual_weight": residual,
                "decision_changes": decision_change_trace[index],
            }
            for index, residual in enumerate(residual_trace)
        ],
        "timing_seconds": {
            "keygen_and_warmup": keygen_seconds,
            "diagnostic_decode": decode_seconds,
        },
    }


def _block_sizes(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test QC-LDPC-CFS BP with planted errors of weight np."
        )
    )
    parser.add_argument("--block-sizes", type=_block_sizes, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--target-weight-ratio",
        type=float,
        default=DEFAULT_TARGET_WEIGHT_RATIO,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cfs_planted_error_results.json")
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=Path("cfs_planted_error_residual_trace.csv"),
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    if not 0.0 < args.target_weight_ratio <= 1.0:
        parser.error("--target-weight-ratio must lie in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results = [
        run_planted_test(
            block_size,
            seed=args.seed,
            target_weight_ratio=args.target_weight_ratio,
        )
        for block_size in args.block_sizes
    ]
    document = {
        "experiment": "QC-LDPC-CFS planted decodable syndrome test v1",
        "base_parameter_set": DEMO.name,
        "method": (
            "Sample e with weight round(target_weight_ratio*n), compute "
            "s=H_secret*e^T and run "
            "the BP decoder on s. A solution is guaranteed to exist; recovery by "
            "BP is not assumed."
        ),
        "seed": args.seed,
        "target_weight_ratio": args.target_weight_ratio,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with args.trace_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "block_size", "iteration", "residual_weight", "decision_changes"
            ),
        )
        writer.writeheader()
        for result in results:
            for trace in result["iteration_trace"]:
                writer.writerow({
                    "block_size": result["block_size"],
                    **trace,
                })
    for result in results:
        decoded = result["decoded"]
        print(
            f"block_size={result['block_size']} planted_weight="
            f"{result['planted']['error_weight']} success="
            f"{decoded['reported_success']} category={decoded['category']} "
            f"unsatisfied={decoded['unsatisfied_checks']}"
        )
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.trace_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
