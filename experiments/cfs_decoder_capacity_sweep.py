from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from experiments.cfs_planted_error_experiment import sample_exact_weight_error
from qc_ldpc_cfs.core import keygen
from qc_ldpc_cfs.decoder import _min_sum_kernel
from qc_ldpc_cfs.parameters import DEMO
from qc_ldpc_cfs.ring import bits_to_int, int_to_bits
from common.rng import ShakeRNG


BLOCK_SIZE = 4096
N = 16384
DEFAULT_KEY_COUNT = 10
DEFAULT_TRIALS_PER_KEY = 100
DEFAULT_COARSE_RATIOS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
RAW_FIELDS = (
    "stage", "target_ratio", "target_weight", "key_id", "trial",
    "success", "exact_recovery", "iterations", "final_residual_weight",
)
SUMMARY_FIELDS = (
    "stage", "target_weight", "target_ratio", "tests", "syndromes_resolved",
    "syndrome_success_rate", "exact_recoveries", "exact_recovery_rate",
    "mean_iterations_all", "mean_iterations_successful",
    "mean_final_residual_weight",
)


def _parameters() -> Any:
    parameters = replace(
        DEMO,
        name=f"{DEMO.name}-CAPACITY-SWEEP-4096",
        block_size=BLOCK_SIZE,
    )
    if parameters.n != N:
        raise RuntimeError(f"unexpected n: {parameters.n}")
    return parameters


def _key_seed(seed: int, key_id: int) -> int:
    return seed + key_id * 10_000_000


def _error_seed(seed: int, key_id: int, weight: int, trial: int) -> int:
    return seed + key_id * 10**12 + weight * 10**6 + trial


def _decode(decoder: Any, syndrome: int) -> tuple[bool, int, int]:
    syndrome_bits = int_to_bits(
        syndrome, decoder.parity_check.r
    ).astype(np.uint8, copy=False)
    success, iterations, hard = _min_sum_kernel(
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
    return bool(success), int(iterations), bits_to_int(hard)


def _run_key_job(
    task: tuple[int, int, tuple[tuple[str, float, int], ...], int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed, key_id, points, trials_per_key = task
    parameters = _parameters()
    key_seed = _key_seed(seed, key_id)
    keys = keygen(
        parameters,
        ShakeRNG.from_int(key_seed),
        warm_up_decoder=True,
    )
    secret_h = keys.secret_key.secret_parity_check
    records: list[dict[str, Any]] = []
    for stage, target_ratio, target_weight in points:
        for trial in range(1, trials_per_key + 1):
            planted_error = sample_exact_weight_error(
                parameters.n,
                target_weight,
                _error_seed(seed, key_id, target_weight, trial),
            )
            syndrome = secret_h.syndrome_int(planted_error)
            success, iterations, decoded_error = _decode(
                keys.secret_key.decoder, syndrome
            )
            decoded_syndrome = secret_h.syndrome_int(decoded_error)
            residual_weight = (decoded_syndrome ^ syndrome).bit_count()
            if success != (residual_weight == 0):
                raise RuntimeError(
                    f"success/residual divergiram: key={key_id}, trial={trial}"
                )
            records.append({
                "stage": stage,
                "target_ratio": target_ratio,
                "target_weight": target_weight,
                "key_id": key_id,
                "trial": trial,
                "success": success,
                "exact_recovery": decoded_error == planted_error,
                "iterations": iterations,
                "final_residual_weight": residual_weight,
            })
    metadata = {
        "key_id": key_id,
        "key_seed": key_seed,
        "secret_h_sha256": hashlib.sha256(secret_h.serialize()).hexdigest(),
        "edge_count": keys.secret_key.decoder.edge_count,
    }
    return records, metadata


def _points(stage: str, ratios: Sequence[float]) -> tuple[tuple[str, float, int], ...]:
    seen_weights: set[int] = set()
    result = []
    for ratio in sorted(ratios):
        weight = round(N * ratio)
        if weight <= 0 or weight in seen_weights:
            continue
        seen_weights.add(weight)
        result.append((stage, ratio, weight))
    return tuple(result)


def run_points(
    points: tuple[tuple[str, float, int], ...],
    *,
    key_count: int,
    trials_per_key: int,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = tuple(
        (seed, key_id, points, trials_per_key)
        for key_id in range(1, key_count + 1)
    )
    if workers == 1:
        outputs = map(_run_key_job, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        outputs = executor.map(_run_key_job, tasks)
    try:
        records: list[dict[str, Any]] = []
        keys: list[dict[str, Any]] = []
        for index, (key_records, metadata) in enumerate(outputs, start=1):
            records.extend(key_records)
            keys.append(metadata)
            print(
                f"keys_complete={index}/{key_count} stage={points[0][0]}",
                flush=True,
            )
    finally:
        if workers != 1:
            executor.shutdown()
    if len({item["secret_h_sha256"] for item in keys}) != key_count:
        raise RuntimeError("the H matrices are not all distinct")
    return records, keys


def summarize(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted({
        (record["stage"], record["target_weight"], record["target_ratio"])
        for record in records
    })
    summaries = []
    for stage, weight, ratio in groups:
        selected = [
            record for record in records
            if record["stage"] == stage and record["target_weight"] == weight
        ]
        successful = [record for record in selected if record["success"]]
        exact = [record for record in selected if record["exact_recovery"]]
        summaries.append({
            "stage": stage,
            "target_weight": weight,
            "target_ratio": ratio,
            "tests": len(selected),
            "syndromes_resolved": len(successful),
            "syndrome_success_rate": len(successful) / len(selected),
            "exact_recoveries": len(exact),
            "exact_recovery_rate": len(exact) / len(selected),
            "mean_iterations_all": mean(r["iterations"] for r in selected),
            "mean_iterations_successful": (
                mean(r["iterations"] for r in successful) if successful else None
            ),
            "mean_final_residual_weight": mean(
                r["final_residual_weight"] for r in selected
            ),
        })
    return summaries


def locate_transition(coarse: Sequence[dict[str, Any]]) -> tuple[float, float]:
    ordered = sorted(coarse, key=lambda item: item["target_ratio"])
    for left, right in zip(ordered, ordered[1:]):
        if (
            left["syndrome_success_rate"] >= 0.9
            and right["syndrome_success_rate"] <= 0.1
        ):
            return left["target_ratio"], right["target_ratio"]
    left, right = max(
        zip(ordered, ordered[1:]),
        key=lambda pair: pair[0]["syndrome_success_rate"]
        - pair[1]["syndrome_success_rate"],
    )
    return left["target_ratio"], right["target_ratio"]


def fine_ratios(low: float, high: float, internal_points: int) -> tuple[float, ...]:
    step = (high - low) / (internal_points + 1)
    return tuple(
        round(low + step * index, 12)
        for index in range(1, internal_points + 1)
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive sweep of empirical QC-LDPC-CFS BP capacity."
    )
    parser.add_argument("--keys", type=int, default=DEFAULT_KEY_COUNT)
    parser.add_argument("--trials-per-key", type=int, default=DEFAULT_TRIALS_PER_KEY)
    parser.add_argument("--coarse-ratios", default=",".join(map(str, DEFAULT_COARSE_RATIOS)))
    parser.add_argument("--fine-points", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--raw-csv", type=Path, default=Path("cfs_capacity_sweep_raw.csv"))
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("cfs_capacity_sweep_summary.csv")
    )
    parser.add_argument("--json-output", type=Path, default=Path("cfs_capacity_sweep.json"))
    args = parser.parse_args(argv)
    try:
        args.coarse_ratios = tuple(
            float(item.strip()) for item in args.coarse_ratios.split(",") if item.strip()
        )
    except ValueError as error:
        parser.error(f"invalid --coarse-ratios: {error}")
    if args.keys <= 0 or args.trials_per_key <= 0 or args.workers <= 0:
        parser.error("keys, trials-per-key, and workers must be positive")
    if args.fine_points < 0:
        parser.error("--fine-points cannot be negative")
    if len(args.coarse_ratios) < 2 or not all(0 < x <= 1 for x in args.coarse_ratios):
        parser.error("provide at least two ratios in (0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = perf_counter()
    coarse_points = _points("coarse", args.coarse_ratios)
    coarse_records, coarse_keys = run_points(
        coarse_points,
        key_count=args.keys,
        trials_per_key=args.trials_per_key,
        seed=args.seed,
        workers=args.workers,
    )
    coarse_summary = summarize(coarse_records)
    low, high = locate_transition(coarse_summary)
    selected_fine_ratios = fine_ratios(low, high, args.fine_points)
    fine_points_spec = _points("fine", selected_fine_ratios)
    fine_records: list[dict[str, Any]] = []
    fine_keys = coarse_keys
    if fine_points_spec:
        fine_records, fine_keys = run_points(
            fine_points_spec,
            key_count=args.keys,
            trials_per_key=args.trials_per_key,
            seed=args.seed,
            workers=args.workers,
        )
    records = coarse_records + fine_records
    summaries = summarize(records)
    if [k["secret_h_sha256"] for k in coarse_keys] != [
        k["secret_h_sha256"] for k in fine_keys
    ]:
        raise RuntimeError("the fine sweep did not reuse the same H matrices")
    _write_csv(args.raw_csv, RAW_FIELDS, records)
    _write_csv(args.summary_csv, SUMMARY_FIELDS, summaries)
    document = {
        "experiment": "QC-LDPC-CFS empirical decoder capacity sweep v1",
        "parameters": asdict(_parameters()),
        "key_count": args.keys,
        "trials_per_key": args.trials_per_key,
        "tests_per_point": args.keys * args.trials_per_key,
        "seed": args.seed,
        "coarse_ratios": args.coarse_ratios,
        "transition_bracket": {"low": low, "high": high},
        "fine_ratios": selected_fine_ratios,
        "keys": coarse_keys,
        "summaries": summaries,
        "elapsed_seconds": perf_counter() - started,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Transition bracket: {low:.6f} .. {high:.6f}")
    print(f"Raw CSV:     {args.raw_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"JSON:        {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
