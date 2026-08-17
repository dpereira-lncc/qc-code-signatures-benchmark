from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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
TARGET_WEIGHT = 82
DEFAULT_KEYS = 10
DEFAULT_TRIALS_PER_KEY = 100
CSV_FIELDS = (
    "key_id",
    "trial",
    "planted_weight",
    "success",
    "exact_recovery",
    "iterations",
    "final_residual_weight",
)


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


def run_experiment(
    *,
    key_count: int = DEFAULT_KEYS,
    trials_per_key: int = DEFAULT_TRIALS_PER_KEY,
    seed: int = 20260807,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parameters = replace(
        DEMO,
        name=f"{DEMO.name}-PLANTED-BATCH-4096",
        block_size=BLOCK_SIZE,
    )
    if parameters.n != N:
        raise RuntimeError(f"unexpected n: {parameters.n}; expected: {N}")

    records: list[dict[str, Any]] = []
    key_metadata: list[dict[str, Any]] = []
    start_experiment = perf_counter()

    for key_offset in range(key_count):
        key_id = key_offset + 1
        key_seed = seed + key_id * 10_000_000
        start = perf_counter()
        keys = keygen(
            parameters,
            ShakeRNG.from_int(key_seed),
            warm_up_decoder=True,
        )
        keygen_seconds = perf_counter() - start
        secret_h = keys.secret_key.secret_parity_check
        decoder = keys.secret_key.decoder
        matrix_hash = hashlib.sha256(secret_h.serialize()).hexdigest()
        key_metadata.append({
            "key_id": key_id,
            "key_seed": key_seed,
            "secret_h_sha256": matrix_hash,
            "edge_count": decoder.edge_count,
            "keygen_and_warmup_seconds": keygen_seconds,
        })

        for trial_offset in range(trials_per_key):
            trial = trial_offset + 1
            error_seed = (
                seed
                + key_id * 10_000_000
                + trial * 10_000
                + 1
            )
            planted_error = sample_exact_weight_error(
                parameters.n, TARGET_WEIGHT, error_seed
            )
            syndrome = secret_h.syndrome_int(planted_error)

            success, iterations, decoded_error = _decode(decoder, syndrome)
            decoded_syndrome = secret_h.syndrome_int(decoded_error)
            final_residual_weight = (
                decoded_syndrome ^ syndrome
            ).bit_count()
            syndrome_correct = final_residual_weight == 0
            if success != syndrome_correct:
                raise RuntimeError(
                    f"indicador e residual divergiram em key={key_id}, trial={trial}"
                )
            records.append({
                "key_id": key_id,
                "trial": trial,
                "planted_weight": planted_error.bit_count(),
                "success": success,
                "exact_recovery": decoded_error == planted_error,
                "iterations": iterations,
                "final_residual_weight": final_residual_weight,
            })

    if len({item["secret_h_sha256"] for item in key_metadata}) != key_count:
        raise RuntimeError("the keys' H matrices are not all independent")

    elapsed_seconds = perf_counter() - start_experiment
    by_key: list[dict[str, Any]] = []
    for key_id in range(1, key_count + 1):
        selected = [record for record in records if record["key_id"] == key_id]
        successful = [record for record in selected if record["success"]]
        exact = [record for record in selected if record["exact_recovery"]]
        by_key.append({
            "key_id": key_id,
            "trials": len(selected),
            "successes": len(successful),
            "success_rate": len(successful) / len(selected),
            "exact_recoveries": len(exact),
            "exact_recovery_rate": len(exact) / len(selected),
            "mean_iterations": mean(record["iterations"] for record in selected),
            "mean_final_residual_weight": mean(
                record["final_residual_weight"] for record in selected
            ),
        })

    successful = [record for record in records if record["success"]]
    exact = [record for record in records if record["exact_recovery"]]
    summary = {
        "tests": len(records),
        "successes": len(successful),
        "failures": len(records) - len(successful),
        "success_rate": len(successful) / len(records),
        "exact_recoveries": len(exact),
        "exact_recovery_rate": len(exact) / len(records),
        "mean_iterations_all": mean(record["iterations"] for record in records),
        "mean_iterations_successful": (
            mean(record["iterations"] for record in successful)
            if successful else None
        ),
        "mean_final_residual_weight": mean(
            record["final_residual_weight"] for record in records
        ),
        "elapsed_seconds": elapsed_seconds,
        "by_key": by_key,
    }
    return records, key_metadata, summary


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run planted errors of weight 82 over 10 QC-LDPC-CFS keys "
            "independentes com block_size=4096."
        )
    )
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS)
    parser.add_argument("--trials-per-key", type=int, default=DEFAULT_TRIALS_PER_KEY)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("cfs_planted_batch_4096.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("cfs_planted_batch_4096.json"),
    )
    args = parser.parse_args(argv)
    if args.keys <= 0 or args.trials_per_key <= 0:
        parser.error("--keys and --trials-per-key must be positive")
    if args.seed < 0:
        parser.error("--seed cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records, key_metadata, summary = run_experiment(
        key_count=args.keys,
        trials_per_key=args.trials_per_key,
        seed=args.seed,
    )
    document = {
        "experiment": "QC-LDPC-CFS planted batch block_size=4096 v1",
        "block_size": BLOCK_SIZE,
        "n": N,
        "target_weight": TARGET_WEIGHT,
        "base_parameters": asdict(DEMO),
        "key_count": args.keys,
        "trials_per_key": args.trials_per_key,
        "seed": args.seed,
        "keys": key_metadata,
        "summary": summary,
        "records": records,
    }
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.csv_output, records)
    args.json_output.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Testes: {summary['tests']}")
    print(f"Sucessos: {summary['successes']}")
    print(f"Exact recoveries: {summary['exact_recoveries']}")
    print(f"CSV:  {args.csv_output}")
    print(f"JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
