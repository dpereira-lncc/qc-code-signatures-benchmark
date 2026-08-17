from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from qc_ldpc_cfs.core import Signature, _apply_block_diagonal, keygen, verify
from qc_ldpc_cfs.hashing import hash_to_syndrome
from qc_ldpc_cfs.parameters import DEMO
from qc_ldpc_cfs.ring import cyclic_mul
from common.rng import ShakeRNG


BLOCK_SIZE = 4096
N = 16384
TARGET_WEIGHT = 82
DEFAULT_ATTEMPTS = 10_000
CSV_FIELDS = (
    "counter",
    "bp_residual_zero",
    "bp_iterations",
    "reached_max_iterations",
    "decoded_weight",
    "valid_weight_decoded_vector",
    "signature_weight",
    "verification_valid_signature",
    "attempt_seconds",
)
_WORKER_KEYS: Any | None = None
_WORKER_PARAMETERS: Any | None = None


def _parameters() -> Any:
    return replace(
        DEMO,
        name=f"{DEMO.name}-SIGN-INSTRUMENTED-4096",
        block_size=BLOCK_SIZE,
    )


def _evaluate_candidate(
    keys: Any,
    parameters: Any,
    counter: int,
    message: bytes,
) -> dict[str, Any]:
    start_attempt = perf_counter()
    syndrome = hash_to_syndrome(message, counter, parameters.r)
    transformed_syndrome = cyclic_mul(
        keys.secret_key.s_inverse, syndrome, parameters.block_size
    )
    decoded = keys.secret_key.decoder.decode(transformed_syndrome)
    reached_max = (
        not decoded.success
        and decoded.iterations == parameters.max_bp_iterations
    )
    decoded_weight: int | None = None
    valid_weight = False
    signature_weight: int | None = None
    verification_valid = False
    if decoded.success:
        decoded_weight = decoded.error.bit_count()
        valid_weight = decoded_weight <= TARGET_WEIGHT
        z = _apply_block_diagonal(
            decoded.error,
            keys.secret_key.q_inverses,
            parameters.block_size,
        )
        signature_weight = z.bit_count()
        signature = Signature(
            z=z,
            counter=counter,
            attempts=counter + 1,
            bp_iterations=decoded.iterations,
        )
        verification_valid = verify(
            message, signature, keys.public_key, parameters
        )
    return {
        "counter": counter,
        "bp_residual_zero": decoded.success,
        "bp_iterations": decoded.iterations,
        "reached_max_iterations": reached_max,
        "decoded_weight": decoded_weight,
        "valid_weight_decoded_vector": valid_weight,
        "signature_weight": signature_weight,
        "verification_valid_signature": verification_valid,
        "attempt_seconds": perf_counter() - start_attempt,
    }


def _initialize_worker(key_seed: int) -> None:
    global _WORKER_KEYS, _WORKER_PARAMETERS
    _WORKER_PARAMETERS = _parameters()
    _WORKER_KEYS = keygen(
        _WORKER_PARAMETERS,
        ShakeRNG.from_int(key_seed),
        warm_up_decoder=False,
    )


def _worker_candidate(task: tuple[int, bytes]) -> dict[str, Any]:
    if _WORKER_KEYS is None or _WORKER_PARAMETERS is None:
        raise RuntimeError("worker not initialized")
    counter, message = task
    return _evaluate_candidate(
        _WORKER_KEYS, _WORKER_PARAMETERS, counter, message
    )


def run_instrumented_sign(
    *,
    attempts: int,
    key_seed: int,
    message: bytes,
    csv_output: Path,
    progress_every: int = 100,
    workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameters = _parameters()
    if parameters.n != N:
        raise RuntimeError(f"unexpected n: {parameters.n}")

    start = perf_counter()
    keys = keygen(
        parameters,
        ShakeRNG.from_int(key_seed),
        warm_up_decoder=True,
    )
    keygen_seconds = perf_counter() - start
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    iteration_histogram: Counter[int] = Counter()
    start_all = perf_counter()

    executor: ProcessPoolExecutor | None = None
    if workers == 1:
        iterator = (
            _evaluate_candidate(keys, parameters, counter, message)
            for counter in range(attempts)
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(key_seed,),
        )
        iterator = executor.map(
            _worker_candidate,
            ((counter, message) for counter in range(attempts)),
            chunksize=5,
        )
    try:
        with csv_output.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for index, record in enumerate(iterator, start=1):
                iteration_histogram[record["bp_iterations"]] += 1
                records.append(record)
                writer.writerow(record)
                if progress_every and index % progress_every == 0:
                    output.flush()
                    elapsed = perf_counter() - start_all
                    print(
                        f"progress={index}/{attempts} "
                        f"bp_zero={sum(r['bp_residual_zero'] for r in records)} "
                        f"elapsed_seconds={elapsed:.1f}",
                        flush=True,
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    elapsed_seconds = perf_counter() - start_all
    bp_zero = [record for record in records if record["bp_residual_zero"]]
    reached_max = [record for record in records if record["reached_max_iterations"]]
    valid_weight = [
        record for record in records if record["valid_weight_decoded_vector"]
    ]
    verification_valid = [
        record for record in records if record["verification_valid_signature"]
    ]
    valid_signatures = [
        record for record in records
        if record["valid_weight_decoded_vector"]
        and record["verification_valid_signature"]
    ]
    summary = {
        "total_candidate_syndromes": len(records),
        "bp_residual_zero": len(bp_zero),
        "bp_reached_max_iterations": len(reached_max),
        "bp_other_failures": len(records) - len(bp_zero) - len(reached_max),
        "valid_weight_decoded_vectors": len(valid_weight),
        "verification_valid_signatures": len(verification_valid),
        "valid_weight_and_verification_valid_signatures": len(valid_signatures),
        "target_weight_limit": TARGET_WEIGHT,
        "iteration_histogram": dict(sorted(iteration_histogram.items())),
        "mean_attempt_seconds": mean(
            record["attempt_seconds"] for record in records
        ),
        "elapsed_seconds": elapsed_seconds,
        "keygen_and_warmup_seconds": keygen_seconds,
        "workers": workers,
    }
    return summary, records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Instrumenta candidatos reais do Sign QC-LDPC-CFS em block_size=4096."
        )
    )
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--key-seed", type=int, default=20260807)
    parser.add_argument(
        "--message",
        default="LEE QC-LDPC CFS real Sign candidate experiment",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("cfs_sign_attempts_10000.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("cfs_sign_attempts_10000.json"),
    )
    args = parser.parse_args(argv)
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    if args.key_seed < 0:
        parser.error("--key-seed cannot be negative")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary, records = run_instrumented_sign(
        attempts=args.attempts,
        key_seed=args.key_seed,
        message=args.message.encode("utf-8"),
        csv_output=args.csv_output,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    parameters = replace(DEMO, block_size=BLOCK_SIZE)
    document = {
        "experiment": "QC-LDPC-CFS 10000 real Sign candidates v1",
        "method": (
            "Run the inner Sign loop for every counter without stopping at the "
            "first success. Each syndrome comes from hash_to_syndrome and passes "
            "through the S^-1 transformation before BP."
        ),
        "parameters": asdict(parameters),
        "target_weight": TARGET_WEIGHT,
        "key_seed": args.key_seed,
        "message": args.message,
        "workers": args.workers,
        "summary": summary,
        "records": records,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"CSV:  {args.csv_output}")
    print(f"JSON: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
