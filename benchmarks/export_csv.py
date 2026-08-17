"""Export aggregated benchmark JSON results as a flat CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STATISTIC_NAMES = ("mean", "median", "std", "min", "max")
TIMED_OPERATIONS = ("keygen", "sign", "verify")
DOCUMENT_FIELDS = (
    "started_at_utc",
    "finished_at_utc",
    "python",
    "platform",
    "message_bits",
    "order_seed",
    "scenario_timeout_seconds",
    "max_sign_attempts_override",
)
RESULT_FIELDS = (
    "implementation",
    "variant",
    "parameter_set",
    "security_bits",
    "status",
    "repetitions",
    "process_isolated",
    "rng",
    "timeout_seconds",
    "wall_seconds",
    "successful_signatures",
    "failed_signatures",
    "sign_success_rate",
    "mean_attempts",
    "total_sign_attempts",
    "total_sign_seconds",
    "seconds_per_sign_attempt",
    "sign_attempt_limit",
    "public_key_bytes",
    "secret_key_bytes",
    "signature_bytes",
)
STATISTIC_FIELDS = tuple(
    f"{operation}_seconds_{statistic}"
    for operation in TIMED_OPERATIONS
    for statistic in STATISTIC_NAMES
)
CSV_FIELDS = (
    "execution_index",
    *DOCUMENT_FIELDS,
    *RESULT_FIELDS,
    *STATISTIC_FIELDS,
    "signing_errors",
)


def _statistics(
    result: Mapping[str, Any], operation: str
) -> Mapping[str, Any] | None:
    value = result.get(f"{operation}_seconds")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{operation}_seconds must be an object or null")
    return value


def flatten_result(
    document: Mapping[str, Any],
    result: Mapping[str, Any],
    execution_index: int,
) -> dict[str, Any]:
    """Flatten one aggregated scenario result into a CSV-compatible row."""
    row: dict[str, Any] = {"execution_index": execution_index}
    row.update({field: document.get(field) for field in DOCUMENT_FIELDS})
    row.update({field: result.get(field) for field in RESULT_FIELDS})

    for operation in TIMED_OPERATIONS:
        statistics = _statistics(result, operation)
        for statistic in STATISTIC_NAMES:
            field = f"{operation}_seconds_{statistic}"
            row[field] = None if statistics is None else statistics.get(statistic)

    signing_errors = result.get("signing_errors", [])
    if signing_errors is None:
        signing_errors = []
    if not isinstance(signing_errors, list):
        raise ValueError("signing_errors must be an array or null")
    row["signing_errors"] = " | ".join(str(error) for error in signing_errors)
    return row


def rows_from_document(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one flattened row for each scenario in a benchmark document."""
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark document must contain a results array")

    rows: list[dict[str, Any]] = []
    for execution_index, result in enumerate(results, start=1):
        if not isinstance(result, Mapping):
            raise ValueError(
                f"results[{execution_index - 1}] must be an object"
            )
        rows.append(flatten_result(document, result, execution_index))
    return rows


def export_benchmark_csv(input_path: Path, output_path: Path) -> int:
    """Read benchmark JSON, write aggregated CSV atomically, and return row count."""
    with input_path.open("r", encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, Mapping):
        raise ValueError("benchmark JSON root must be an object")

    rows = rows_from_document(document)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    return len(rows)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".csv")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export aggregated signature benchmark statistics from JSON to CSV."
        )
    )
    parser.add_argument("input", type=Path, help="Benchmark JSON input file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output file; defaults to the input name with a .csv suffix.",
    )
    args = parser.parse_args(argv)
    output = args.output or _default_output_path(args.input)

    try:
        row_count = export_benchmark_csv(args.input, output)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported {row_count} aggregated benchmark rows to {output}")


if __name__ == "__main__":
    main()
