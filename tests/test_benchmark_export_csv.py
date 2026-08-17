from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.export_csv import CSV_FIELDS, export_benchmark_csv, rows_from_document


def _statistics(base: float) -> dict[str, float]:
    return {
        "mean": base,
        "median": base + 1.0,
        "std": base + 2.0,
        "min": base + 3.0,
        "max": base + 4.0,
    }


class BenchmarkCSVExportTests(unittest.TestCase):
    def test_flattens_aggregated_statistics_and_metadata(self):
        document = {
            "started_at_utc": "2026-08-14T12:00:00+00:00",
            "message_bits": 1024,
            "order_seed": 7,
            "results": [
                {
                    "implementation": "example",
                    "variant": "128",
                    "parameter_set": "EXAMPLE-I",
                    "status": "ok",
                    "repetitions": 10,
                    "keygen_seconds": _statistics(1.0),
                    "sign_seconds": _statistics(10.0),
                    "verify_seconds": _statistics(20.0),
                    "signing_errors": [],
                }
            ],
        }

        rows = rows_from_document(document)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_index"], 1)
        self.assertEqual(rows[0]["message_bits"], 1024)
        self.assertEqual(rows[0]["implementation"], "example")
        self.assertEqual(rows[0]["keygen_seconds_mean"], 1.0)
        self.assertEqual(rows[0]["sign_seconds_std"], 12.0)
        self.assertEqual(rows[0]["verify_seconds_max"], 24.0)

    def test_timeout_result_has_blank_statistics(self):
        rows = rows_from_document({
            "results": [{
                "implementation": "example",
                "status": "timeout",
                "keygen_seconds": None,
                "sign_seconds": None,
                "verify_seconds": None,
            }]
        })

        for field in (
            "keygen_seconds_mean",
            "sign_seconds_median",
            "verify_seconds_max",
        ):
            self.assertIsNone(rows[0][field])

    def test_writes_one_csv_row_per_scenario(self):
        document = {
            "results": [
                {
                    "implementation": "first",
                    "keygen_seconds": _statistics(1.0),
                    "sign_seconds": _statistics(2.0),
                    "verify_seconds": _statistics(3.0),
                },
                {
                    "implementation": "second",
                    "keygen_seconds": None,
                    "sign_seconds": None,
                    "verify_seconds": None,
                    "signing_errors": ["bounded failure"],
                },
            ]
        }
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "benchmark.json"
            output_path = Path(directory) / "benchmark.csv"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            row_count = export_benchmark_csv(input_path, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(row_count, 2)
        self.assertEqual(tuple(rows[0]), CSV_FIELDS)
        self.assertEqual(rows[0]["implementation"], "first")
        self.assertEqual(rows[1]["implementation"], "second")
        self.assertEqual(rows[1]["signing_errors"], "bounded failure")

    def test_rejects_missing_results_array(self):
        with self.assertRaisesRegex(ValueError, "results array"):
            rows_from_document({})


if __name__ == "__main__":
    unittest.main()
