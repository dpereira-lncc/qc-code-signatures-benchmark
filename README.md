# Unified signature benchmark

The `benchmarks` package compares the implementations in these directories:

- `hqcs_r_signature`;
- `lm_qcs_python`;
- `lmqcs_python`;
- `qc_ldpc_cfs`;
- `qc_ldpc_cfs_punc`.

The first three implementations provide 128-, 192-, and 256-bit scenarios.
The CFS profiles (`demo`, `original`, and `estimated_128/192/256`) are
experimental and have no validated security equivalence to those levels.
They therefore remain opt-in. The default configuration runs 9 scenarios
with 10 repetitions per scenario.

> **HQCS-R note:** the implemented paper provides only 128-bit parameters.
> The 192- and 256-bit sets used by this benchmark are experimental candidates
> available in this workspace and must not be treated as cryptographically
> validated parameter sets.

> **Security warning:** performance results do not constitute cryptographic
> validation. Every implementation in this repository is intended exclusively
> for research.

## Authors, license, and citation

This implementation was developed by Pablo H. Santos Moreira, Diogo Pereira,
and Fábio Borges. The cryptographic algorithms were not proposed by the
authors of this code; their sources are identified in each package README.

Original code in this repository is distributed under the
[BSD 3-Clause License](LICENSE). Because the associated article has not yet
been published, [CITATION.cff](CITATION.cff) currently provides provisional
metadata for citing the software. It should be updated with the article title,
DOI, and bibliographic details after publication.

## What the benchmark does

For each repetition, the program runs the complete workflow:

1. key generation (`KeyGen`);
2. signing (`Sign`);
3. verification (`Verify`).

Every generated signature is verified. If a freshly generated signature is
rejected, execution stops with an error message.

A warm-up cycle runs before each scenario is measured. It initializes paths
such as Numba and other lazy initialization work, but its timings are excluded
from the statistics.

Each scenario runs in an isolated `spawn` process. Scenario order is shuffled
reproducibly with `--order-seed`, and every scheme uses the same SHAKE256-based
`ShakeRNG` implementation. Timers are maintained by the runner and externally
wrap each `keygen`, `sign`, and `verify` call.

The same message is reused across all repetitions and schemes. By default, it
contains exactly 1024 bits (128 bytes), all of which are the ASCII character
`A`.

For `KeyGen`, `Sign`, and `Verify`, the results report times in seconds with:

- mean;
- median;
- standard deviation;
- minimum;
- maximum.

The output also records total scenario time, the mean number of attempts when
provided by the implementation, and the public-key, secret-key, and signature
sizes.

## Dependencies

Use Python 3.10 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Repository layout

- `benchmarks`: unified benchmark runner;
- `experiments`: reproducible statistical experiments;
- `common`: shared randomness and error types;
- `tests`: automated tests for the published code;
- `scripts`: shell helpers for benchmarking and selective staging;
- each cryptographic implementation remains in its own package directory.

Run the complete published test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Helper scripts

For a reproducible Linux benchmark with 100 repetitions, 1024-bit messages,
and CPU affinity fixed to CPU 0, run:

```bash
./scripts/run_benchmark_1024_bits.sh
```

The generated `benchmark_results_1024_bits_100.json` file is ignored by Git.

When preparing the first commit in a new, empty clone, stage only the files
intended for the article repository with:

```bash
./scripts/stage_article_repository.sh
```

The staging script requires an empty Git index. It includes the published
implementations, statistical experiment code, tests, and documentation, while
excluding local-only implementations, generated results, and figure-generation
files. It does not create a commit or push to a remote.

## Running the benchmark

Run the following command from the workspace root:

```bash
python -m benchmarks
```

The default command runs all 9 scenarios with:

- 10 repetitions per scenario;
- a 1024-bit message made of 128 `A` characters;
- isolated processes;
- reproducibly shuffled order;
- output written to `benchmark_results_10.json`.

The result is written to:

```text
benchmark_results_10.json
```

The JSON file is updated after each scenario finishes. Results from completed
scenarios therefore remain available if a long run is interrupted.

## Options

To set the number of repetitions explicitly:

```bash
python -m benchmarks --repetitions 10
```

To choose another output file:

```bash
python -m benchmarks --output results.json
```

To choose a message length, which must be positive and divisible by 8:

```bash
python -m benchmarks --message-bits 2048
```

Options can be combined:

```bash
python -m benchmarks \
  --repetitions 10 \
  --message-bits 2048 \
  --output results.json
```

To run only one scenario:

```bash
python -m benchmarks --scenario lm_qcs_python:128
```

To run multiple scenarios, repeat the option:

```bash
python -m benchmarks \
  --scenario lm_qcs_python:128 \
  --scenario lmqcs_python:128 \
  --scenario hqcs_r_signature:128
```

Security-level scenarios use `lm_qcs_python`, `lmqcs_python`, or
`hqcs_r_signature` combined with `128`, `192`, or `256`. The opt-in profiles
for both `qc_ldpc_cfs` and `qc_ldpc_cfs_punc` are `demo`, `original`,
`estimated_128`, `estimated_192`, and `estimated_256`. The `estimated_` prefix
indicates only the scale targeted by the experiment, not a cryptographically
validated security level.

To run both demonstration CFS profiles:

```bash
python -m benchmarks \
  --scenario qc_ldpc_cfs:demo \
  --scenario qc_ldpc_cfs_punc:demo
```

For the `original`, `estimated_128`, `estimated_192`, and `estimated_256`
profiles, signing is limited to 100 attempts per repetition. When no signature
is found, the JSON records `status: signing_incomplete`, the success rate, and
the elapsed time without blocking execution. Each scenario also has an
external timeout:

```bash
python -m benchmarks \
  --scenario qc_ldpc_cfs:original \
  --scenario-timeout 30
```

To run both CFS profiles up to the full limit of 10,000 attempts and save the
accumulated time until failure:

```bash
python -m benchmarks \
  --repetitions 1 \
  --scenario qc_ldpc_cfs:original \
  --scenario qc_ldpc_cfs_punc:original \
  --max-sign-attempts 10000 \
  --scenario-timeout 3600 \
  --output benchmark_cfs_full_attempts.json
```

The timeout is applied separately to each process. It must exceed the time
needed to exhaust the attempts; otherwise, the scenario ends with
`status: timeout` before producing the complete measurement. If a signature is
found before the limit, execution succeeds as specified by the algorithm.

For each repetition, `signing_runs` records `status`, `attempts`, and
`elapsed_seconds`. The result also contains `total_sign_attempts`,
`total_sign_seconds`, and `seconds_per_sign_attempt`.

To control how scenario order is shuffled:

```bash
python -m benchmarks --order-seed 20260722
```

To display the program help:

```bash
python -m benchmarks --help
```

## Output structure

The JSON contains environment information, the message used, and one entry for
each selected implementation and parameter-set or profile combination.
Simplified example:

```json
{
  "repetitions_per_scenario": 10,
  "message_bits": 1024,
  "message_utf8": "AAAA...",
  "execution_order": ["hqcs_r_signature:128", "lm_qcs_python:128"],
  "results": [
    {
      "implementation": "lm_qcs_python",
      "parameter_set": "LM-QCS-I",
      "security_bits": 128,
      "repetitions": 10,
      "process_isolated": true,
      "rng": "ShakeRNG-SHAKE256-v1",
      "keygen_seconds": {},
      "sign_seconds": {},
      "verify_seconds": {}
    }
  ]
}
```

JSON files produced before this refactoring use the previous methodology.
Regenerate the results before comparing them with the standardized runner.

## Exporting aggregated benchmark statistics to CSV

Convert the benchmark JSON into a flat CSV table with one row per scenario:

```bash
python -m benchmarks.export_csv \
  benchmark_results_1024_bits_100.json \
  --output benchmark_results_1024_bits_100.csv
```

If `--output` is omitted, the exporter replaces the input suffix with `.csv`.
The CSV includes execution metadata, status, key and signature sizes, signing
success and attempt totals, and the mean, median, standard deviation, minimum,
and maximum times for KeyGen, Sign, and Verify. Statistics unavailable for a
timed-out or incomplete scenario are written as empty cells.

## QC-LDPC-CFS decoder experiments

The scripts below document auxiliary analyses used to investigate the decoder
and signing success rate. Check `--help` before a complete run to adjust the
experiment cost:

| Module | Purpose |
|---|---|
| `experiments.sensitivity_analysis` | general sensitivity analysis described in `SENSITIVITY_ANALYSIS.md` |
| `experiments.sensitivity_cfs_demo_weights` | sensitivity of decoder weights and parameters in the `demo` profile |
| `experiments.cfs_decoder_failure_analysis` | decoder failure and stagnation diagnostics |
| `experiments.cfs_planted_error_experiment` | decoder experiment using planted decodable errors |
| `experiments.cfs_planted_batch_4096` | batch of planted errors over independent keys with block size 4096 |
| `experiments.cfs_decoder_capacity_sweep` | empirical error-correction capacity sweep |
| `experiments.cfs_sign_attempt_instrumentation` | instrumentation of actual candidates produced by `Sign` |

Help and execution examples:

```bash
python -m experiments.sensitivity_analysis --help
python -m experiments.sensitivity_cfs_demo_weights --help
python -m experiments.cfs_decoder_failure_analysis --help
python -m experiments.cfs_planted_error_experiment --help
python -m experiments.cfs_planted_batch_4096 --help
python -m experiments.cfs_decoder_capacity_sweep --help
python -m experiments.cfs_sign_attempt_instrumentation --help
```

By default, the new experiments use deterministic seeds and write JSON/CSV
results in the repository root. These result artifacts are listed in
`.gitignore`: scripts, tests, and documentation should be versioned, while
locally generated experimental results are excluded from the push.
