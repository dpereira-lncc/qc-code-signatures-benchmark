# QC-LDPC-CFS sensitivity analysis

The `experiments.sensitivity_analysis` module executes the experimental design
using the `DEMO` parameter sets from `qc_ldpc_cfs` and `qc_ldpc_cfs_punc`:

| Phase | Variation | Default design |
|---|---|---|
| Control | none | `DEMO` values |
| A | `channel_error_probability` | 0.5x and 1.5x the control value |
| B | `min_sum_normalization` | 0.6 and 1.0 |
| C | `max_bp_iterations` | 0.5x and 2x the control value |
| D | A + B + C | full 2³ factorial design |

Phases A, B, and C use a *one factor at a time* design: the other two values
remain at their control levels. Phase D estimates main effects and
interactions. In each repetition, every case receives the same key seed and
message. Case order is shuffled reproducibly to reduce temporal bias.

## Running the analysis

Short smoke test:

```bash
python -m experiments.sensitivity_analysis \
  --repetitions 3 \
  --max-sign-attempts 200 \
  --output sensitivity_smoke.json \
  --csv-output sensitivity_smoke.csv
```

Recommended initial run:

```bash
python -m experiments.sensitivity_analysis \
  --repetitions 30 \
  --max-sign-attempts 1000
```

Specific phases or a single scheme can also be selected:

```bash
python -m experiments.sensitivity_analysis \
  --schemes qc_ldpc_cfs_punc \
  --phases control,A,B,C \
  --repetitions 30
```

Levels can be overridden with `--channel-scales`, `--normalization-values`,
and `--iteration-scales`. Each option accepts two comma-separated numbers.

## Interpreting the results

The JSON preserves every run and includes summaries. The CSV contains one row
per configuration. The primary response variables are:

- `sign_success_rate` and its 95% Wilson interval;
- `attempts_all_failures_capped`, which includes failures as right-censored
  observations at the configured limit;
- `attempts_successful_only` and `bp_iterations_successful_only`;
- `accepted_attempt_rate`, which measures acceptance by the complete pipeline
  per attempt and must not be described as a pure BP convergence rate;
- `sign_seconds`;
- `signature_weight_successful_only`, available for the punctured variant;
- `factorial_effects_phase_D`, expressed as the mean at the high level minus
  the mean at the low level. Interactions use the same factorial contrast.

The studied hyperparameters affect decoder behavior rather than structural
parameters. Nevertheless, `DEMO` provides no cryptographic security. This
analysis evaluates implementation behavior and convergence; it neither
supports security claims nor replaces an assessment of timing or failure
leakage.
