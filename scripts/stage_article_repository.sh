#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPOSITORY_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: run this script inside the new Git repository clone." >&2
    exit 1
fi

if ! git diff --cached --quiet --; then
    echo "Error: the Git index already contains staged changes." >&2
    echo "Unstage them before running this selective staging script." >&2
    exit 1
fi

readonly -a ROOT_FILES=(
    .gitignore
    CITATION.cff
    LICENSE
    README.md
    SENSITIVITY_ANALYSIS.md
    requirements.txt
)

readonly -a PUBLISHED_DIRECTORIES=(
    benchmarks
    common
    experiments
    hqcs_r_signature
    lm_qcs_python
    lmqcs_python
    qc_ldpc_cfs
    qc_ldpc_cfs_punc
    scripts
    tests
)

for path in "${ROOT_FILES[@]}" "${PUBLISHED_DIRECTORIES[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Error: required publication file is missing: $path" >&2
        exit 1
    fi
done

git add -- "${ROOT_FILES[@]}" "${PUBLISHED_DIRECTORIES[@]}"

readonly FORBIDDEN_PATTERN='(^|/)(fuleeca_python|qc_ldpc_2022|old|benchmark_latex)(/|$)|(^|/)(fuleeca\.py|run_fuleeca_flow\.py|generate_benchmark_latex\.py|NLength\.tex)$|\.(json|csv)$'

if git diff --cached --diff-filter=ACMR --name-only | grep -E "$FORBIDDEN_PATTERN"; then
    echo "Error: a local-only implementation, result, or figure generator was staged." >&2
    exit 1
fi

echo "Selected article files staged successfully."
echo
git status --short
