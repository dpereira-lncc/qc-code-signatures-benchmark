#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly MESSAGE_1024_BITS="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

if (( ${#MESSAGE_1024_BITS} * 8 != 1024 )); then
    echo "Error: the configured message does not contain exactly 1024 bits." >&2
    exit 1
fi

cd -- "$REPOSITORY_ROOT"

exec taskset --cpu-list 0 \
    python3 -m benchmarks \
    --repetitions 100 \
    --message-bits 1024 \
    --output benchmark_results_1024_bits_100.json
