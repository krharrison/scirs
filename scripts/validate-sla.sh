#!/usr/bin/env bash
# SciRS2 Performance SLA Validation Script
# Runs benchmarks for key crates and reports results for comparison
# against the SLA baselines defined in baselines/performance_sla.toml.
#
# Usage:
#   ./scripts/validate-sla.sh [--crate <crate-name>] [--verbose]
#
# Environment variables:
#   BASELINE_DIR  - Directory containing SLA baseline files (default: ./baselines)
#   CARGO_OPTS    - Additional cargo options (default: empty)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$PROJECT_DIR/baselines}"
SLA_FILE="$BASELINE_DIR/performance_sla.toml"
VERBOSE="${VERBOSE:-false}"
TARGET_CRATE="${1:-all}"

echo "=============================================="
echo " SciRS2 Performance SLA Validation"
echo "=============================================="
echo "Platform:    $(uname -s) $(uname -m)"
echo "Rust:        $(rustc --version 2>/dev/null || echo 'not found')"
echo "Date:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLA file:    $SLA_FILE"
echo "Target:      $TARGET_CRATE"
echo ""

if [ ! -f "$SLA_FILE" ]; then
    echo "WARNING: SLA baseline file not found at $SLA_FILE"
    echo "         Benchmarks will run but cannot be compared against baselines."
    echo ""
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

run_crate_bench() {
    local crate="$1"
    echo "--- Benchmarking: $crate ---"

    # Check if the crate has benchmarks
    local bench_dir="$PROJECT_DIR/$crate/benches"
    if [ ! -d "$bench_dir" ] || [ -z "$(ls -A "$bench_dir" 2>/dev/null)" ]; then
        echo "  (no benchmarks found, skipping)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        echo ""
        return
    fi

    local output
    if output=$(cargo bench -p "$crate" --all-features ${CARGO_OPTS:-} 2>&1); then
        echo "$output" | grep -E "(time:|thrpt:)" | head -100 || echo "  (benchmark completed, no timing lines)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  FAILED: benchmark returned non-zero exit code"
        if [ "$VERBOSE" = "true" ]; then
            echo "$output" | tail -100
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

CRATES=(
    scirs2-linalg
    scirs2-fft
    scirs2-stats
    scirs2-signal
    scirs2-sparse
    scirs2-integrate
    scirs2-interpolate
    scirs2-optimize
)

if [ "$TARGET_CRATE" = "all" ]; then
    for crate in "${CRATES[@]}"; do
        run_crate_bench "$crate"
    done
else
    # Handle --crate flag
    if [ "$TARGET_CRATE" = "--crate" ] && [ -n "${2:-}" ]; then
        run_crate_bench "$2"
    else
        run_crate_bench "$TARGET_CRATE"
    fi
fi

echo "=============================================="
echo " SLA Validation Summary"
echo "=============================================="
echo "  Passed:  $PASS_COUNT"
echo "  Failed:  $FAIL_COUNT"
echo "  Skipped: $SKIP_COUNT"
echo ""
echo "Compare detailed results against: $SLA_FILE"
echo "=============================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
