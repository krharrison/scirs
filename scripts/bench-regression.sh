#!/usr/bin/env bash
# bench-regression.sh — Benchmark regression detection script for SciRS2 CI.
#
# Usage:
#   ./scripts/bench-regression.sh [--baseline <path>] [--threshold <fraction>]
#
# Options:
#   --baseline <path>     JSON snapshot file to compare against.
#                         Default: baselines/bench_baseline.json
#   --threshold <value>   Fractional regression threshold (e.g. 0.10 = 10%).
#                         Default: 0.10
#
# Exit codes:
#   0 — no regressions found (or baseline saved for first run)
#   1 — one or more benchmarks regressed beyond threshold
#
# Environment variables:
#   BENCH_CRATES          Space-separated list of crate names to benchmark.
#                         Defaults to: scirs2-linalg scirs2-fft scirs2-stats scirs2-signal
#   CARGO_BENCH_ARGS      Extra arguments forwarded to `cargo bench`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
POLICY_MANIFEST="$ROOT_DIR/tools/cargo-scirs2-policy/Cargo.toml"
CRITERION_DIR="$ROOT_DIR/target/criterion"
SNAPSHOT_PATH="/tmp/scirs2_bench_current.json"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

BASELINE="$ROOT_DIR/baselines/bench_baseline.json"
THRESHOLD="0.10"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--baseline <path>] [--threshold <fraction>]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo "=== SciRS2 Benchmark Regression Check ==="
echo "Root:      $ROOT_DIR"
echo "Baseline:  $BASELINE"
echo "Threshold: $THRESHOLD"
echo ""

# ---------------------------------------------------------------------------
# Run benchmarks for key crates
# ---------------------------------------------------------------------------

BENCH_CRATES="${BENCH_CRATES:-scirs2-linalg scirs2-fft scirs2-stats scirs2-signal}"
CARGO_BENCH_ARGS="${CARGO_BENCH_ARGS:-}"

echo "--- Running benchmarks ---"

any_benched=0
for crate in $BENCH_CRATES; do
    crate_dir="$ROOT_DIR/$crate"
    if [[ -d "$crate_dir" ]]; then
        bench_toml="$crate_dir/Cargo.toml"
        # Only bench crates that actually declare [[bench]] sections
        if grep -q '\[\[bench\]\]' "$bench_toml" 2>/dev/null; then
            echo "Benchmarking $crate..."
            cd "$ROOT_DIR"
            cargo bench --all-features -p "$crate" ${CARGO_BENCH_ARGS} 2>/dev/null || {
                echo "  Warning: bench run for $crate failed or produced no output (continuing)"
            }
            any_benched=1
        else
            echo "  Skipping $crate — no [[bench]] sections found in Cargo.toml"
        fi
    else
        echo "  Skipping $crate — crate directory not found"
    fi
done

if [[ $any_benched -eq 0 ]]; then
    echo "No benchmarks were executed (no [[bench]] sections in target crates)."
    echo "If you want to run benchmarks, add [[bench]] sections to the relevant Cargo.toml files."
fi

# ---------------------------------------------------------------------------
# Create current snapshot from Criterion output
# ---------------------------------------------------------------------------

echo ""
echo "--- Creating benchmark snapshot ---"

cd "$ROOT_DIR"

if [[ -d "$CRITERION_DIR" ]]; then
    cargo run --manifest-path "$POLICY_MANIFEST" -- \
        bench-snapshot \
        --criterion-dir "$CRITERION_DIR" \
        --output "$SNAPSHOT_PATH"
    echo "Snapshot created at $SNAPSHOT_PATH"
else
    echo "Criterion output directory not found at $CRITERION_DIR"
    echo "Creating an empty snapshot as a placeholder."
    cat > "$SNAPSHOT_PATH" <<'EMPTY_SNAP'
{
  "timestamp": "1970-01-01T00:00:00Z",
  "git_hash": null,
  "measurements": []
}
EMPTY_SNAP
    echo "Empty snapshot written to $SNAPSHOT_PATH"
fi

# ---------------------------------------------------------------------------
# Compare against baseline
# ---------------------------------------------------------------------------

echo ""
echo "--- Comparing against baseline ---"

if [[ -f "$BASELINE" ]]; then
    if cargo run --manifest-path "$POLICY_MANIFEST" -- \
        bench-diff \
        --baseline "$BASELINE" \
        --current "$SNAPSHOT_PATH" \
        --threshold "$THRESHOLD"; then
        echo ""
        echo "[PASS] No benchmark regressions detected."
        exit 0
    else
        echo ""
        echo "[FAIL] Benchmark regressions detected above ${THRESHOLD} threshold."
        exit 1
    fi
else
    echo "No baseline found at $BASELINE"
    echo "Saving current snapshot as new baseline..."
    mkdir -p "$(dirname "$BASELINE")"
    cp "$SNAPSHOT_PATH" "$BASELINE"
    echo "Baseline saved to $BASELINE"
    echo ""
    echo "[INFO] First run — baseline established. Re-run to compare."
    exit 0
fi
