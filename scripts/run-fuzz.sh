#!/usr/bin/env bash
# run-fuzz.sh — Run a cargo-fuzz target for a SciRS2 crate.
#
# Usage:
#   ./scripts/run-fuzz.sh [TARGET] [CRATE] [EXTRA_FUZZ_ARGS...]
#
# Defaults:
#   TARGET = fuzz_csv
#   CRATE  = scirs2-io
#
# Examples:
#   ./scripts/run-fuzz.sh fuzz_csv scirs2-io
#   ./scripts/run-fuzz.sh fuzz_json scirs2-io -- -max_len=8192 -timeout=30
#   ./scripts/run-fuzz.sh fuzz_bpe  scirs2-text
#   ./scripts/run-fuzz.sh fuzz_tokenizer scirs2-text -- -jobs=4 -workers=4
#
# Prerequisites:
#   cargo install cargo-fuzz
#
# The fuzzer requires a nightly Rust compiler:
#   rustup toolchain install nightly
#   rustup override set nightly   (inside the crate directory, or use +nightly)
#
# Corpus and crash artefacts are stored under:
#   <CRATE>/fuzz/corpus/<TARGET>/
#   <CRATE>/fuzz/artifacts/<TARGET>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET="${1:-fuzz_csv}"
CRATE="${2:-scirs2-io}"
shift 2 2>/dev/null || true           # consume the first two args; ignore if absent

CRATE_DIR="${REPO_ROOT}/${CRATE}"

if [[ ! -d "${CRATE_DIR}" ]]; then
    echo "ERROR: crate directory not found: ${CRATE_DIR}" >&2
    exit 1
fi

if [[ ! -d "${CRATE_DIR}/fuzz" ]]; then
    echo "ERROR: no fuzz/ directory in ${CRATE_DIR}" >&2
    echo "       Supported crates: scirs2-io, scirs2-text" >&2
    exit 1
fi

# Validate that the requested target exists.
TARGET_FILE="${CRATE_DIR}/fuzz/fuzz_targets/${TARGET}.rs"
if [[ ! -f "${TARGET_FILE}" ]]; then
    echo "ERROR: fuzz target '${TARGET}' not found at ${TARGET_FILE}" >&2
    echo ""
    echo "Available targets in ${CRATE}/fuzz/:" >&2
    ls "${CRATE_DIR}/fuzz/fuzz_targets/"*.rs 2>/dev/null \
        | xargs -n1 basename -s .rs \
        | sed 's/^/  /' >&2
    exit 1
fi

# Default libFuzzer flags; override via extra args after '--'.
MAX_LEN="${FUZZ_MAX_LEN:-4096}"
TIMEOUT="${FUZZ_TIMEOUT:-10}"
MAX_TOTAL_TIME="${FUZZ_MAX_TOTAL_TIME:-0}"    # 0 = run indefinitely

echo "=== SciRS2 cargo-fuzz runner ==="
echo "  Crate  : ${CRATE}"
echo "  Target : ${TARGET}"
echo "  Max len: ${MAX_LEN} bytes"
echo "  Timeout: ${TIMEOUT}s per input"
echo "  Dir    : ${CRATE_DIR}"
echo ""

cd "${CRATE_DIR}"

# Use +nightly to ensure the correct toolchain without changing the global
# override.  cargo-fuzz requires nightly.
RUSTUP_TOOLCHAIN="${FUZZ_TOOLCHAIN:-nightly}"

cargo "+${RUSTUP_TOOLCHAIN}" fuzz run "${TARGET}" \
    -- \
    -max_len="${MAX_LEN}" \
    -timeout="${TIMEOUT}" \
    ${MAX_TOTAL_TIME:+-max_total_time="${MAX_TOTAL_TIME}"} \
    "$@"
