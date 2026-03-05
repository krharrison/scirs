#!/usr/bin/env bash
# build_shared_lib.sh — Build the SciRS2 shared library and install it for Julia.
#
# Usage:
#   bash scripts/build_shared_lib.sh [--release | --debug] [--features FEATURES]
#
# Options:
#   --release    Build in release mode (default).
#   --debug      Build in debug mode.
#   --features   Comma-separated extra Cargo features (default: ffi,linalg).
#   --help       Show this help message.
#
# What this script does:
#   1. Detects the host operating system to determine the correct shared library
#      extension (.dylib on macOS, .so on Linux, .dll on Windows).
#   2. Runs `cargo build` for the `scirs2-core` crate with the `ffi` feature
#      (and optionally `linalg` for the linear algebra FFI functions).
#   3. Copies the resulting shared library to `julia/SciRS2/deps/`.
#
# Prerequisites:
#   - Rust toolchain (https://rustup.rs/)
#   - Cargo in $PATH
#   - The workspace root must contain a `Cargo.toml` for the workspace.
#
# Examples:
#   # Default release build with ffi + linalg features:
#   bash scripts/build_shared_lib.sh
#
#   # Debug build:
#   bash scripts/build_shared_lib.sh --debug
#
#   # Release build without linalg (no OxiBLAS dependency):
#   bash scripts/build_shared_lib.sh --features ffi

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

BUILD_MODE="release"
FEATURES="ffi,linalg"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            BUILD_MODE="release"
            shift
            ;;
        --debug)
            BUILD_MODE="debug"
            shift
            ;;
        --features)
            shift
            FEATURES="$1"
            shift
            ;;
        --help|-h)
            sed -n '2,/^$/p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate workspace root (where this script's parent scripts/ dir lives)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "==> Workspace root: ${WORKSPACE_ROOT}"
echo "==> Build mode:     ${BUILD_MODE}"
echo "==> Features:       ${FEATURES}"

# ---------------------------------------------------------------------------
# Detect OS and set library name
# ---------------------------------------------------------------------------

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Darwin*)
        LIB_PREFIX="lib"
        LIB_EXT=".dylib"
        ;;
    Linux*)
        LIB_PREFIX="lib"
        LIB_EXT=".so"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT*)
        LIB_PREFIX=""
        LIB_EXT=".dll"
        ;;
    *)
        echo "WARNING: Unrecognised OS '${OS_NAME}', assuming .so extension." >&2
        LIB_PREFIX="lib"
        LIB_EXT=".so"
        ;;
esac

LIB_NAME="${LIB_PREFIX}scirs2_core${LIB_EXT}"
echo "==> Library name:   ${LIB_NAME}"

# ---------------------------------------------------------------------------
# Run Cargo build
# ---------------------------------------------------------------------------

cd "${WORKSPACE_ROOT}"

CARGO_ARGS=(
    "build"
    "--package" "scirs2-core"
    "--features" "${FEATURES}"
)

if [[ "${BUILD_MODE}" == "release" ]]; then
    CARGO_ARGS+=("--release")
fi

echo ""
echo "==> Running: cargo ${CARGO_ARGS[*]}"
echo ""

cargo "${CARGO_ARGS[@]}"

# ---------------------------------------------------------------------------
# Locate the compiled artifact
# ---------------------------------------------------------------------------

if [[ "${BUILD_MODE}" == "release" ]]; then
    TARGET_DIR="${WORKSPACE_ROOT}/target/release"
else
    TARGET_DIR="${WORKSPACE_ROOT}/target/debug"
fi

SRC_LIB="${TARGET_DIR}/${LIB_NAME}"

if [[ ! -f "${SRC_LIB}" ]]; then
    # On macOS, Cargo sometimes uses a different dylib name format.
    # Try looking for any matching dylib.
    FALLBACK="$(find "${TARGET_DIR}" -maxdepth 1 -name "*scirs2*core*${LIB_EXT}" 2>/dev/null | head -1)"
    if [[ -n "${FALLBACK}" ]]; then
        SRC_LIB="${FALLBACK}"
        echo "WARNING: expected '${LIB_NAME}' not found; using '${SRC_LIB}'" >&2
    else
        echo "ERROR: Shared library not found at '${SRC_LIB}'" >&2
        echo "       Available files in ${TARGET_DIR}:" >&2
        ls "${TARGET_DIR}"/*.* 2>/dev/null || echo "  (none)" >&2
        exit 1
    fi
fi

echo ""
echo "==> Found library: ${SRC_LIB}"

# ---------------------------------------------------------------------------
# Copy to Julia deps directory
# ---------------------------------------------------------------------------

DEST_DIR="${WORKSPACE_ROOT}/julia/SciRS2/deps"
mkdir -p "${DEST_DIR}"

DEST_LIB="${DEST_DIR}/${LIB_NAME}"
cp -v "${SRC_LIB}" "${DEST_LIB}"

# Also create a canonical symlink `libscirs2_core.dylib` (or .so) so that
# the Julia module can find the library regardless of versioning suffixes.
CANON_NAME="${LIB_PREFIX}scirs2_core${LIB_EXT}"
CANON_PATH="${DEST_DIR}/${CANON_NAME}"
if [[ "${DEST_LIB}" != "${CANON_PATH}" ]]; then
    ln -sf "${LIB_NAME}" "${CANON_PATH}"
    echo "==> Symlink: ${CANON_PATH} -> ${LIB_NAME}"
fi

echo ""
echo "==> Build successful!"
echo "    Library installed to: ${DEST_LIB}"
echo ""
echo "==> To run the Julia test suite:"
echo "    julia --project=${WORKSPACE_ROOT}/julia/SciRS2 ${WORKSPACE_ROOT}/julia/SciRS2/test/runtests.jl"
