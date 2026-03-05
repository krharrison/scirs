#!/usr/bin/env bash
# check_policy.sh — SciRS2 Policy Compliance Checker
#
# Enforces the SciRS2 Dependency Abstraction Policy defined in SCIRS2_POLICY.md.
# Non-core crates must not import rand, ndarray, or ndarray_rand directly.
# All such functionality must be accessed through scirs2_core abstractions.
#
# Usage:
#   bash scripts/check_policy.sh [--no-color] [--strict] [--help]
#
# Options:
#   --no-color    Disable ANSI color output (for non-terminal environments).
#   --strict      Also check benches/ and integration-tests as violations.
#   --help        Show this help message and exit.
#
# Exit codes:
#   0   No policy violations found.
#   1   One or more policy violations detected.
#
# Policy rules enforced:
#   RULE-01  No `use rand::` in non-core crates
#   RULE-02  No `use ndarray::` in non-core crates
#   RULE-03  No `use ndarray_rand::` in non-core crates
#   RULE-04  No `use scirs2_autograd::ndarray` (use scirs2_core::ndarray)
#   RULE-05  No `extern crate rand/ndarray` in non-core crates
#
# Exemptions — these paths are allowed to use the external crates directly:
#   scirs2-core/**      The one crate that re-exports rand/ndarray for the ecosystem.
#                       This includes its src/, tests/, examples/, and benches/.
#   scirs2-numpy/**     SciRS2 fork of rust-numpy; needs ndarray as a first-class dep.
#   */target/*          Compiled artifacts (never source).
#   */examples_disabled/* Disabled examples not part of the build graph.
#   *.backup.*          Backup directories created by scirs2-policy-refactor.sh.
#
# See: SCIRS2_POLICY.md for full policy documentation.

set -euo pipefail

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Policy version (should match SCIRS2_POLICY.md)
POLICY_VERSION="3.0.0"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

USE_COLOR=true
STRICT_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-color)
            USE_COLOR=false
            shift
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --help|-h)
            # Print the header comment block (lines 3 to first non-comment line)
            awk 'NR>2 && /^[^#]/ { exit } NR>2 { sub(/^#[ ]?/, ""); print }' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Color setup
# ---------------------------------------------------------------------------

if [[ "${USE_COLOR}" == "true" ]] && [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    NC='\033[0m'  # No Color / Reset
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    BOLD=''
    DIM=''
    NC=''
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log_header() {
    echo -e "${BOLD}${CYAN}$1${NC}"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC}  $1"
}

log_violation() {
    echo -e "${RED}[VIOLATION]${NC} $1"
}

log_dim() {
    echo -e "${DIM}$1${NC}"
}

# ---------------------------------------------------------------------------
# Build the grep exclude arguments
#
# We skip:
#   - /target/          compiled artifacts
#   - /examples_disabled/ disabled examples not in the build graph
#   - .git              version control internals
# ---------------------------------------------------------------------------

build_grep_excludes() {
    echo "--exclude-dir=target"
    echo "--exclude-dir=examples_disabled"
    echo "--exclude-dir=.git"
}

# ---------------------------------------------------------------------------
# Path exemption helper
#
# Returns 0 (true) if the given workspace-relative path is exempt from the
# specified rule; 1 (false) otherwise.
#
# Args:
#   $1  workspace-relative path  (e.g. "scirs2-core/tests/foo.rs")
#   $2  rule id                  (e.g. "RULE-01")
# ---------------------------------------------------------------------------

is_exempt_path() {
    local rel_path="$1"
    local rule_id="$2"

    # Backup directories created by the refactor tool are always exempt.
    case "${rel_path}" in
        *.backup.*/*) return 0 ;;
    esac

    case "${rule_id}" in
        RULE-01|RULE-02|RULE-05)
            # scirs2-core is entirely exempt for rand and ndarray:
            # its src/, tests/, examples/, and benches/ all legitimately
            # use these crates (they are re-exported from here for the rest
            # of the ecosystem).
            # scirs2-numpy is a direct fork of rust-numpy that depends on
            # ndarray natively.
            case "${rel_path}" in
                scirs2-core/*) return 0 ;;
                scirs2-numpy/*) return 0 ;;
            esac
            ;;
        RULE-03)
            # scirs2-core is exempt for ndarray_rand (it re-exports it).
            # scirs2-numpy doesn't use ndarray_rand, but exempt anyway.
            case "${rel_path}" in
                scirs2-core/*) return 0 ;;
                scirs2-numpy/*) return 0 ;;
            esac
            ;;
        RULE-04)
            # scirs2-autograd itself defines the ndarray re-export;
            # only callers outside the crate are flagged.
            case "${rel_path}" in
                scirs2-autograd/*) return 0 ;;
            esac
            ;;
    esac

    return 1
}

# ---------------------------------------------------------------------------
# Violation tracking
# ---------------------------------------------------------------------------

TOTAL_VIOLATIONS=0
VIOLATION_FILES=()

# Record a single violation and increment counter.
# Args: rule_id  location  description
record_violation() {
    local rule_id="$1"
    local location="$2"
    local description="$3"

    TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + 1))
    log_violation "${BOLD}${rule_id}${NC}${RED} ${location}${NC}"
    echo -e "         ${DIM}${description}${NC}"
}

# ---------------------------------------------------------------------------
# Check helpers
#
# Each check_rule_* function searches the workspace for a specific pattern,
# filters out exempt paths, and records any remaining hits.
# ---------------------------------------------------------------------------

# Run grep and emit path:line pairs.
# Args: pattern  search_root  [additional grep args...]
grep_violations() {
    local pattern="$1"
    local search_root="$2"
    shift 2

    # Build the array of exclude args
    local excludes=()
    while IFS= read -r ex; do
        excludes+=("$ex")
    done < <(build_grep_excludes)

    grep -rn \
        --include="*.rs" \
        "${excludes[@]}" \
        "$@" \
        -E "${pattern}" \
        "${search_root}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Helper to track unique files with violations
# ---------------------------------------------------------------------------

record_violation_file() {
    local path="$1"
    local already_recorded=false
    for f in "${VIOLATION_FILES[@]+"${VIOLATION_FILES[@]}"}"; do
        if [[ "${f}" == "${path}" ]]; then
            already_recorded=true
            break
        fi
    done
    if [[ "${already_recorded}" == "false" ]]; then
        VIOLATION_FILES+=("${path}")
    fi
}

# ---------------------------------------------------------------------------
# Generic rule checker
#
# Args:
#   $1  rule_id       (e.g. "RULE-01")
#   $2  rule_title    (human-readable header)
#   $3  pattern       (ERE regex for grep -E)
#   $4  description   (what was found)
#   $5  suggestion    (how to fix)
# ---------------------------------------------------------------------------

run_rule_check() {
    local rule_id="$1"
    local rule_title="$2"
    local pattern="$3"
    local description="$4"
    local suggestion="$5"
    local violations_found=0

    log_header "${rule_id}: ${rule_title}..."

    while IFS= read -r hit; do
        [[ -z "${hit}" ]] && continue

        local filepath="${hit%%:*}"
        local rest="${hit#*:}"
        local lineno="${rest%%:*}"

        local rel_path="${filepath#${WORKSPACE_ROOT}/}"

        # Apply rule-specific exemptions.
        if is_exempt_path "${rel_path}" "${rule_id}"; then
            continue
        fi

        record_violation "${rule_id}" "${rel_path}:${lineno}" "${description}"
        record_violation_file "${rel_path}"
        violations_found=$((violations_found + 1))
    done < <(grep_violations "${pattern}" "${WORKSPACE_ROOT}")

    if [[ "${violations_found}" -eq 0 ]]; then
        log_ok "Rule ${rule_id} — no violations found."
    fi
}

# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

check_rule_01() {
    run_rule_check \
        "RULE-01" \
        "Checking for direct 'use rand::' imports" \
        '^\s*use rand::' \
        "Direct 'use rand::' found — use 'scirs2_core::random' instead" \
        "Replace: use rand::Foo  →  use scirs2_core::random::Foo"
}

check_rule_02() {
    run_rule_check \
        "RULE-02" \
        "Checking for direct 'use ndarray::' imports" \
        '^\s*use ndarray::' \
        "Direct 'use ndarray::' found — use 'scirs2_core::ndarray' instead" \
        "Replace: use ndarray::Foo  →  use scirs2_core::ndarray::Foo"
}

check_rule_03() {
    run_rule_check \
        "RULE-03" \
        "Checking for direct 'use ndarray_rand::' imports" \
        '^\s*use ndarray_rand::' \
        "Direct 'use ndarray_rand::' found — use scirs2_core ndarray array feature" \
        "Replace: use ndarray_rand::Foo  →  use scirs2_core::ndarray::Foo"
}

check_rule_04() {
    run_rule_check \
        "RULE-04" \
        "Checking for 'use scirs2_autograd::ndarray'" \
        '^\s*use scirs2_autograd::ndarray' \
        "'use scirs2_autograd::ndarray' found — this re-export may cause type mismatches" \
        "Replace: use scirs2_autograd::ndarray  →  use scirs2_core::ndarray"
}

check_rule_05() {
    run_rule_check \
        "RULE-05" \
        "Checking for 'extern crate rand/ndarray/ndarray_rand'" \
        '^\s*extern crate (rand|ndarray|ndarray_rand)\b' \
        "'extern crate' for rand/ndarray found — use scirs2_core abstractions instead" \
        "Remove extern crate declaration; import from scirs2_core"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_header "SciRS2 Policy Check Summary  (policy v${POLICY_VERSION})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    if [[ "${TOTAL_VIOLATIONS}" -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}  ALL POLICY CHECKS PASSED${NC}"
        echo -e "  ${GREEN}No violations found.${NC} The workspace follows the SciRS2 dependency policy."
        echo ""
        echo -e "  ${DIM}Checked rules:${NC}"
        echo -e "  ${DIM}  RULE-01  No direct 'use rand::' in non-core crates${NC}"
        echo -e "  ${DIM}  RULE-02  No direct 'use ndarray::' in non-core crates${NC}"
        echo -e "  ${DIM}  RULE-03  No direct 'use ndarray_rand::' in non-core crates${NC}"
        echo -e "  ${DIM}  RULE-04  No 'use scirs2_autograd::ndarray' (use scirs2_core::ndarray)${NC}"
        echo -e "  ${DIM}  RULE-05  No 'extern crate rand/ndarray' in non-core crates${NC}"
    else
        echo -e "${RED}${BOLD}  POLICY VIOLATIONS FOUND: ${TOTAL_VIOLATIONS}${NC}"
        echo ""

        local unique_count="${#VIOLATION_FILES[@]}"
        echo -e "  ${RED}Affected files: ${unique_count}${NC}"
        echo ""

        if [[ "${unique_count}" -gt 0 ]]; then
            echo -e "  ${BOLD}Files to fix:${NC}"
            for f in "${VIOLATION_FILES[@]}"; do
                echo -e "    ${RED}${f}${NC}"
            done
        fi

        echo ""
        echo -e "  ${YELLOW}How to fix:${NC}"
        echo -e "  ${DIM}  Replace 'use rand::*' with 'use scirs2_core::random::*'${NC}"
        echo -e "  ${DIM}  Replace 'use ndarray::*' with 'use scirs2_core::ndarray::*'${NC}"
        echo -e "  ${DIM}  Replace 'use ndarray_rand::*' with 'use scirs2_core::ndarray::*'${NC}"
        echo -e "  ${DIM}  Replace 'use scirs2_autograd::ndarray' with 'use scirs2_core::ndarray'${NC}"
        echo ""
        echo -e "  ${DIM}Use the automated refactor tool for bulk fixes:${NC}"
        echo -e "  ${DIM}  bash scripts/scirs2-policy-refactor.sh --dry-run <crate-path>${NC}"
        echo -e "  ${DIM}  bash scripts/scirs2-policy-refactor.sh --backup --verify <crate-path>${NC}"
        echo ""
        echo -e "  ${DIM}For more details see: SCIRS2_POLICY.md${NC}"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    cd "${WORKSPACE_ROOT}"

    echo ""
    log_header "SciRS2 Policy Compliance Checker  (policy v${POLICY_VERSION})"
    log_dim "Workspace: ${WORKSPACE_ROOT}"
    log_dim "Strict mode: ${STRICT_MODE}"
    echo ""

    check_rule_01
    echo ""
    check_rule_02
    echo ""
    check_rule_03
    echo ""
    check_rule_04
    echo ""
    check_rule_05

    print_summary

    if [[ "${TOTAL_VIOLATIONS}" -gt 0 ]]; then
        exit 1
    fi

    exit 0
}

main "$@"
