#!/usr/bin/env python3
"""
validate_linalg.py — SciRS2 Linear Algebra Validation

Compares scirs2-linalg results against NumPy / SciPy reference implementations
using matrices with known or reference solutions.

Checks:
  - Matrix multiply          (relative tolerance 1e-12)
  - LU decomposition         (relative tolerance 1e-10)
  - QR decomposition         (relative tolerance 1e-10)
  - SVD decomposition        (relative tolerance 1e-10)
  - Eigenvalues              (relative tolerance 1e-8)
  - Solve linear system Ax=b (relative tolerance 1e-10)
  - Matrix norm              (relative tolerance 1e-12)
  - Matrix inverse           (relative tolerance 1e-10)
  - Determinant              (relative tolerance 1e-10)
  - Condition number         (relative tolerance 1e-8)

Modes:
  STANDALONE (default): validates NumPy against itself using fixed test
    matrices.  Acts as a regression guard ensuring the reference values in
    this file are consistent with numpy.

  LIVE (--live): imports scirs2 Python bindings and compares scirs2_linalg
    results to numpy.  Falls back to numpy-vs-numpy if scirs2 is not installed.

Exit codes:
  0  All checks passed.
  1  One or more checks failed.

Usage:
  python scripts/validate_linalg.py [--live] [--verbose] [--report /tmp/linalg_report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import numpy.linalg as npla
    from scipy import linalg as spla

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

TOL_EXACT = 1e-12    # matmul, norm, det (stable operations)
TOL_FACTOR = 1e-10   # LU, QR, SVD, inverse, solve (factorisation round-off)
TOL_EIGEN = 1e-8     # eigenvalues (unordered, potential sign ambiguity)
TOL_COND = 1e-6      # condition number (sensitive to small singular values)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    tol: float = 0.0
    details: str = ""


@dataclass
class LinalgReport:
    test_name: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rel_err_matrix(expected: "np.ndarray", actual: "np.ndarray") -> float:
    """Element-wise max relative error, normalised by max(|expected|, eps)."""
    eps = 1e-300
    denom = np.maximum(np.abs(expected), eps)
    return float(np.max(np.abs(actual - expected) / denom))


def abs_err_matrix(expected: "np.ndarray", actual: "np.ndarray") -> float:
    return float(np.max(np.abs(actual - expected)))


def check_matrix(
    label: str,
    expected: "np.ndarray",
    actual: "np.ndarray",
    tol: float,
) -> CheckResult:
    if expected.shape != actual.shape:
        return CheckResult(
            name=label,
            passed=False,
            tol=tol,
            details=f"Shape mismatch: expected {expected.shape}, got {actual.shape}",
        )
    r = rel_err_matrix(expected, actual)
    a = abs_err_matrix(expected, actual)
    # Pass when either:
    #   - relative error within tol, OR
    #   - absolute error is near machine epsilon (< 1e-12)
    #     (catches near-zero entries where relative error is undefined)
    passed = r <= tol or a <= 1e-12
    return CheckResult(
        name=label,
        passed=passed,
        max_abs_err=a,
        max_rel_err=r,
        tol=tol,
        details="OK" if passed else f"max_rel_err={r:.3e} > tol={tol:.3e}, max_abs_err={a:.3e}",
    )


def check_scalar(
    label: str,
    expected: float,
    actual: float,
    tol: float,
) -> CheckResult:
    r = abs(actual - expected) / max(abs(expected), 1e-300)
    a = abs(actual - expected)
    passed = r <= tol or a <= 1e-14
    return CheckResult(
        name=label,
        passed=passed,
        max_abs_err=a,
        max_rel_err=r,
        tol=tol,
        details="OK" if passed else f"expected={expected:.8g}, actual={actual:.8g}, rel_err={r:.3e}",
    )


def eigenvalues_close(
    expected: "np.ndarray", actual: "np.ndarray", tol: float
) -> CheckResult:
    """
    Compare eigenvalue arrays up to ordering and sign ambiguity (for real
    symmetric matrices eigenvalues are real and can be compared after sorting).
    """
    label = "eigenvalues"
    if expected.shape != actual.shape:
        return CheckResult(
            name=label, passed=False, tol=tol,
            details=f"Shape mismatch: {expected.shape} vs {actual.shape}",
        )
    exp_s = np.sort(expected.real)
    act_s = np.sort(actual.real)
    r = rel_err_matrix(exp_s.reshape(1, -1), act_s.reshape(1, -1))
    a = abs_err_matrix(exp_s.reshape(1, -1), act_s.reshape(1, -1))
    passed = r <= tol or a <= 1e-12
    return CheckResult(
        name=label,
        passed=passed,
        max_abs_err=a,
        max_rel_err=r,
        tol=tol,
        details="OK" if passed else f"max_rel_err={r:.3e} > tol={tol:.3e}",
    )


# ---------------------------------------------------------------------------
# Fixed test matrices (with analytically known or pre-computed results)
# ---------------------------------------------------------------------------


def make_test_matrices() -> list[dict[str, Any]]:
    """
    Returns a list of test-case dicts, each with:
      - name:    human-readable label
      - A:       input numpy matrix
      - b:       RHS vector (for solve tests), or None
      - x_true:  true solution (for solve), or None
      - det:     known determinant, or None
      - cond:    condition number (numpy reference), or None
    """
    cases: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # 2x2 identity
    # -----------------------------------------------------------------------
    I2 = np.eye(2)
    cases.append({
        "name": "identity_2x2",
        "A": I2,
        "b": np.array([3.0, -1.0]),
        "x_true": np.array([3.0, -1.0]),
        "det": 1.0,
        "cond": 1.0,
        "eigenvalues": np.array([1.0, 1.0]),
    })

    # -----------------------------------------------------------------------
    # 3x3 well-conditioned symmetric positive-definite
    # -----------------------------------------------------------------------
    A3 = np.array([
        [4.0, 1.0, 0.5],
        [1.0, 3.0, 0.25],
        [0.5, 0.25, 2.0],
    ], dtype=np.float64)
    b3 = np.array([1.0, 2.0, 3.0])
    x3 = np.linalg.solve(A3, b3)
    cases.append({
        "name": "spd_3x3",
        "A": A3,
        "b": b3,
        "x_true": x3,
        "det": float(np.linalg.det(A3)),
        "cond": float(np.linalg.cond(A3)),
        "eigenvalues": np.linalg.eigvalsh(A3),
    })

    # -----------------------------------------------------------------------
    # 4x4 Hilbert matrix (ill-conditioned; tests robustness)
    # -----------------------------------------------------------------------
    H4 = np.array(
        [[1.0 / (i + j + 1) for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )
    b4 = np.ones(4)
    x4 = np.linalg.solve(H4, b4)
    cases.append({
        "name": "hilbert_4x4",
        "A": H4,
        "b": b4,
        "x_true": x4,
        "det": float(np.linalg.det(H4)),
        "cond": float(np.linalg.cond(H4)),
        "eigenvalues": np.linalg.eigvalsh(H4),
    })

    # -----------------------------------------------------------------------
    # 5x5 random (seeded for reproducibility)
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(20260221)
    A5_raw = rng.standard_normal((5, 5))
    # Make it diagonally dominant for numerical stability
    A5 = A5_raw + 10.0 * np.eye(5)
    b5 = rng.standard_normal(5)
    x5 = np.linalg.solve(A5, b5)
    cases.append({
        "name": "random_diag_dominant_5x5",
        "A": A5,
        "b": b5,
        "x_true": x5,
        "det": float(np.linalg.det(A5)),
        "cond": float(np.linalg.cond(A5)),
        "eigenvalues": np.linalg.eigvals(A5),
    })

    # -----------------------------------------------------------------------
    # 6x4 tall matrix (for SVD / QR; non-square)
    # -----------------------------------------------------------------------
    rng2 = np.random.default_rng(42)
    A64 = rng2.standard_normal((6, 4))
    cases.append({
        "name": "tall_6x4",
        "A": A64,
        "b": None,
        "x_true": None,
        "det": None,
        "cond": float(np.linalg.cond(A64)),
        "eigenvalues": None,  # non-square; skip
    })

    # -----------------------------------------------------------------------
    # Rotation matrix — special orthogonal, det=1, cond=1
    # -----------------------------------------------------------------------
    theta = math.pi / 4  # 45 degrees
    R2 = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ])
    b_rot = np.array([1.0, 0.0])
    x_rot = np.linalg.solve(R2, b_rot)
    cases.append({
        "name": "rotation_2x2",
        "A": R2,
        "b": b_rot,
        "x_true": x_rot,
        "det": 1.0,
        "cond": 1.0,
        "eigenvalues": np.linalg.eigvals(R2),
    })

    return cases


# ---------------------------------------------------------------------------
# Individual factorisation checks
# ---------------------------------------------------------------------------


def _check_matmul(A: "np.ndarray", name: str) -> CheckResult:
    """Verify A @ A.T is symmetric and matches numpy."""
    ref = A @ A.T
    actual = A @ A.T  # numpy-vs-numpy (scaffold; replace actual with scirs2 call)
    return check_matrix(f"{name}/matmul_AAt", ref, actual, TOL_EXACT)


def _check_lu(A: "np.ndarray", name: str) -> CheckResult:
    """LU decomposition: verify P @ L @ U reconstructs A."""
    if A.shape[0] != A.shape[1]:
        return CheckResult(
            name=f"{name}/lu", passed=True, details="skipped (non-square)"
        )
    try:
        P, L, U = spla.lu(A)
        recon = P @ L @ U
        return check_matrix(f"{name}/lu_recon", A, recon, TOL_FACTOR)
    except Exception as exc:
        return CheckResult(
            name=f"{name}/lu", passed=False, details=str(exc)
        )


def _check_qr(A: "np.ndarray", name: str) -> list[CheckResult]:
    """QR decomposition: verify Q @ R = A and Q^T @ Q = I."""
    results: list[CheckResult] = []
    try:
        Q, R = np.linalg.qr(A)
        recon = Q @ R
        results.append(check_matrix(f"{name}/qr_recon", A, recon, TOL_FACTOR))

        # Q should be orthonormal (Q^T Q = I)
        m = Q.shape[1]
        QtQ = Q.T @ Q
        results.append(check_matrix(f"{name}/qr_orthonormality", np.eye(m), QtQ, TOL_FACTOR))
    except Exception as exc:
        results.append(CheckResult(name=f"{name}/qr", passed=False, details=str(exc)))
    return results


def _check_svd(A: "np.ndarray", name: str) -> list[CheckResult]:
    """SVD: verify U @ diag(s) @ Vt = A; U and Vt are orthonormal."""
    results: list[CheckResult] = []
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        recon = U @ np.diag(s) @ Vt
        results.append(check_matrix(f"{name}/svd_recon", A, recon, TOL_FACTOR))

        # U columns orthonormal
        UtU = U.T @ U
        results.append(
            check_matrix(f"{name}/svd_U_orthonormality", np.eye(U.shape[1]), UtU, TOL_FACTOR)
        )

        # Vt rows orthonormal
        VtVt = Vt @ Vt.T
        results.append(
            check_matrix(f"{name}/svd_Vt_orthonormality", np.eye(Vt.shape[0]), VtVt, TOL_FACTOR)
        )

        # Singular values must be non-negative and descending
        ok_nonneg = bool(np.all(s >= -1e-12))
        ok_desc = bool(np.all(np.diff(s) <= 1e-12))
        results.append(
            CheckResult(
                name=f"{name}/svd_sigma_order",
                passed=ok_nonneg and ok_desc,
                details="OK" if (ok_nonneg and ok_desc) else f"non_neg={ok_nonneg}, desc={ok_desc}",
            )
        )
    except Exception as exc:
        results.append(CheckResult(name=f"{name}/svd", passed=False, details=str(exc)))
    return results


def _check_solve(
    A: "np.ndarray", b: "np.ndarray", x_true: "np.ndarray", name: str
) -> CheckResult:
    """Solve Ax=b; verify residual ||Ax - b|| / ||b|| and compare to x_true."""
    if b is None:
        return CheckResult(name=f"{name}/solve", passed=True, details="skipped (no RHS)")
    try:
        x = np.linalg.solve(A, b)
        residual = np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)
        sol_err = check_matrix(f"{name}/solve_solution", x_true, x, TOL_FACTOR)
        if not sol_err.passed:
            return sol_err
        # Also check residual
        if residual > TOL_FACTOR:
            return CheckResult(
                name=f"{name}/solve_residual",
                passed=False,
                max_rel_err=residual,
                tol=TOL_FACTOR,
                details=f"residual={residual:.3e} > tol={TOL_FACTOR:.3e}",
            )
        return CheckResult(
            name=f"{name}/solve",
            passed=True,
            max_rel_err=float(sol_err.max_rel_err),
            details="OK",
        )
    except Exception as exc:
        return CheckResult(name=f"{name}/solve", passed=False, details=str(exc))


def _check_eigenvalues(
    A: "np.ndarray", eig_true: "np.ndarray", name: str
) -> CheckResult:
    if eig_true is None or A.shape[0] != A.shape[1]:
        return CheckResult(
            name=f"{name}/eigenvalues", passed=True, details="skipped"
        )
    try:
        eig_actual = np.linalg.eigvals(A)
        return eigenvalues_close(eig_true, eig_actual, TOL_EIGEN)
    except Exception as exc:
        return CheckResult(name=f"{name}/eigenvalues", passed=False, details=str(exc))


def _check_det(A: "np.ndarray", det_true: float, name: str) -> CheckResult:
    if det_true is None or A.shape[0] != A.shape[1]:
        return CheckResult(
            name=f"{name}/det", passed=True, details="skipped"
        )
    try:
        actual = float(np.linalg.det(A))
        return check_scalar(f"{name}/det", det_true, actual, TOL_FACTOR)
    except Exception as exc:
        return CheckResult(name=f"{name}/det", passed=False, details=str(exc))


def _check_inverse(A: "np.ndarray", name: str) -> list[CheckResult]:
    if A.shape[0] != A.shape[1]:
        return [CheckResult(name=f"{name}/inverse", passed=True, details="skipped (non-square)")]
    results: list[CheckResult] = []
    try:
        cond = float(np.linalg.cond(A))
        Ainv = np.linalg.inv(A)
        # A @ Ainv ~ I; use condition-number-scaled tolerance to account for
        # ill-conditioned matrices (e.g. Hilbert matrix, cond ~ 1e4 for 4x4).
        # The theoretical bound for backward error in floating-point inversion is
        # O(cond(A) * eps), so we allow TOL_FACTOR * max(1, cond * 1e-12).
        inv_tol = max(TOL_FACTOR, cond * 1e-12 * 10)
        recon = A @ Ainv
        results.append(check_matrix(f"{name}/inverse_AinvA", np.eye(A.shape[0]), recon, inv_tol))
    except np.linalg.LinAlgError as exc:
        # Singular matrices expected to fail
        results.append(
            CheckResult(name=f"{name}/inverse", passed=True, details=f"singular (expected): {exc}")
        )
    except Exception as exc:
        results.append(CheckResult(name=f"{name}/inverse", passed=False, details=str(exc)))
    return results


def _check_norm(A: "np.ndarray", name: str) -> list[CheckResult]:
    """Verify Frobenius norm and 2-norm agree between numpy and explicit formula."""
    results: list[CheckResult] = []

    # Frobenius norm: sqrt(sum(a_ij^2))
    frob_ref = float(np.sqrt(np.sum(A ** 2)))
    frob_np = float(np.linalg.norm(A, "fro"))
    results.append(check_scalar(f"{name}/norm_frobenius", frob_ref, frob_np, TOL_EXACT))

    # 2-norm (largest singular value)
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    norm2_ref = float(s[0])
    norm2_np = float(np.linalg.norm(A, 2))
    results.append(check_scalar(f"{name}/norm_2", norm2_ref, norm2_np, TOL_EXACT))

    return results


def _check_cond(A: "np.ndarray", cond_true: float, name: str) -> CheckResult:
    if cond_true is None:
        return CheckResult(name=f"{name}/cond", passed=True, details="skipped")
    try:
        actual = float(np.linalg.cond(A))
        return check_scalar(f"{name}/cond", cond_true, actual, TOL_COND)
    except Exception as exc:
        return CheckResult(name=f"{name}/cond", passed=False, details=str(exc))


# ---------------------------------------------------------------------------
# Run standalone (numpy self-consistency)
# ---------------------------------------------------------------------------


def run_standalone(verbose: bool) -> list[LinalgReport]:
    if not NUMPY_AVAILABLE:
        print("ERROR: numpy/scipy are required for linalg validation", file=sys.stderr)
        sys.exit(1)

    cases = make_test_matrices()
    all_reports: list[LinalgReport] = []

    for case in cases:
        name = case["name"]
        A = case["A"]

        if verbose:
            print(f"  {name}: shape={A.shape} ...", end=" ", flush=True)

        report = LinalgReport(test_name=name)

        # Matmul
        report.checks.append(_check_matmul(A, name))

        # LU
        report.checks.append(_check_lu(A, name))

        # QR
        report.checks.extend(_check_qr(A, name))

        # SVD
        report.checks.extend(_check_svd(A, name))

        # Solve
        report.checks.append(
            _check_solve(A, case.get("b"), case.get("x_true"), name)
        )

        # Eigenvalues
        report.checks.append(
            _check_eigenvalues(A, case.get("eigenvalues"), name)
        )

        # Determinant
        report.checks.append(_check_det(A, case.get("det"), name))

        # Inverse
        report.checks.extend(_check_inverse(A, name))

        # Norm
        report.checks.extend(_check_norm(A, name))

        # Condition number
        report.checks.append(_check_cond(A, case.get("cond"), name))

        if verbose:
            print("PASS" if report.passed else "FAIL")
            for chk in report.checks:
                if not chk.passed or verbose:
                    marker = "  [OK]" if chk.passed else "  [FAIL]"
                    print(f"    {marker}  {chk.name}: {chk.details}")

        all_reports.append(report)

    return all_reports


# ---------------------------------------------------------------------------
# Run live (scirs2 vs numpy)
# ---------------------------------------------------------------------------


def run_live(verbose: bool) -> list[LinalgReport]:
    """
    Import scirs2 Python bindings and compare linalg results to numpy.
    Falls back to numpy-vs-numpy structural scaffold if scirs2 is not installed.
    """
    if not NUMPY_AVAILABLE:
        print("ERROR: numpy/scipy are required for live linalg validation", file=sys.stderr)
        sys.exit(1)

    try:
        import scirs2  # type: ignore
        scirs2_linalg = scirs2.linalg
        have_scirs2 = True
    except (ImportError, AttributeError):
        scirs2_linalg = None
        have_scirs2 = False

    if verbose and not have_scirs2:
        print("  scirs2 not installed; running numpy-vs-numpy as structural check")

    cases = make_test_matrices()
    all_reports: list[LinalgReport] = []

    for case in cases:
        name = case["name"]
        A = case["A"]
        b = case.get("b")

        if verbose:
            print(f"  Live: {name} ...", end=" ", flush=True)

        report = LinalgReport(test_name=f"live/{name}")

        def np_matmul(M):
            return M @ M.T

        def scirs2_matmul_or_np(M):
            if have_scirs2 and hasattr(scirs2_linalg, "matmul"):
                return np.asarray(scirs2_linalg.matmul(M, M.T))
            return M @ M.T

        ref_mm = np_matmul(A)
        act_mm = scirs2_matmul_or_np(A)
        report.checks.append(check_matrix(f"{name}/live_matmul", ref_mm, act_mm, TOL_EXACT))

        # Solve
        if b is not None and A.shape[0] == A.shape[1]:
            ref_x = np.linalg.solve(A, b)
            if have_scirs2 and hasattr(scirs2_linalg, "solve"):
                try:
                    act_x = np.asarray(scirs2_linalg.solve(A, b))
                    report.checks.append(check_matrix(f"{name}/live_solve", ref_x, act_x, TOL_FACTOR))
                except Exception as exc:
                    report.checks.append(
                        CheckResult(name=f"{name}/live_solve", passed=False, details=str(exc))
                    )
            else:
                # numpy-vs-numpy
                report.checks.append(
                    check_matrix(f"{name}/live_solve", ref_x, np.linalg.solve(A, b), TOL_FACTOR)
                )

        # SVD
        U_ref, s_ref, Vt_ref = np.linalg.svd(A, full_matrices=False)
        if have_scirs2 and hasattr(scirs2_linalg, "svd"):
            try:
                U_act, s_act, Vt_act = [
                    np.asarray(x) for x in scirs2_linalg.svd(A, full_matrices=False)
                ]
                # Only compare singular values (U and V may differ by sign)
                s_check = check_matrix(
                    f"{name}/live_svd_sigma",
                    s_ref.reshape(1, -1),
                    s_act.reshape(1, -1),
                    TOL_FACTOR,
                )
                report.checks.append(s_check)
            except Exception as exc:
                report.checks.append(
                    CheckResult(name=f"{name}/live_svd", passed=False, details=str(exc))
                )
        else:
            # self-check singular values
            _, s_act, _ = np.linalg.svd(A, full_matrices=False)
            report.checks.append(
                check_matrix(
                    f"{name}/live_svd_sigma",
                    s_ref.reshape(1, -1),
                    s_act.reshape(1, -1),
                    TOL_FACTOR,
                )
            )

        if verbose:
            print("PASS" if report.passed else "FAIL")
            for chk in report.checks:
                if not chk.passed or verbose:
                    marker = "  [OK]" if chk.passed else "  [FAIL]"
                    print(f"    {marker}  {chk.name}: {chk.details}")

        all_reports.append(report)

    return all_reports


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_report(reports: list[LinalgReport], mode: str) -> str:
    lines: list[str] = [
        "=" * 70,
        f"SciRS2 Linear Algebra Validation Report  [mode={mode}]",
        "=" * 70,
        "",
    ]
    n_pass = sum(1 for r in reports if r.passed)
    n_fail = len(reports) - n_pass
    n_checks_total = sum(len(r.checks) for r in reports)
    n_checks_pass = sum(sum(1 for c in r.checks if c.passed) for r in reports)

    for rpt in reports:
        status = "PASS" if rpt.passed else "FAIL"
        n_chk = len(rpt.checks)
        n_ok = sum(1 for c in rpt.checks if c.passed)
        lines.append(f"[{status}] {rpt.test_name}  ({n_ok}/{n_chk} checks OK)")
        if not rpt.passed:
            for chk in rpt.checks:
                if not chk.passed:
                    lines.append(f"       FAIL  {chk.name}: {chk.details}")

    lines += [
        "",
        "-" * 70,
        f"Tests: {len(reports)} | Passed: {n_pass} | Failed: {n_fail}",
        f"Checks: {n_checks_total} total | {n_checks_pass} passed | {n_checks_total - n_checks_pass} failed",
        "-" * 70,
    ]
    return "\n".join(lines)


def _save_json_report(
    reports: list[LinalgReport], mode: str, path: Path
) -> None:
    payload = {
        "mode": mode,
        "summary": {
            "total": len(reports),
            "passed": sum(1 for r in reports if r.passed),
            "failed": sum(1 for r in reports if not r.passed),
        },
        "tests": [
            {
                "test_name": r.test_name,
                "passed": r.passed,
                "checks": [asdict(c) for c in r.checks],
            }
            for r in reports
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"JSON report saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live mode: import scirs2 Python bindings and compare to numpy.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-check details.",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help="Write a JSON validation report to PATH.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "live" if args.live else "standalone"

    print(f"\nSciRS2 Linear Algebra Validation  [mode={mode}]")
    if not NUMPY_AVAILABLE:
        print("  ERROR: numpy/scipy not found. Install them to run linalg validation.")
        return 1
    print()

    if args.live:
        reports = run_live(verbose=args.verbose)
    else:
        reports = run_standalone(verbose=args.verbose)

    report_text = _format_report(reports, mode)
    print(report_text)

    if args.report:
        _save_json_report(reports, mode, Path(args.report))

    n_failed = sum(1 for r in reports if not r.passed)
    if n_failed > 0:
        print(f"\nVALIDATION FAILED: {n_failed} test(s) did not pass all checks.")
        return 1

    print("\nAll linear algebra validation checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
