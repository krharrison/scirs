// API Stability Tests for scirs2-linalg
//
// These tests verify that the core public API surface has not been accidentally
// broken. If this file fails to compile, a previously-stable public item has
// been removed or its signature changed in a backward-incompatible way.
//
// Guidelines:
// - Compile-time checks are sufficient — tests verify *existence* of items.
// - No `unwrap()` — use `expect("…")` for any fallible calls.
// - Keep each test focused on one logical group of exports.

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

// Verify that the error module types are accessible.
#[test]
fn test_error_types_accessible() {
    use scirs2_linalg::error::{LinalgError, LinalgResult};

    fn _returns_linalg_result() -> LinalgResult<f64> {
        Ok(1.0)
    }
    let v = _returns_linalg_result().expect("should succeed");
    assert!((v - 1.0).abs() < 1e-14);

    // LinalgError can be constructed
    let _err = LinalgError::InvalidInput("test".to_string());
}

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

/// Verify that det, inv, trace, matrix_power are accessible.
#[test]
fn test_basic_ops_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{basic_trace, det, inv, matrix_power};

    let a = array![[4.0f64, 2.0], [2.0, 3.0]];

    let d = det(&a.view(), None).expect("det should succeed on non-singular matrix");
    assert!((d - 8.0).abs() < 1e-10);

    let inv_a = inv(&a.view(), None).expect("inv should succeed on non-singular matrix");
    assert_eq!(inv_a.shape(), &[2, 2]);

    let tr = basic_trace(&a.view()).expect("trace should succeed");
    assert!((tr - 7.0).abs() < 1e-10);

    let p2 = matrix_power(&a.view(), 2, None).expect("matrix_power should succeed");
    assert_eq!(p2.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Decompositions
// ---------------------------------------------------------------------------

/// Verify that LU, QR, SVD, Cholesky decompositions are accessible.
#[test]
fn test_decompositions_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{cholesky, lu, qr, svd};

    let a = array![[1.0f64, 2.0], [3.0, 4.0]];

    let (p, l, u) = lu(&a.view(), None).expect("lu should succeed");
    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);

    let (q, r) = qr(&a.view(), None).expect("qr should succeed");
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let (u_svd, s, vt) = svd(&a.view(), true, None).expect("svd should succeed");
    assert_eq!(s.len(), 2);
    let _ = (u_svd, vt);

    let spd = array![[4.0f64, 2.0], [2.0, 3.0]];
    let l_chol = cholesky(&spd.view(), None).expect("cholesky should succeed on SPD matrix");
    assert_eq!(l_chol.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Eigenvalues
// ---------------------------------------------------------------------------

/// Verify that eig, eigh, eigvals are accessible.
#[test]
fn test_eigenvalues_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{eig, eigh, eigvals};

    let a = array![[2.0f64, 1.0], [1.0, 2.0]];

    // eigvals returns complex eigenvalues
    let vals = eigvals(&a.view(), None).expect("eigvals should succeed");
    assert_eq!(vals.len(), 2);

    // eig returns complex eigenvalues and eigenvectors
    let (vals2, vecs) = eig(&a.view(), None).expect("eig should succeed");
    assert_eq!(vals2.len(), 2);
    assert_eq!(vecs.shape(), &[2, 2]);

    // eigh for symmetric matrices returns real eigenvalues
    let sym = array![[3.0f64, 1.0], [1.0, 3.0]];
    let (evals, evecs) = eigh(&sym.view(), None).expect("eigh should succeed");
    assert_eq!(evals.len(), 2);
    assert_eq!(evecs.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Linear system solvers
// ---------------------------------------------------------------------------

/// Verify that solve, lstsq, solve_triangular are accessible.
#[test]
fn test_solvers_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{lstsq, solve, solve_triangular};

    let a = array![[4.0f64, 2.0], [2.0, 3.0]];
    let b = array![6.0f64, 7.0];

    let x = solve(&a.view(), &b.view(), None).expect("solve should succeed");
    assert_eq!(x.len(), 2);

    let result = lstsq(&a.view(), &b.view(), None).expect("lstsq should succeed");
    assert_eq!(result.x.len(), 2);

    // Triangular solve (lower triangular, non-unit diagonal)
    let lower = array![[1.0f64, 0.0], [0.5, 1.0]];
    let rhs = array![1.0f64, 1.5];
    let xt = solve_triangular(&lower.view(), &rhs.view(), true, false)
        .expect("solve_triangular should succeed");
    assert_eq!(xt.len(), 2);
}

// ---------------------------------------------------------------------------
// Matrix functions
// ---------------------------------------------------------------------------

/// Verify that expm, logm, sqrtm are accessible.
#[test]
fn test_matrix_functions_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{expm, logm, sqrtm};

    let a = array![[1.0f64, 0.0], [0.0, 1.0]]; // identity

    let exp_a = expm(&a.view(), None).expect("expm of identity should succeed");
    assert_eq!(exp_a.shape(), &[2, 2]);

    // logm of identity (result should be near zero); logm takes only the matrix view
    let log_a = logm(&a.view()).expect("logm of identity should succeed");
    assert_eq!(log_a.shape(), &[2, 2]);

    // sqrtm requires (a, maxiter, tol)
    let sqrt_a = sqrtm(&a.view(), 100, 1e-12f64).expect("sqrtm of identity should succeed");
    assert_eq!(sqrt_a.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

/// Verify that matrix and vector norms are accessible.
#[test]
fn test_norms_accessible() {
    use scirs2_core::ndarray::array;
    use scirs2_linalg::{cond, matrix_norm, vector_norm};

    let a = array![[3.0f64, 0.0], [0.0, 4.0]];

    let fro = matrix_norm(&a.view(), "fro", None).expect("frobenius norm should succeed");
    assert!((fro - 5.0).abs() < 1e-10);

    let v = array![3.0f64, 4.0];
    let l2 = vector_norm(&v.view(), 2).expect("l2 norm should succeed");
    assert!((l2 - 5.0).abs() < 1e-10);

    let c = cond(&a.view(), None, None).expect("cond should succeed");
    assert!(c > 0.0);
}

// ---------------------------------------------------------------------------
// LinalgError variants for backward compatibility
// ---------------------------------------------------------------------------

/// Verify key error variants that callers may pattern-match on still exist.
#[test]
fn test_error_variants_accessible() {
    use scirs2_linalg::LinalgError;

    // These variants form part of the stable error surface.
    let _invalid = LinalgError::InvalidInput("test".to_string());
    let _singular = LinalgError::SingularMatrixError("test".to_string());
    let _conv = LinalgError::ConvergenceError("test".to_string());
    let _shape = LinalgError::ShapeError("test".to_string());
}
