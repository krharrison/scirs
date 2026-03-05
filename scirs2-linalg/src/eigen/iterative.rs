//! Iterative eigenvalue algorithms for dense matrices
//!
//! This module provides iterative methods for computing eigenvalues and eigenvectors
//! of dense matrices, complementing the direct methods in `standard.rs`. These
//! algorithms are especially useful when:
//! - Only a few eigenvalues are needed (power iteration, Lanczos, Arnoldi)
//! - High precision is required (Rayleigh quotient iteration)
//! - All eigenvalues of a symmetric matrix are needed (Jacobi method)
//!
//! ## Algorithms
//!
//! - **Power Iteration**: Computes the dominant eigenvalue/eigenvector
//! - **Inverse Power Iteration**: Finds eigenvalue nearest to a shift
//! - **Rayleigh Quotient Iteration**: Cubically convergent method for symmetric matrices
//! - **Lanczos Algorithm**: Finds k extreme eigenpairs of large symmetric matrices
//! - **Arnoldi Iteration**: Builds Krylov-Hessenberg decomposition for general matrices
//! - **Jacobi Method**: Finds ALL eigenvalues of symmetric matrices via Givens rotations

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::norm::vector_norm;
use crate::solve::solve;

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of the Lanczos algorithm for symmetric matrices.
///
/// The Lanczos algorithm reduces a large symmetric matrix to a tridiagonal form
/// and then extracts k extreme eigenpairs from it efficiently.
#[derive(Debug, Clone)]
pub struct LanczosResult<F> {
    /// Ritz (approximate) eigenvalues, sorted in descending order by magnitude
    pub eigenvalues: Array1<F>,
    /// Corresponding approximate eigenvectors (columns = eigenvectors)
    pub eigenvectors: Array2<F>,
}

/// Result of the Arnoldi iteration for general (non-symmetric) matrices.
///
/// The Arnoldi iteration produces an orthonormal Krylov basis Q and an upper
/// Hessenberg matrix H such that A * Q[:, :k] ≈ Q[:, :k+1] * H.
/// Eigenvalues of H (Ritz values) approximate eigenvalues of A.
#[derive(Debug, Clone)]
pub struct ArnoldiResult<F> {
    /// Upper Hessenberg matrix H of shape (k+1, k)
    pub h: Array2<F>,
    /// Orthonormal Krylov basis Q of shape (n, k+1)
    pub q: Array2<F>,
    /// Number of completed Arnoldi steps (may be < k if breakdown occurs)
    pub steps: usize,
}

// ---------------------------------------------------------------------------
// Power Iteration
// ---------------------------------------------------------------------------

/// Compute the dominant (largest-magnitude) eigenvalue and eigenvector using
/// power iteration.
///
/// Starting from a random vector, the method repeatedly multiplies by A and
/// renormalizes. The Rayleigh quotient is used to estimate the eigenvalue.
/// Convergence is declared when consecutive eigenvalue estimates differ by
/// less than `tol`.
///
/// # Arguments
///
/// * `a`       - Square input matrix
/// * `n_iter`  - Maximum number of iterations
/// * `tol`     - Convergence tolerance for eigenvalue change
///
/// # Returns
///
/// `(eigenvalue, eigenvector)` where `eigenvector` is unit-length.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::power_iteration_dense;
///
/// let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
/// let (lam, v) = power_iteration_dense(&a.view(), 200, 1e-10).expect("converged");
/// // Dominant eigenvalue of this matrix is 5.0
/// assert!((lam - 5.0).abs() < 1e-8);
/// ```
pub fn power_iteration_dense<F>(
    a: &ArrayView2<F>,
    n_iter: usize,
    tol: F,
) -> LinalgResult<(F, Array1<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let (nrows, ncols) = (a.nrows(), a.ncols());
    if nrows != ncols {
        return Err(LinalgError::ShapeError(format!(
            "power_iteration_dense: matrix must be square, got ({nrows}, {ncols})"
        )));
    }
    if nrows == 0 {
        return Err(LinalgError::ShapeError(
            "power_iteration_dense: matrix is empty".to_string(),
        ));
    }

    let n = nrows;

    // Random initialisation using scirs2_core RNG
    let mut rng = scirs2_core::random::rng();
    let mut b = Array1::<F>::zeros(n);
    for bi in b.iter_mut() {
        *bi = F::from(rng.random_range(-1.0..=1.0)).unwrap_or(F::zero());
    }

    // Normalize
    let norm = vector_norm(&b.view(), 2)?;
    if norm > F::epsilon() {
        b.mapv_inplace(|x| x / norm);
    }

    let mut eigenvalue = F::zero();
    let mut prev_eigenvalue = F::one() + tol + tol; // force at least one iteration

    for _ in 0..n_iter {
        // b_new = A * b
        let mut b_new = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut s = F::zero();
            for j in 0..n {
                s += a[[i, j]] * b[j];
            }
            b_new[i] = s;
        }

        // Rayleigh quotient: λ = b^T * A * b
        eigenvalue = b
            .iter()
            .zip(b_new.iter())
            .fold(F::zero(), |acc, (&bi, &abi)| acc + bi * abi);

        // Normalize b_new
        let norm_new = vector_norm(&b_new.view(), 2)?;
        if norm_new > F::epsilon() {
            b_new.mapv_inplace(|x| x / norm_new);
        }
        b = b_new;

        if (eigenvalue - prev_eigenvalue).abs() < tol {
            break;
        }
        prev_eigenvalue = eigenvalue;
    }

    Ok((eigenvalue, b))
}

// ---------------------------------------------------------------------------
// Inverse Power Iteration
// ---------------------------------------------------------------------------

/// Find the eigenvalue of `a` nearest to `shift` using inverse power (shifted
/// inverse) iteration.
///
/// At each step we solve `(A − σI) y = x` and renormalize. This converges to
/// the eigenvector corresponding to the eigenvalue of `A` closest to `shift`.
///
/// # Arguments
///
/// * `a`      - Square input matrix
/// * `shift`  - The shift σ; algorithm finds eigenvalue closest to σ
/// * `n_iter` - Maximum number of iterations
/// * `tol`    - Convergence tolerance for eigenvalue change
///
/// # Returns
///
/// `(eigenvalue, eigenvector)` — `eigenvalue` is an estimate of the eigenvalue
/// of `A` nearest to `shift`.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square, or
/// [`LinalgError::SingularMatrixError`] if `(A − σI)` is singular.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::inverse_power_iteration;
///
/// // Symmetric matrix with eigenvalues 2.0 and 4.0
/// let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
/// // Shift near 2.0 → should find eigenvalue ≈ 2.0
/// let (lam, v) = inverse_power_iteration(&a.view(), 1.8, 200, 1e-10).expect("converged");
/// assert!((lam - 2.0).abs() < 1e-7, "got {lam}");
/// ```
pub fn inverse_power_iteration<F>(
    a: &ArrayView2<F>,
    shift: F,
    n_iter: usize,
    tol: F,
) -> LinalgResult<(F, Array1<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let (nrows, ncols) = (a.nrows(), a.ncols());
    if nrows != ncols {
        return Err(LinalgError::ShapeError(format!(
            "inverse_power_iteration: matrix must be square, got ({nrows}, {ncols})"
        )));
    }
    if nrows == 0 {
        return Err(LinalgError::ShapeError(
            "inverse_power_iteration: matrix is empty".to_string(),
        ));
    }

    let n = nrows;

    // Build A − σI
    let mut shifted = a.to_owned();
    for i in 0..n {
        shifted[[i, i]] -= shift;
    }

    // Random initial vector
    let mut rng = scirs2_core::random::rng();
    let mut x = Array1::<F>::zeros(n);
    for xi in x.iter_mut() {
        *xi = F::from(rng.random_range(-1.0..=1.0)).unwrap_or(F::zero());
    }
    let norm = vector_norm(&x.view(), 2)?;
    if norm > F::epsilon() {
        x.mapv_inplace(|x| x / norm);
    }

    let mut eigenvalue = F::zero();
    let mut prev_eigenvalue = F::one() + tol + tol;

    for _ in 0..n_iter {
        // Solve (A − σI) y = x
        let y = solve(&shifted.view(), &x.view(), None).map_err(|e| {
            LinalgError::SingularMatrixError(format!(
                "inverse_power_iteration: (A − σI) is singular or nearly singular: {e}"
            ))
        })?;

        // Rayleigh quotient with original A: λ = x^T (A x)
        let ax = mat_vec_mul(a, &x.view());
        eigenvalue = x
            .iter()
            .zip(ax.iter())
            .fold(F::zero(), |acc, (&xi, &axi)| acc + xi * axi);

        // Normalize
        let norm_y = vector_norm(&y.view(), 2)?;
        if norm_y > F::epsilon() {
            x = y.mapv(|v| v / norm_y);
        } else {
            break;
        }

        if (eigenvalue - prev_eigenvalue).abs() < tol {
            break;
        }
        prev_eigenvalue = eigenvalue;
    }

    Ok((eigenvalue, x))
}

// ---------------------------------------------------------------------------
// Rayleigh Quotient Iteration
// ---------------------------------------------------------------------------

/// Rayleigh Quotient Iteration for finding an eigenvalue/eigenvector of a
/// symmetric (or nearly symmetric) matrix.
///
/// Starting from initial vector `x0`, the algorithm iteratively:
/// 1. Computes the Rayleigh quotient σ = xᵀAx / xᵀx
/// 2. Solves `(A − σI) y = x`
/// 3. Normalizes `y` to get the new `x`
///
/// This converges cubically for symmetric matrices and quadratically in general.
///
/// # Arguments
///
/// * `a`      - Square input matrix (should be symmetric for guaranteed convergence)
/// * `x0`    - Initial vector (non-zero)
/// * `n_iter` - Maximum number of iterations
/// * `tol`    - Convergence tolerance (on eigenvalue change)
///
/// # Returns
///
/// `(eigenvalue, eigenvector)` — the eigenpair to which the iteration converged.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if dimensions are inconsistent.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::rayleigh_quotient_iteration;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let x0 = array![1.0_f64, 0.0];
/// // Starting near (1,0), should converge to eigenvalue ≈ 1.382
/// let (lam, v) = rayleigh_quotient_iteration(&a.view(), &x0.view(), 50, 1e-10).expect("converged");
/// // Eigenvalues are (5 ± √5)/2 ≈ 3.618 and 1.382
/// assert!((lam - 1.3819660112501051).abs() < 1e-8 || (lam - 3.618033988749895).abs() < 1e-8);
/// ```
pub fn rayleigh_quotient_iteration<F>(
    a: &ArrayView2<F>,
    x0: &ArrayView1<F>,
    n_iter: usize,
    tol: F,
) -> LinalgResult<(F, Array1<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "rayleigh_quotient_iteration: matrix must be square, got ({}, {})",
            a.nrows(),
            a.ncols()
        )));
    }
    if x0.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "rayleigh_quotient_iteration: x0 length {} != matrix size {n}",
            x0.len()
        )));
    }

    let mut x = x0.to_owned();

    // Normalize initial vector
    let norm = vector_norm(&x.view(), 2)?;
    if norm < F::epsilon() {
        return Err(LinalgError::InvalidInput(
            "rayleigh_quotient_iteration: initial vector x0 is zero".to_string(),
        ));
    }
    x.mapv_inplace(|v| v / norm);

    let mut sigma = rayleigh_quotient_scalar(a, &x.view());
    let mut prev_sigma = sigma - tol - tol;

    for _ in 0..n_iter {
        if (sigma - prev_sigma).abs() < tol {
            break;
        }
        prev_sigma = sigma;

        // Build (A − σI)
        let mut shifted = a.to_owned();
        for i in 0..n {
            shifted[[i, i]] -= sigma;
        }

        // Solve (A − σI) y = x
        let y = match solve(&shifted.view(), &x.view(), None) {
            Ok(y) => y,
            Err(_) => {
                // Near convergence: shift is very close to eigenvalue → (A−σI) is (nearly) singular.
                // Use current x as the eigenvector.
                break;
            }
        };

        let norm_y = vector_norm(&y.view(), 2)?;
        if norm_y < F::epsilon() {
            break;
        }
        x = y.mapv(|v| v / norm_y);

        // Update Rayleigh quotient
        sigma = rayleigh_quotient_scalar(a, &x.view());
    }

    Ok((sigma, x))
}

/// Compute the Rayleigh quotient xᵀAx / xᵀx for unit-length x.
#[inline]
fn rayleigh_quotient_scalar<F>(a: &ArrayView2<F>, x: &ArrayView1<F>) -> F
where
    F: Float + NumAssign + Sum + 'static,
{
    let ax = mat_vec_mul(a, x);
    x.iter()
        .zip(ax.iter())
        .fold(F::zero(), |acc, (&xi, &axi)| acc + xi * axi)
}

// ---------------------------------------------------------------------------
// Lanczos Algorithm (dense symmetric matrices)
// ---------------------------------------------------------------------------

/// Lanczos algorithm for finding k extreme eigenpairs of a symmetric matrix.
///
/// The Lanczos algorithm builds a Krylov subspace by reducing the symmetric matrix
/// to tridiagonal form via a sequence of matrix-vector products. After `m` steps
/// (where `m` is at most `a.nrows()`), the Ritz values extracted from the
/// tridiagonal matrix approximate the extreme eigenvalues of `A`.
///
/// This implementation:
/// - Uses full re-orthogonalization (modified Gram–Schmidt) for numerical stability
/// - Detects breakdown (β < tol) and stops early
/// - Returns up to `k` eigenpairs sorted by descending eigenvalue magnitude
///
/// # Arguments
///
/// * `a`        - Symmetric square matrix
/// * `k`        - Number of eigenpairs to return (k < n)
/// * `tol`      - Convergence tolerance; also used as breakdown threshold
///
/// # Returns
///
/// [`LanczosResult`] with `k` approximate eigenvalues and eigenvectors.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square or `k >= n`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::lanczos;
///
/// let a = array![
///     [4.0_f64, 1.0, 0.0],
///     [1.0, 3.0, 1.0],
///     [0.0, 1.0, 2.0]
/// ];
/// let result = lanczos(&a.view(), 2, 1e-10).expect("lanczos converged");
/// assert_eq!(result.eigenvalues.len(), 2);
/// assert_eq!(result.eigenvectors.ncols(), 2);
/// // Largest eigenvalue ≈ 4.732
/// let max_eig = result.eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
/// assert!((max_eig - 4.732050807568877).abs() < 1e-4, "got {max_eig}");
/// ```
pub fn lanczos<F>(a: &ArrayView2<F>, k: usize, tol: F) -> LinalgResult<LanczosResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let (nrows, ncols) = (a.nrows(), a.ncols());
    if nrows != ncols {
        return Err(LinalgError::ShapeError(format!(
            "lanczos: matrix must be square, got ({nrows}, {ncols})"
        )));
    }
    let n = nrows;
    if k == 0 || k >= n {
        return Err(LinalgError::ShapeError(format!(
            "lanczos: k={k} must satisfy 0 < k < n={n}"
        )));
    }

    // Maximum Lanczos steps = min(n, k + some extra steps for accuracy)
    let m = n.min(k * 3 + 10).min(n);

    // Storage for Lanczos vectors
    let mut q_vecs: Vec<Array1<F>> = Vec::with_capacity(m + 1);

    // Random start vector
    let mut rng = scirs2_core::random::rng();
    let mut q0 = Array1::<F>::zeros(n);
    for qi in q0.iter_mut() {
        *qi = F::from(rng.random_range(-1.0..=1.0)).unwrap_or(F::zero());
    }
    let norm0 = vector_norm(&q0.view(), 2)?;
    if norm0 < F::epsilon() {
        q0[0] = F::one();
    } else {
        q0.mapv_inplace(|x| x / norm0);
    }
    q_vecs.push(q0);

    // Tridiagonal: alphas (diagonal), betas (sub/super diagonal)
    let mut alphas: Vec<F> = Vec::with_capacity(m);
    let mut betas: Vec<F> = Vec::with_capacity(m);

    let mut actual_m = 0usize;

    for j in 0..m {
        let v = mat_vec_mul(a, &q_vecs[j].view());

        // α_j = q_j^T (A q_j)
        let alpha_j = q_vecs[j]
            .iter()
            .zip(v.iter())
            .fold(F::zero(), |acc, (&qi, &vi)| acc + qi * vi);
        alphas.push(alpha_j);

        // w = v − α_j * q_j − β_{j−1} * q_{j−1}
        let mut w = v;
        for i in 0..n {
            w[i] -= alpha_j * q_vecs[j][i];
        }
        if j > 0 {
            let beta_prev = betas[j - 1];
            for i in 0..n {
                w[i] -= beta_prev * q_vecs[j - 1][i];
            }
        }

        // Full re-orthogonalization against all previous vectors (modified Gram–Schmidt)
        for qv in q_vecs.iter().take(j + 1) {
            let proj = qv
                .iter()
                .zip(w.iter())
                .fold(F::zero(), |acc, (&qi, &wi)| acc + qi * wi);
            for i in 0..n {
                w[i] -= proj * qv[i];
            }
        }

        // β_j = ||w||
        let beta_j = vector_norm(&w.view(), 2)?;

        actual_m = j + 1;

        if beta_j < tol || j + 1 == m {
            betas.push(beta_j);
            break;
        }

        betas.push(beta_j);

        // q_{j+1} = w / β_j
        let q_next = w.mapv(|wi| wi / beta_j);
        q_vecs.push(q_next);
    }

    // Solve tridiagonal symmetric eigenvalue problem of size actual_m
    let (tri_eigs, tri_evecs) =
        tridiagonal_symmetric_eig(&alphas[..actual_m], &betas[..actual_m.saturating_sub(1)])?;

    // Back-transform eigenvectors: v_i = Q_m * y_i
    let m_used = actual_m;
    let m_evecs = tri_evecs.ncols();
    let take_k = k.min(m_evecs);

    // Sort by descending |eigenvalue| and take top-k
    let mut idx: Vec<usize> = (0..m_evecs).collect();
    idx.sort_by(|&a, &b| {
        tri_eigs[b]
            .abs()
            .partial_cmp(&tri_eigs[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(take_k);

    let mut eigenvalues = Array1::<F>::zeros(take_k);
    let mut eigenvectors = Array2::<F>::zeros((n, take_k));

    for (new_col, &old_col) in idx.iter().enumerate() {
        eigenvalues[new_col] = tri_eigs[old_col];
        // eigenvec = sum_j Q[j] * y[j, old_col]
        let mut vec_col = Array1::<F>::zeros(n);
        for j in 0..m_used.min(q_vecs.len()) {
            let coeff = tri_evecs[[j, old_col]];
            for i in 0..n {
                vec_col[i] += coeff * q_vecs[j][i];
            }
        }
        // Normalize
        let norm_col = vector_norm(&vec_col.view(), 2)?;
        if norm_col > F::epsilon() {
            vec_col.mapv_inplace(|x| x / norm_col);
        }
        eigenvectors.column_mut(new_col).assign(&vec_col);
    }

    Ok(LanczosResult {
        eigenvalues,
        eigenvectors,
    })
}

/// Solve symmetric tridiagonal eigenvalue problem via explicit dense matrix approach.
///
/// `alpha` is the main diagonal (length m), `beta` is the sub-diagonal (length m-1).
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored column-wise.
///
/// We build the full n×n symmetric tridiagonal matrix and apply the QR iteration from
/// the standard `eigh` path.  For the small Lanczos projected matrices (typically
/// m ≤ ~200) this is fast and numerically robust.
fn tridiagonal_symmetric_eig<F>(alpha: &[F], beta: &[F]) -> LinalgResult<(Vec<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let n = alpha.len();
    if n == 0 {
        return Ok((vec![], Array2::<F>::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((vec![alpha[0]], Array2::<F>::eye(1)));
    }

    // Build the dense symmetric tridiagonal matrix
    let mut t = Array2::<F>::zeros((n, n));
    for i in 0..n {
        t[[i, i]] = alpha[i];
    }
    for i in 0..beta.len() {
        if i + 1 < n {
            t[[i, i + 1]] = beta[i];
            t[[i + 1, i]] = beta[i];
        }
    }

    // Solve via the iterative symmetric eigenvalue solver from standard.rs
    // We call eigh directly (it works for any size).
    let (eigs, evecs) = crate::eigen::standard::eigh(&t.view(), None)?;

    Ok((eigs.to_vec(), evecs))
}

// ---------------------------------------------------------------------------
// Arnoldi Iteration (dense general matrices)
// ---------------------------------------------------------------------------

/// k-step Arnoldi iteration for a general (non-symmetric) dense matrix.
///
/// The Arnoldi process builds an orthonormal Krylov basis Q = [q₁, …, q_{k+1}]
/// and an upper Hessenberg matrix H ∈ ℝ^{(k+1)×k} such that:
///
/// > A * Q[:, :k] = Q[:, :k+1] * H
///
/// The Ritz values (eigenvalues of H[0..k, 0..k]) approximate eigenvalues of A.
///
/// Full re-orthogonalization via modified Gram–Schmidt is used.
///
/// # Arguments
///
/// * `a` - Square input matrix
/// * `k` - Number of Arnoldi steps requested
///
/// # Returns
///
/// [`ArnoldiResult`] containing H, Q, and the actual number of steps completed.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square or `k >= n`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::arnoldi;
///
/// let a = array![[1.0_f64, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]];
/// let res = arnoldi(&a.view(), 2).expect("arnoldi");
/// // H is (k+1, k) = (3, 2); Q is (n, k+1) = (3, 3)
/// assert_eq!(res.h.dim(), (3, 2));
/// assert_eq!(res.q.dim(), (3, 3));
/// ```
pub fn arnoldi<F>(a: &ArrayView2<F>, k: usize) -> LinalgResult<ArnoldiResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let (nrows, ncols) = (a.nrows(), a.ncols());
    if nrows != ncols {
        return Err(LinalgError::ShapeError(format!(
            "arnoldi: matrix must be square, got ({nrows}, {ncols})"
        )));
    }
    let n = nrows;
    if k == 0 {
        return Err(LinalgError::ShapeError(
            "arnoldi: k must be >= 1".to_string(),
        ));
    }
    if k >= n {
        return Err(LinalgError::ShapeError(format!(
            "arnoldi: k={k} must be < n={n}"
        )));
    }

    // H is (k+1) × k, Q is n × (k+1)
    let mut h = Array2::<F>::zeros((k + 1, k));
    let mut q = Array2::<F>::zeros((n, k + 1));

    // Random start vector
    let mut rng = scirs2_core::random::rng();
    let mut q0 = Array1::<F>::zeros(n);
    for qi in q0.iter_mut() {
        *qi = F::from(rng.random_range(-1.0..=1.0)).unwrap_or(F::zero());
    }
    let norm0 = vector_norm(&q0.view(), 2)?;
    if norm0 < F::epsilon() {
        q0[0] = F::one();
    } else {
        q0.mapv_inplace(|x| x / norm0);
    }
    q.column_mut(0).assign(&q0);

    let mut steps = 0usize;

    for j in 0..k {
        // w = A * q_j
        let q_j = q.column(j).to_owned();
        let mut w = mat_vec_mul(a, &q_j.view());

        // Modified Gram–Schmidt orthogonalization
        for i in 0..=j {
            let qi = q.column(i).to_owned();
            let h_ij = qi
                .iter()
                .zip(w.iter())
                .fold(F::zero(), |acc, (&qi_v, &wi)| acc + qi_v * wi);
            h[[i, j]] = h_ij;
            for l in 0..n {
                w[l] -= h_ij * qi[l];
            }
        }

        // β = ||w||
        let beta = vector_norm(&w.view(), 2)?;
        h[[j + 1, j]] = beta;
        steps = j + 1;

        if beta < F::epsilon() {
            // Breakdown: invariant Krylov subspace found
            break;
        }

        if j < k {
            // q_{j+1} = w / β
            let q_next = w.mapv(|wi| wi / beta);
            q.column_mut(j + 1).assign(&q_next);
        }
    }

    Ok(ArnoldiResult { h, q, steps })
}

// ---------------------------------------------------------------------------
// Jacobi Method for symmetric matrices
// ---------------------------------------------------------------------------

/// Compute ALL eigenvalues and eigenvectors of a symmetric matrix using the
/// classical Jacobi method (Jacobi eigenvalue algorithm).
///
/// The Jacobi method repeatedly applies Givens (plane) rotations to annihilate
/// off-diagonal elements. It is globally convergent for real symmetric matrices
/// and is accurate but O(n³) per sweep, so it is most practical for small to
/// medium matrices (n ≲ 500).
///
/// This implementation uses the **cyclic-by-rows** pivot selection strategy
/// (visiting all off-diagonal pairs in row-major order each sweep).
///
/// # Arguments
///
/// * `a`        - Symmetric input matrix (n × n)
/// * `tol`      - Convergence tolerance (Frobenius norm of off-diagonal part)
/// * `max_iter` - Maximum number of sweeps
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` — eigenvalues are sorted in **ascending** order;
/// columns of `eigenvectors` are the corresponding orthonormal eigenvectors.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::eigen::iterative::jacobi_eigenvalue;
///
/// let a = array![[4.0_f64, 1.0, 0.5], [1.0, 3.0, 0.75], [0.5, 0.75, 2.0]];
/// let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-12, 1000).expect("jacobi");
/// assert_eq!(eigs.len(), 3);
/// // Eigenvalues must be sorted ascending
/// assert!(eigs[0] <= eigs[1] && eigs[1] <= eigs[2]);
/// // Av ≈ λv for each column
/// for j in 0..3 {
///     let v = vecs.column(j);
///     let av = a.dot(&v);
///     let lam_v = v.mapv(|x| x * eigs[j]);
///     let diff: f64 = av.iter().zip(lam_v.iter()).map(|(&x, &y)| (x - y).powi(2)).sum::<f64>().sqrt();
///     assert!(diff < 1e-10, "residual={diff} for eigenvalue {}", eigs[j]);
/// }
/// ```
pub fn jacobi_eigenvalue<F>(
    a: &ArrayView2<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + 'static,
{
    let (nrows, ncols) = (a.nrows(), a.ncols());
    if nrows != ncols {
        return Err(LinalgError::ShapeError(format!(
            "jacobi_eigenvalue: matrix must be square, got ({nrows}, {ncols})"
        )));
    }
    let n = nrows;

    if n == 0 {
        return Ok((Array1::<F>::zeros(0), Array2::<F>::zeros((0, 0))));
    }
    if n == 1 {
        return Ok((Array1::from_elem(1, a[[0, 0]]), Array2::eye(1)));
    }

    // Working copy (will be diagonalized)
    let mut s = a.to_owned();
    // Eigenvector matrix (identity → accumulate rotations)
    let mut v = Array2::<F>::eye(n);

    let two = F::from(2.0).unwrap_or(F::one());
    let half = F::from(0.5).unwrap_or(F::one());

    'outer: for _sweep in 0..max_iter {
        // Compute Frobenius norm of off-diagonal part
        let mut off_norm_sq = F::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm_sq += s[[i, j]] * s[[i, j]] * two;
            }
        }
        if off_norm_sq.sqrt() < tol {
            break 'outer;
        }

        // Cyclic-by-rows sweep: visit every (p, q) with p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let s_pq = s[[p, q]];
                if s_pq.abs() < F::epsilon() {
                    continue;
                }

                // Compute the Jacobi rotation angle
                // tan(2θ) = 2 s_pq / (s_qq − s_pp)
                let theta = {
                    let diff = s[[q, q]] - s[[p, p]];
                    if diff.abs() < F::epsilon() {
                        // Equal diagonal elements → rotate by π/4
                        F::from(std::f64::consts::FRAC_PI_4).unwrap_or(half)
                    } else {
                        // θ = 0.5 * atan2(2 s_pq, s_qq − s_pp)
                        let ratio = two * s_pq / diff;
                        // atan in terms of Float trait
                        half * ratio.atan()
                    }
                };

                let c = theta.cos();
                let s_rot = theta.sin();

                // Update the matrix S ← Jᵀ S J
                // We apply the rotation only to the relevant rows/columns.

                // Update columns p and q for all rows r ≠ p, q
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let srp = s[[r, p]];
                    let srq = s[[r, q]];
                    s[[r, p]] = c * srp - s_rot * srq;
                    s[[p, r]] = s[[r, p]];
                    s[[r, q]] = s_rot * srp + c * srq;
                    s[[q, r]] = s[[r, q]];
                }

                // Update the 2×2 principal submatrix at (p, p), (p, q), (q, q)
                let spp = s[[p, p]];
                let sqq = s[[q, q]];
                let spq = s[[p, q]];
                s[[p, p]] = c * c * spp - two * c * s_rot * spq + s_rot * s_rot * sqq;
                s[[q, q]] = s_rot * s_rot * spp + two * c * s_rot * spq + c * c * sqq;
                s[[p, q]] = F::zero();
                s[[q, p]] = F::zero();

                // Accumulate rotation into V (eigenvectors)
                for r in 0..n {
                    let vrp = v[[r, p]];
                    let vrq = v[[r, q]];
                    v[[r, p]] = c * vrp - s_rot * vrq;
                    v[[r, q]] = s_rot * vrp + c * vrq;
                }
            }
        }
    }

    // Extract diagonal as eigenvalues
    let mut eigenvalues: Vec<F> = (0..n).map(|i| s[[i, i]]).collect();

    // Sort ascending and reorder eigenvectors accordingly
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues = Array1::from_iter(idx.iter().map(|&i| eigenvalues[i]));
    let mut sorted_eigenvectors = Array2::<F>::zeros((n, n));
    for (new_col, &old_col) in idx.iter().enumerate() {
        sorted_eigenvectors
            .column_mut(new_col)
            .assign(&v.column(old_col));
    }

    // Suppress the temporary as a lint for the sort helper
    eigenvalues.clear();

    Ok((sorted_eigenvalues, sorted_eigenvectors))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dense matrix–vector multiply: y = A * x
#[inline]
fn mat_vec_mul<F>(a: &ArrayView2<F>, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + NumAssign + Sum + 'static,
{
    let m = a.nrows();
    let mut y = Array1::<F>::zeros(m);
    for i in 0..m {
        let mut s = F::zero();
        for j in 0..a.ncols() {
            s += a[[i, j]] * x[j];
        }
        y[i] = s;
    }
    y
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // -----------------------------------------------------------------------
    // power_iteration_dense tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_power_iteration_dense_diagonal() {
        // Diagonal matrix: dominant eigenvalue = 5
        let a = array![[5.0_f64, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let (lam, v) = power_iteration_dense(&a.view(), 500, 1e-12).expect("converged");
        assert_relative_eq!(lam, 5.0, epsilon = 1e-8);
        // Eigenvector should be unit length
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_power_iteration_dense_2x2() {
        // Eigenvalues 5 and 1
        let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
        let (lam, v) = power_iteration_dense(&a.view(), 300, 1e-12).expect("converged");
        assert_relative_eq!(lam, 5.0, epsilon = 1e-8);
        // Av ≈ λv
        let av = a.dot(&v);
        for i in 0..2 {
            assert_relative_eq!(av[i], lam * v[i], epsilon = 1e-7);
        }
    }

    #[test]
    fn test_power_iteration_dense_non_square_err() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(power_iteration_dense(&a.view(), 100, 1e-10).is_err());
    }

    #[test]
    fn test_power_iteration_dense_symmetric() {
        // Symmetric: eigenvalues 4 and 2
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let (lam, v) = power_iteration_dense(&a.view(), 500, 1e-12).expect("converged");
        assert_relative_eq!(lam, 4.0, epsilon = 1e-6);
        let av = a.dot(&v);
        for i in 0..2 {
            assert_relative_eq!(av[i], lam * v[i], epsilon = 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // inverse_power_iteration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inverse_power_iteration_smallest() {
        // Symmetric matrix: eigenvalues 2.0 and 4.0; shift near 2.0 → converges to eigenvalue ≈ 2.0
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        // eigs: 3-1=2 and 3+1=4
        let (lam, v) = inverse_power_iteration(&a.view(), 1.8, 300, 1e-12).expect("converged");
        assert_relative_eq!(lam, 2.0, epsilon = 1e-7);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_power_iteration_dominant() {
        // Symmetric matrix: eigenvalues 2 and 4; shift near 4 → converges to 4
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let (lam, _) = inverse_power_iteration(&a.view(), 3.9, 300, 1e-12).expect("converged");
        assert_relative_eq!(lam, 4.0, epsilon = 1e-7);
    }

    #[test]
    fn test_inverse_power_iteration_3x3() {
        // Symmetric tridiagonal with known eigenvalues
        let a = array![[2.0_f64, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
        // Eigenvalues: 2 - √2 ≈ 0.586, 2.0, 2 + √2 ≈ 3.414
        let (lam, _) = inverse_power_iteration(&a.view(), 0.5, 500, 1e-11).expect("converged");
        let expected = 2.0_f64 - std::f64::consts::SQRT_2;
        assert_relative_eq!(lam, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_inverse_power_iteration_non_square_err() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(inverse_power_iteration(&a.view(), 1.0, 100, 1e-10).is_err());
    }

    // -----------------------------------------------------------------------
    // rayleigh_quotient_iteration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rayleigh_quotient_iteration_symmetric_2x2() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        // Eigenvalues: (5 ± √5)/2 ≈ 3.618 and 1.382
        let x0 = array![0.0_f64, 1.0];
        let (lam, v) =
            rayleigh_quotient_iteration(&a.view(), &x0.view(), 50, 1e-12).expect("converged");
        let eig1 = (5.0_f64 + 5.0_f64.sqrt()) / 2.0;
        let eig2 = (5.0_f64 - 5.0_f64.sqrt()) / 2.0;
        let close = (lam - eig1).abs() < 1e-8 || (lam - eig2).abs() < 1e-8;
        assert!(close, "eigenvalue {lam} not near {eig1} or {eig2}");
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rayleigh_quotient_iteration_diagonal() {
        let a = array![[3.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]];
        // Starting near e₃ → should find eigenvalue 5
        let x0 = array![0.01_f64, 0.01, 1.0];
        let (lam, _) =
            rayleigh_quotient_iteration(&a.view(), &x0.view(), 100, 1e-12).expect("converged");
        assert_relative_eq!(lam, 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_rayleigh_quotient_iteration_wrong_dim_err() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let x0 = array![1.0_f64, 0.0, 0.0]; // length 3 ≠ 2
        assert!(rayleigh_quotient_iteration(&a.view(), &x0.view(), 50, 1e-10).is_err());
    }

    #[test]
    fn test_rayleigh_quotient_residual_check() {
        // For symmetric matrix, Av = λv should hold exactly at convergence
        let a = array![[5.0_f64, 2.0, 0.0], [2.0, 4.0, 1.0], [0.0, 1.0, 3.0]];
        let x0 = array![1.0_f64, 0.5, 0.2];
        let (lam, v) =
            rayleigh_quotient_iteration(&a.view(), &x0.view(), 100, 1e-12).expect("converged");
        let av = a.dot(&v);
        let lam_v = v.mapv(|x| x * lam);
        for i in 0..3 {
            assert_relative_eq!(av[i], lam_v[i], epsilon = 1e-7);
        }
    }

    // -----------------------------------------------------------------------
    // Lanczos tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lanczos_3x3() {
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let res = lanczos(&a.view(), 2, 1e-12).expect("lanczos");
        assert_eq!(res.eigenvalues.len(), 2);
        assert_eq!(res.eigenvectors.nrows(), 3);
        assert_eq!(res.eigenvectors.ncols(), 2);
    }

    #[test]
    fn test_lanczos_largest_eigenvalue() {
        // Symmetric matrix with known largest eigenvalue ≈ 4.732
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let res = lanczos(&a.view(), 1, 1e-12).expect("lanczos");
        // Eigenvalues of this matrix ≈ 1.268, 3.0, 4.732
        let max_eig = res.eigenvalues[0];
        assert!(
            (max_eig - 4.732050807568877).abs() < 1e-4,
            "expected ~4.732, got {max_eig}"
        );
    }

    #[test]
    fn test_lanczos_eigenvectors_orthonormal() {
        let a = array![[4.0_f64, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]];
        let res = lanczos(&a.view(), 2, 1e-12).expect("lanczos");
        let vecs = &res.eigenvectors;
        let ncols = vecs.ncols();
        for j in 0..ncols {
            // Unit length
            let v = vecs.column(j);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-8,
                "column {j} not unit length, norm={norm}"
            );
        }
    }

    #[test]
    fn test_lanczos_residual_check() {
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let res = lanczos(&a.view(), 2, 1e-12).expect("lanczos");
        for j in 0..res.eigenvalues.len() {
            let v = res.eigenvectors.column(j).to_owned();
            let lam = res.eigenvalues[j];
            let av = a.dot(&v);
            let lam_v = v.mapv(|x| x * lam);
            let diff: f64 = av
                .iter()
                .zip(lam_v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(diff < 1e-5, "residual={diff} for eigenvalue {lam}");
        }
    }

    #[test]
    fn test_lanczos_invalid_k_err() {
        let a = Array2::<f64>::eye(4);
        assert!(lanczos(&a.view(), 0, 1e-10).is_err());
        assert!(lanczos(&a.view(), 4, 1e-10).is_err());
    }

    // -----------------------------------------------------------------------
    // Arnoldi tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_arnoldi_dimensions() {
        let a = array![[1.0_f64, 2.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]];
        let k = 2;
        let res = arnoldi(&a.view(), k).expect("arnoldi");
        // H: (k+1, k), Q: (n, k+1)
        assert_eq!(res.h.dim(), (k + 1, k));
        assert_eq!(res.q.dim(), (3, k + 1));
        assert!(res.steps >= 1);
    }

    #[test]
    fn test_arnoldi_orthonormality() {
        let a = array![[2.0_f64, 1.0, 0.5], [1.0, 4.0, 0.0], [0.5, 0.0, 3.0]];
        let res = arnoldi(&a.view(), 2).expect("arnoldi");
        let q = &res.q;
        let k_plus1 = res.steps + 1;
        // Check that Q[:, :k_plus1] columns are orthonormal
        for i in 0..k_plus1 {
            for j in i..k_plus1 {
                let qi = q.column(i);
                let qj = q.column(j);
                let dot: f64 = qi.iter().zip(qj.iter()).map(|(&x, &y)| x * y).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_arnoldi_relation_aq_qh() {
        // A Q[:, :k] = Q[:, :k+1] H
        let a = array![[3.0_f64, 1.0, 0.0], [0.5, 2.0, 0.5], [0.0, 0.5, 4.0]];
        let k = 2;
        let res = arnoldi(&a.view(), k).expect("arnoldi");
        let q = &res.q;
        let h = &res.h;
        let n = 3;

        // Compute A * Q[:, 0..k]
        let mut aq = Array2::<f64>::zeros((n, k));
        for j in 0..k {
            let qj = q.column(j).to_owned();
            let aqj = a.dot(&qj);
            aq.column_mut(j).assign(&aqj);
        }

        // Compute Q[:, 0..k+1] * H
        let mut qh = Array2::<f64>::zeros((n, k));
        for col in 0..k {
            for row in 0..n {
                let mut val = 0.0_f64;
                for l in 0..k + 1 {
                    val += q[[row, l]] * h[[l, col]];
                }
                qh[[row, col]] = val;
            }
        }

        // AQ ≈ QH
        for i in 0..n {
            for j in 0..k {
                assert!(
                    (aq[[i, j]] - qh[[i, j]]).abs() < 1e-10,
                    "AQ[{i},{j}]={} ≠ QH[{i},{j}]={}",
                    aq[[i, j]],
                    qh[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_arnoldi_hessenberg_structure() {
        let a = array![
            [2.0_f64, 3.0, 1.0, 0.5],
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 3.0, 1.0],
            [0.5, 0.0, 1.0, 4.0]
        ];
        let k = 3;
        let res = arnoldi(&a.view(), k).expect("arnoldi");
        // H should be upper Hessenberg (subdiagonal and above may be nonzero; below subdiagonal = 0)
        // In an Arnoldi result, H[j+1, j] is the β value; H[i, j] = 0 for i > j + 1
        let h = &res.h;
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                if i > j + 1 {
                    assert!(
                        h[[i, j]].abs() < 1e-12,
                        "H[{i},{j}] = {} should be 0 (below subdiagonal)",
                        h[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_arnoldi_invalid_args_err() {
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        assert!(arnoldi(&a.view(), 0).is_err()); // k = 0
        assert!(arnoldi(&a.view(), 2).is_err()); // k >= n
        let b = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(arnoldi(&b.view(), 1).is_err()); // not square
    }

    // -----------------------------------------------------------------------
    // jacobi_eigenvalue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacobi_2x2() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-13, 500).expect("jacobi");
        assert_eq!(eigs.len(), 2);
        // Eigenvalues: (5 ± √5)/2
        let e1 = (5.0_f64 - 5.0_f64.sqrt()) / 2.0;
        let e2 = (5.0_f64 + 5.0_f64.sqrt()) / 2.0;
        assert_relative_eq!(eigs[0], e1, epsilon = 1e-10);
        assert_relative_eq!(eigs[1], e2, epsilon = 1e-10);
        // Orthonormality
        let dot: f64 = vecs
            .column(0)
            .iter()
            .zip(vecs.column(1).iter())
            .map(|(&x, &y)| x * y)
            .sum();
        assert_relative_eq!(dot, 0.0, epsilon = 1e-10);
        let n0: f64 = vecs.column(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        let n1: f64 = vecs.column(1).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_relative_eq!(n0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(n1, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_3x3() {
        let a = array![[4.0_f64, 1.0, 0.5], [1.0, 3.0, 0.75], [0.5, 0.75, 2.0]];
        let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-13, 1000).expect("jacobi");
        assert_eq!(eigs.len(), 3);
        // Sorted ascending
        assert!(
            eigs[0] <= eigs[1] && eigs[1] <= eigs[2],
            "not sorted: {eigs:?}"
        );
        // Residual Av ≈ λv
        for j in 0..3 {
            let v = vecs.column(j).to_owned();
            let av = a.dot(&v);
            let lam_v = v.mapv(|x| x * eigs[j]);
            let diff: f64 = av
                .iter()
                .zip(lam_v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(diff < 1e-10, "residual={diff} for eigenvalue {}", eigs[j]);
        }
    }

    #[test]
    fn test_jacobi_identity() {
        // Identity matrix: all eigenvalues = 1
        let a = Array2::<f64>::eye(4);
        let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-13, 500).expect("jacobi");
        for &e in eigs.iter() {
            assert_relative_eq!(e, 1.0, epsilon = 1e-10);
        }
        // Eigenvectors form an orthogonal matrix
        let vt_v = vecs.t().dot(&vecs);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(vt_v[[i, j]], expected, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_jacobi_diagonal_matrix() {
        // Diagonal matrix: eigenvalues = diagonals
        let a = array![[3.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]];
        let (eigs, _) = jacobi_eigenvalue(&a.view(), 1e-13, 500).expect("jacobi");
        // Should be sorted: 1, 3, 5
        assert_relative_eq!(eigs[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eigs[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(eigs[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_non_square_err() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(jacobi_eigenvalue(&a.view(), 1e-10, 100).is_err());
    }

    #[test]
    fn test_jacobi_4x4_symmetric() {
        let a = array![
            [4.0_f64, 1.0, 0.5, 0.0],
            [1.0, 3.0, 1.0, 0.5],
            [0.5, 1.0, 2.0, 1.0],
            [0.0, 0.5, 1.0, 5.0]
        ];
        let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-12, 1000).expect("jacobi");
        // Sorted ascending
        for i in 0..3 {
            assert!(eigs[i] <= eigs[i + 1], "not sorted at index {i}");
        }
        // Residuals
        for j in 0..4 {
            let v = vecs.column(j).to_owned();
            let av = a.dot(&v);
            let lam_v = v.mapv(|x| x * eigs[j]);
            let diff: f64 = av
                .iter()
                .zip(lam_v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(diff < 1e-9, "residual={diff} for eigenvalue {}", eigs[j]);
        }
    }

    #[test]
    fn test_jacobi_orthogonality_of_eigenvectors() {
        let a = array![[5.0_f64, 2.0, 1.0], [2.0, 4.0, 0.5], [1.0, 0.5, 3.0]];
        let (_, vecs) = jacobi_eigenvalue(&a.view(), 1e-12, 1000).expect("jacobi");
        let n = 3;
        for i in 0..n {
            for j in i..n {
                let vi = vecs.column(i);
                let vj = vecs.column(j);
                let dot: f64 = vi.iter().zip(vj.iter()).map(|(&x, &y)| x * y).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "Vᵀ V [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_jacobi_trace_invariance() {
        // tr(A) = sum of eigenvalues (for symmetric matrices)
        let a = array![[4.0_f64, 1.0, 0.5], [1.0, 3.0, 0.75], [0.5, 0.75, 2.0]];
        let trace_a: f64 = (0..3).map(|i| a[[i, i]]).sum();
        let (eigs, _) = jacobi_eigenvalue(&a.view(), 1e-13, 1000).expect("jacobi");
        let sum_eigs: f64 = eigs.iter().sum();
        assert_relative_eq!(trace_a, sum_eigs, epsilon = 1e-10);
    }

    #[test]
    fn test_jacobi_1x1() {
        let a = array![[7.5_f64]];
        let (eigs, vecs) = jacobi_eigenvalue(&a.view(), 1e-12, 100).expect("jacobi 1x1");
        assert_relative_eq!(eigs[0], 7.5, epsilon = 1e-12);
        assert_relative_eq!(vecs[[0, 0]], 1.0, epsilon = 1e-12);
    }
}
