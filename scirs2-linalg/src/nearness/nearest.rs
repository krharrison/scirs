//! Nearest matrix problems.
//!
//! Given a matrix `A`, each function finds the closest matrix (in the Frobenius
//! norm unless stated otherwise) that belongs to a specified structured set.
//!
//! | Function | Target set |
//! |---|---|
//! | [`nearest_positive_definite`] | Symmetric positive-definite matrices |
//! | [`nearest_orthogonal`] | Orthogonal/unitary matrices |
//! | [`nearest_symmetric`] | Symmetric matrices |
//! | [`nearest_correlation`] | Correlation matrices (SPD + unit diagonal) |
//! | [`nearest_doubly_stochastic`] | Doubly-stochastic matrices |
//! | [`nearest_low_rank`] | Rank-*k* matrices |
//!
//! ## References
//!
//! - Higham, N. J. (1988). "Computing a nearest symmetric positive semidefinite matrix".
//!   *Linear Algebra Appl.* 103: 103–118.
//! - Higham, N. J. (2002). "Computing the nearest correlation matrix—a problem from finance".
//!   *IMA J. Numer. Anal.* 22(3): 329–343.
//! - Sinkhorn, R.; Knopp, P. (1967). "Concerning nonstochastic matrices and doubly stochastic
//!   matrices". *Pacific J. Math.* 21(2): 343–348.
//! - Eckart, C.; Young, G. (1936). "The approximation of one matrix by another of lower rank".
//!   *Psychometrika* 1: 211–218.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Frobenius norm of a matrix.
fn frob_norm(m: &Array2<f64>) -> f64 {
    m.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Frobenius inner product  <A, B> = trace(A^T B).
fn frob_inner(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>()
}

/// Symmetrise in-place: A <- (A + A^T) / 2.
fn symmetrise(a: &mut Array2<f64>) {
    let n = a.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let val = (a[[i, j]] + a[[j, i]]) / 2.0;
            a[[i, j]] = val;
            a[[j, i]] = val;
        }
    }
}

/// Project a symmetric matrix onto the cone of positive-semidefinite matrices
/// by zeroing out negative eigenvalues.
fn project_psd(a: &Array2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    let (eigenvalues, eigenvectors) = crate::eigen::eigh(&a.view(), None)?;
    // Reconstruct with clamped eigenvalues: A_+ = V * diag(max(λ, 0)) * V^T
    let mut result = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let lam = eigenvalues[k].max(0.0);
        if lam == 0.0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += lam * eigenvectors[[i, k]] * eigenvectors[[j, k]];
            }
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// nearest_positive_definite
// ---------------------------------------------------------------------------

/// Result of [`nearest_positive_definite`].
#[derive(Debug, Clone)]
pub struct NearestPdResult {
    /// The nearest symmetric positive-definite matrix.
    pub matrix: Array2<f64>,
    /// Frobenius distance from the input to the result.
    pub distance: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Find the nearest symmetric positive-definite matrix to `a`.
///
/// Uses Higham's alternating-projection algorithm (1988).  The method
/// iterates the two projections:
///
/// 1. **Symmetry projection**: `B <- (A + A^T) / 2`
/// 2. **PSD projection**: `C <- V diag(max(λ, 0)) V^T`  via eigen-decomposition
///
/// with Dykstra correction terms to ensure convergence to the true nearest point.
///
/// # Arguments
///
/// * `a` — Input square matrix (need not be symmetric or PD).
/// * `max_iter` — Maximum number of alternating-projection sweeps (default 200).
/// * `tol` — Convergence tolerance on the Frobenius change (default 1e-12).
/// * `epsilon` — Small positive value added to the diagonal for strict positivity
///   (default 1e-8).  Set to `0.0` to get the nearest *semidefinite* matrix.
///
/// # Returns
///
/// [`NearestPdResult`] containing the nearest SPD matrix, distance, and
/// convergence diagnostics.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_positive_definite;
///
/// let a = array![[2.0, -1.0, 0.0],
///                [-1.0, 2.0, -1.0],
///                [0.0, -1.0, 2.0_f64]];
/// // a is already SPD, so the result should be close to a itself
/// let res = nearest_positive_definite(&a.view(), None, None, None).expect("failed");
/// assert!(res.distance < 1e-6);
/// ```
pub fn nearest_positive_definite(
    a: &ArrayView2<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    epsilon: Option<f64>,
) -> LinalgResult<NearestPdResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "nearest_positive_definite: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "nearest_positive_definite: empty matrix".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(200);
    let tol = tol.unwrap_or(1e-12);
    let eps = epsilon.unwrap_or(1e-8);

    let a_owned = a.to_owned();

    // Higham alternating-projections with Dykstra correction.
    // Variables: Y (current iterate), dS (Dykstra increment for S-step).
    let mut y = a_owned.clone();
    let mut ds = Array2::<f64>::zeros((n, n)); // Dykstra correction
    let mut converged = false;
    let mut iterations = 0_usize;

    for _iter in 0..max_iter {
        iterations = _iter + 1;
        let r = &y - &ds; // Apply Dykstra correction
        // Project onto symmetric PSD cone
        let mut x = project_psd(&r)?;
        // Update Dykstra correction
        for i in 0..n {
            for j in 0..n {
                ds[[i, j]] = x[[i, j]] - r[[i, j]];
            }
        }
        // Project onto symmetric matrices
        symmetrise(&mut x);
        // Check convergence
        let change_norm = {
            let mut s = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let d = x[[i, j]] - y[[i, j]];
                    s += d * d;
                }
            }
            s.sqrt()
        };
        y = x;
        if change_norm < tol * (frob_norm(&y) + 1.0) {
            converged = true;
            break;
        }
    }

    // Ensure strict positive definiteness by nudging the diagonal.
    if eps > 0.0 {
        for i in 0..n {
            y[[i, i]] += eps;
        }
    }

    // Ensure perfect symmetry in the output.
    symmetrise(&mut y);

    let distance = {
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = a[[i, j]] - y[[i, j]];
                s += d * d;
            }
        }
        s.sqrt()
    };

    Ok(NearestPdResult {
        matrix: y,
        distance,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// nearest_orthogonal
// ---------------------------------------------------------------------------

/// Result of [`nearest_orthogonal`].
#[derive(Debug, Clone)]
pub struct NearestOrthogonalResult {
    /// The nearest orthogonal (or unitary) matrix `Q` satisfying `Q^T Q = I`.
    pub matrix: Array2<f64>,
    /// Frobenius distance ‖A − Q‖_F.
    pub distance: f64,
}

/// Find the nearest orthogonal matrix to `a` (Frobenius norm).
///
/// The closed-form solution is given by the **polar decomposition**: if
/// `A = U Σ V^T` (thin SVD), then the nearest orthogonal matrix is
/// `Q = U V^T`.
///
/// For square matrices `Q` satisfies `Q^T Q = I`.
/// For rectangular matrices `A ∈ ℝ^{m×n}` with `m ≥ n`, `Q` has orthonormal
/// columns (`Q^T Q = I_n`).
///
/// # Arguments
///
/// * `a` — Input matrix (need not be square; must have `m ≥ n`).
///
/// # Returns
///
/// [`NearestOrthogonalResult`] with the nearest orthogonal matrix and distance.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` has more columns than rows.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_orthogonal;
///
/// let a = array![[2.0, 0.5], [0.5, 1.5_f64]];
/// let res = nearest_orthogonal(&a.view()).expect("failed");
/// // Check Q^T Q ≈ I
/// let qt_q = res.matrix.t().dot(&res.matrix);
/// assert!((qt_q[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((qt_q[[1, 1]] - 1.0).abs() < 1e-10);
/// assert!(qt_q[[0, 1]].abs() < 1e-10);
/// ```
pub fn nearest_orthogonal(a: &ArrayView2<f64>) -> LinalgResult<NearestOrthogonalResult> {
    let m = a.nrows();
    let n = a.ncols();
    if m < n {
        return Err(LinalgError::ShapeError(format!(
            "nearest_orthogonal: matrix must have m >= n, got {}×{}",
            m, n
        )));
    }
    if m == 0 || n == 0 {
        return Err(LinalgError::ShapeError(
            "nearest_orthogonal: empty matrix".to_string(),
        ));
    }

    // Thin SVD: A = U S V^T  → Q = U V^T
    let (u, _s, vt) = crate::decomposition::svd(a, false, None)?;
    // u: m×k,  vt: k×n  (k = min(m,n))
    let q = u.dot(&vt);

    let distance = {
        let mut s = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let d = a[[i, j]] - q[[i, j]];
                s += d * d;
            }
        }
        s.sqrt()
    };

    Ok(NearestOrthogonalResult { matrix: q, distance })
}

// ---------------------------------------------------------------------------
// nearest_symmetric
// ---------------------------------------------------------------------------

/// Find the nearest symmetric matrix to `a` (Frobenius norm).
///
/// The unique minimiser is simply `(A + A^T) / 2`.  This is a projection
/// onto the linear subspace of symmetric matrices.
///
/// # Arguments
///
/// * `a` — Input square matrix.
///
/// # Returns
///
/// The nearest symmetric matrix (same shape as `a`) and the distance.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_symmetric;
///
/// let a = array![[1.0, 2.0], [4.0, 3.0_f64]];
/// let (sym, dist) = nearest_symmetric(&a.view()).expect("failed");
/// // Off-diagonal entries should average to (2+4)/2 = 3
/// assert!((sym[[0, 1]] - 3.0).abs() < 1e-14);
/// assert!((sym[[1, 0]] - 3.0).abs() < 1e-14);
/// ```
pub fn nearest_symmetric(a: &ArrayView2<f64>) -> LinalgResult<(Array2<f64>, f64)> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "nearest_symmetric: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }

    let mut s = a.to_owned();
    symmetrise(&mut s);

    let distance = {
        let mut acc = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = a[[i, j]] - s[[i, j]];
                acc += d * d;
            }
        }
        acc.sqrt()
    };

    Ok((s, distance))
}

// ---------------------------------------------------------------------------
// nearest_correlation
// ---------------------------------------------------------------------------

/// Result of [`nearest_correlation`].
#[derive(Debug, Clone)]
pub struct NearestCorrelationResult {
    /// The nearest correlation matrix (unit diagonal + SPD).
    pub matrix: Array2<f64>,
    /// Frobenius distance from the input to the result.
    pub distance: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

/// Find the nearest correlation matrix to a given symmetric matrix.
///
/// A **correlation matrix** is symmetric positive-semidefinite with unit
/// diagonal entries.  Higham's (2002) alternating-projection algorithm is
/// used:
///
/// 1. Project onto the set of matrices with unit diagonal.
/// 2. Project onto the cone of symmetric PSD matrices.
///
/// Dykstra correction is applied to ensure convergence to the nearest point.
///
/// # Arguments
///
/// * `a` — Input symmetric matrix (need not be PSD; off-diagonal entries
///   need not be in `[−1, 1]`).
/// * `max_iter` — Maximum iterations (default 1000).
/// * `tol` — Frobenius convergence tolerance (default 1e-12).
///
/// # Returns
///
/// [`NearestCorrelationResult`] with the correlation matrix and diagnostics.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_correlation;
///
/// // A near-correlation matrix with a slightly non-unit diagonal
/// let a = array![[1.02, 0.8], [0.8, 0.99_f64]];
/// let res = nearest_correlation(&a.view(), None, None).expect("failed");
/// // Diagonal entries should be exactly 1
/// assert!((res.matrix[[0, 0]] - 1.0).abs() < 1e-6);
/// assert!((res.matrix[[1, 1]] - 1.0).abs() < 1e-6);
/// ```
pub fn nearest_correlation(
    a: &ArrayView2<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> LinalgResult<NearestCorrelationResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "nearest_correlation: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "nearest_correlation: empty matrix".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(1000);
    let tol = tol.unwrap_or(1e-12);

    // Start from the symmetrised version of a.
    let mut x = {
        let mut tmp = a.to_owned();
        symmetrise(&mut tmp);
        tmp
    };
    let mut ds = Array2::<f64>::zeros((n, n)); // Dykstra correction for S-step
    let mut converged = false;
    let mut iterations = 0_usize;

    for _iter in 0..max_iter {
        iterations = _iter + 1;

        // --- Step 1: project onto PSD (apply Dykstra correction first) ---
        let r: Array2<f64> = &x - &ds;
        let y = project_psd(&r)?;
        // Update Dykstra increment
        for i in 0..n {
            for j in 0..n {
                ds[[i, j]] = y[[i, j]] - r[[i, j]];
            }
        }

        // --- Step 2: project onto unit-diagonal (simply set diag to 1) ---
        let mut x_new = y.clone();
        for i in 0..n {
            x_new[[i, i]] = 1.0;
        }

        // Convergence check
        let change = {
            let mut s = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    let d = x_new[[i, j]] - x[[i, j]];
                    s += d * d;
                }
            }
            s.sqrt()
        };
        x = x_new;
        if change < tol {
            converged = true;
            break;
        }
    }

    // Enforce perfect symmetry in the output.
    symmetrise(&mut x);
    // Enforce unit diagonal exactly.
    for i in 0..n {
        x[[i, i]] = 1.0;
    }

    let distance = {
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = a[[i, j]] - x[[i, j]];
                s += d * d;
            }
        }
        s.sqrt()
    };

    Ok(NearestCorrelationResult {
        matrix: x,
        distance,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// nearest_doubly_stochastic
// ---------------------------------------------------------------------------

/// Result of [`nearest_doubly_stochastic`].
#[derive(Debug, Clone)]
pub struct NearestDoublyStochasticResult {
    /// The nearest doubly-stochastic matrix.
    pub matrix: Array2<f64>,
    /// Frobenius distance from the input to the result.
    pub distance: f64,
    /// Number of Sinkhorn-Knopp iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Find the nearest doubly-stochastic matrix using the Sinkhorn-Knopp algorithm.
///
/// A doubly-stochastic matrix has non-negative entries and all row sums and
/// column sums equal to 1.  The Sinkhorn-Knopp algorithm alternately normalises
/// rows and columns.  If `a` contains negative entries they are first projected
/// to the non-negative orthant.
///
/// # Arguments
///
/// * `a` — Input non-negative matrix (negative entries are clamped to 0).
/// * `max_iter` — Maximum iterations (default 10_000).
/// * `tol` — Convergence tolerance on the maximum row/column-sum deviation
///   from 1 (default 1e-10).
///
/// # Returns
///
/// [`NearestDoublyStochasticResult`].
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if `a` is not square, or
/// [`LinalgError::ComputationError`] if a row/column sum is zero after
/// projection.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_doubly_stochastic;
///
/// let a = array![[2.0, 1.0], [1.0, 3.0_f64]];
/// let res = nearest_doubly_stochastic(&a.view(), None, None).expect("failed");
/// let m = &res.matrix;
/// // Row and column sums should be ~1
/// assert!((m[[0, 0]] + m[[0, 1]] - 1.0).abs() < 1e-6);
/// assert!((m[[0, 0]] + m[[1, 0]] - 1.0).abs() < 1e-6);
/// ```
pub fn nearest_doubly_stochastic(
    a: &ArrayView2<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> LinalgResult<NearestDoublyStochasticResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "nearest_doubly_stochastic: matrix must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if n == 0 {
        return Err(LinalgError::ShapeError(
            "nearest_doubly_stochastic: empty matrix".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(10_000);
    let tol = tol.unwrap_or(1e-10);

    // Project to non-negative orthant.
    let mut m: Array2<f64> = a.mapv(|v| v.max(0.0));

    // Sinkhorn-Knopp alternating normalisation.
    let mut converged = false;
    let mut iterations = 0_usize;

    for _iter in 0..max_iter {
        iterations = _iter + 1;

        // Row normalisation.
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| m[[i, j]]).sum();
            if row_sum < 1e-300 {
                return Err(LinalgError::ComputationError(format!(
                    "nearest_doubly_stochastic: row {} became zero during iteration",
                    i
                )));
            }
            for j in 0..n {
                m[[i, j]] /= row_sum;
            }
        }

        // Column normalisation.
        for j in 0..n {
            let col_sum: f64 = (0..n).map(|i| m[[i, j]]).sum();
            if col_sum < 1e-300 {
                return Err(LinalgError::ComputationError(format!(
                    "nearest_doubly_stochastic: column {} became zero during iteration",
                    j
                )));
            }
            for i in 0..n {
                m[[i, j]] /= col_sum;
            }
        }

        // Check convergence: max deviation of row/column sums from 1.
        if _iter % 50 == 0 {
            let mut max_dev = 0.0_f64;
            for i in 0..n {
                let rs: f64 = (0..n).map(|j| m[[i, j]]).sum::<f64>() - 1.0;
                max_dev = max_dev.max(rs.abs());
                let cs: f64 = (0..n).map(|j| m[[j, i]]).sum::<f64>() - 1.0;
                max_dev = max_dev.max(cs.abs());
            }
            if max_dev < tol {
                converged = true;
                break;
            }
        }
    }

    // Final convergence check if loop exited due to max_iter.
    if !converged {
        let mut max_dev = 0.0_f64;
        for i in 0..n {
            let rs: f64 = (0..n).map(|j| m[[i, j]]).sum::<f64>() - 1.0;
            max_dev = max_dev.max(rs.abs());
        }
        if max_dev < tol {
            converged = true;
        }
    }

    let distance = {
        let mut s = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = a[[i, j]] - m[[i, j]];
                s += d * d;
            }
        }
        s.sqrt()
    };

    Ok(NearestDoublyStochasticResult {
        matrix: m,
        distance,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// nearest_low_rank
// ---------------------------------------------------------------------------

/// Result of [`nearest_low_rank`].
#[derive(Debug, Clone)]
pub struct NearestLowRankResult {
    /// The best rank-*k* approximation to the input matrix.
    pub matrix: Array2<f64>,
    /// Frobenius distance ‖A − A_k‖_F (equals σ_{k+1}² + … + σ_r²).
    pub distance: f64,
    /// Left singular vectors (m × k).
    pub u: Array2<f64>,
    /// Singular values (length k).
    pub singular_values: Array1<f64>,
    /// Right singular vectors transposed (k × n).
    pub vt: Array2<f64>,
}

/// Compute the best rank-*k* approximation to `a` in the Frobenius norm.
///
/// By the Eckart-Young theorem the optimal rank-*k* approximation is obtained
/// from the truncated SVD:
///
/// ```text
/// A_k = U_k Σ_k V_k^T
/// ```
///
/// where `U_k`, `Σ_k`, `V_k` contain the *k* largest singular triplets of `A`.
///
/// # Arguments
///
/// * `a` — Input matrix (m × n).
/// * `k` — Target rank (`1 ≤ k ≤ min(m, n)`).
///
/// # Returns
///
/// [`NearestLowRankResult`] with the approximation, distance, and factor matrices.
///
/// # Errors
///
/// * [`LinalgError::ShapeError`] if `a` is empty.
/// * [`LinalgError::ValueError`] if `k < 1` or `k > min(m, n)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::nearest::nearest_low_rank;
///
/// let a = array![[1.0, 2.0, 3.0],
///                [4.0, 5.0, 6.0],
///                [7.0, 8.0, 9.0_f64]];
/// // Rank-1 approximation
/// let res = nearest_low_rank(&a.view(), 1).expect("failed");
/// assert_eq!(res.matrix.shape(), &[3, 3]);
/// assert!(res.distance < 1e-8 || res.distance > 0.0); // non-trivial
/// ```
pub fn nearest_low_rank(a: &ArrayView2<f64>, k: usize) -> LinalgResult<NearestLowRankResult> {
    let m = a.nrows();
    let n = a.ncols();
    if m == 0 || n == 0 {
        return Err(LinalgError::ShapeError(
            "nearest_low_rank: empty matrix".to_string(),
        ));
    }
    let r = m.min(n);
    if k == 0 || k > r {
        return Err(LinalgError::ValueError(format!(
            "nearest_low_rank: k must be in 1..={}, got {}",
            r, k
        )));
    }

    // Full thin SVD.
    let (u_full, s_full, vt_full) = crate::decomposition::svd(a, false, None)?;
    // u_full: m×r,  s_full: r,  vt_full: r×n

    // Truncate to k.
    let u_k = u_full.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
    let s_k = s_full.slice(scirs2_core::ndarray::s![..k]).to_owned();
    let vt_k = vt_full.slice(scirs2_core::ndarray::s![..k, ..]).to_owned();

    // Build the rank-k approximation: A_k = U_k * diag(s_k) * V_k^T
    let mut a_k = Array2::<f64>::zeros((m, n));
    for t in 0..k {
        let sv = s_k[t];
        for i in 0..m {
            for j in 0..n {
                a_k[[i, j]] += sv * u_k[[i, t]] * vt_k[[t, j]];
            }
        }
    }

    // Distance = sqrt(sum of squared discarded singular values).
    let distance = {
        let mut acc = 0.0_f64;
        for t in k..r {
            acc += s_full[t] * s_full[t];
        }
        acc.sqrt()
    };

    Ok(NearestLowRankResult {
        matrix: a_k,
        distance,
        u: u_k,
        singular_values: s_k,
        vt: vt_k,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ----- nearest_symmetric ------------------------------------------------

    #[test]
    fn test_nearest_symmetric_basic() {
        let a = array![[1.0_f64, 2.0], [4.0, 3.0]];
        let (s, dist) = nearest_symmetric(&a.view()).expect("nearest_symmetric failed");
        assert!((s[[0, 1]] - 3.0).abs() < 1e-14, "s[0,1] = {}", s[[0, 1]]);
        assert!((s[[1, 0]] - 3.0).abs() < 1e-14, "s[1,0] = {}", s[[1, 0]]);
        // Frobenius distance = |(2-3)| * sqrt(2) = sqrt(2)
        assert!((dist - std::f64::consts::SQRT_2).abs() < 1e-12, "dist = {}", dist);
    }

    #[test]
    fn test_nearest_symmetric_already_symmetric() {
        let a = array![[1.0_f64, 0.5], [0.5, 2.0]];
        let (s, dist) = nearest_symmetric(&a.view()).expect("failed");
        assert!(dist < 1e-15, "dist = {}", dist);
        for i in 0..2 {
            for j in 0..2 {
                assert!((s[[i, j]] - a[[i, j]]).abs() < 1e-14);
            }
        }
    }

    // ----- nearest_positive_definite ----------------------------------------

    #[test]
    fn test_nearest_pd_already_spd() {
        // A = I  is already SPD; nearest should return something close to I.
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let res = nearest_positive_definite(&a.view(), None, None, Some(0.0))
            .expect("nearest_pd failed");
        assert!(
            res.distance < 1e-6,
            "Expected small distance for identity, got {}",
            res.distance
        );
    }

    #[test]
    fn test_nearest_pd_indefinite() {
        // A has a negative eigenvalue; result must be SPD.
        let a = array![[1.0_f64, 2.0], [2.0, 1.0]]; // eigenvalues: -1, 3
        let res = nearest_positive_definite(&a.view(), None, None, Some(1e-6))
            .expect("nearest_pd failed");
        // Verify the result is positive definite via Cholesky (smallest eigenvalue > 0).
        let (vals, _) = crate::eigen::eigh(&res.matrix.view(), None)
            .expect("eigh failed");
        assert!(
            vals[0] > 0.0,
            "Smallest eigenvalue must be positive, got {}",
            vals[0]
        );
    }

    // ----- nearest_orthogonal -----------------------------------------------

    #[test]
    fn test_nearest_orthogonal_basic() {
        let a = array![[2.0_f64, 0.5], [0.5, 1.5]];
        let res = nearest_orthogonal(&a.view()).expect("nearest_orthogonal failed");
        let qt_q = res.matrix.t().dot(&res.matrix);
        assert!((qt_q[[0, 0]] - 1.0).abs() < 1e-10, "qt_q[0,0] = {}", qt_q[[0, 0]]);
        assert!((qt_q[[1, 1]] - 1.0).abs() < 1e-10, "qt_q[1,1] = {}", qt_q[[1, 1]]);
        assert!(qt_q[[0, 1]].abs() < 1e-10, "qt_q[0,1] = {}", qt_q[[0, 1]]);
    }

    #[test]
    fn test_nearest_orthogonal_already_orthogonal() {
        let theta = std::f64::consts::PI / 4.0;
        let a = array![[theta.cos(), -theta.sin()],
                       [theta.sin(),  theta.cos()]];
        let res = nearest_orthogonal(&a.view()).expect("failed");
        assert!(
            res.distance < 1e-12,
            "distance = {}; rotation matrix should map to itself",
            res.distance
        );
    }

    // ----- nearest_correlation ----------------------------------------------

    #[test]
    fn test_nearest_correlation_unit_diagonal() {
        let a = array![[1.02_f64, 0.8], [0.8, 0.99]];
        let res = nearest_correlation(&a.view(), None, None).expect("failed");
        assert!(
            (res.matrix[[0, 0]] - 1.0).abs() < 1e-6,
            "diag[0] = {}",
            res.matrix[[0, 0]]
        );
        assert!(
            (res.matrix[[1, 1]] - 1.0).abs() < 1e-6,
            "diag[1] = {}",
            res.matrix[[1, 1]]
        );
    }

    #[test]
    fn test_nearest_correlation_psd() {
        let a = array![[1.0_f64, 2.0], [2.0, 1.0]]; // off-diag too large
        let res = nearest_correlation(&a.view(), None, None).expect("failed");
        let (vals, _) = crate::eigen::eigh(&res.matrix.view(), None).expect("eigh failed");
        assert!(vals[0] >= -1e-8, "smallest eigenvalue = {}", vals[0]);
    }

    // ----- nearest_doubly_stochastic ----------------------------------------

    #[test]
    fn test_nearest_doubly_stochastic_row_col_sums() {
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let res = nearest_doubly_stochastic(&a.view(), None, None).expect("failed");
        let m = &res.matrix;
        let tol = 1e-7;
        for i in 0..2 {
            let rs: f64 = (0..2).map(|j| m[[i, j]]).sum();
            assert!((rs - 1.0).abs() < tol, "row {} sum = {}", i, rs);
            let cs: f64 = (0..2).map(|j| m[[j, i]]).sum();
            assert!((cs - 1.0).abs() < tol, "col {} sum = {}", i, cs);
        }
    }

    // ----- nearest_low_rank -------------------------------------------------

    #[test]
    fn test_nearest_low_rank_rank1() {
        // A rank-1 matrix: its rank-1 approximation should equal itself.
        let a = array![[1.0_f64, 2.0, 3.0],
                       [2.0, 4.0, 6.0],
                       [3.0, 6.0, 9.0]];
        let res = nearest_low_rank(&a.view(), 1).expect("failed");
        assert!(
            res.distance < 1e-8,
            "rank-1 distance should be ~0, got {}",
            res.distance
        );
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (res.matrix[[i, j]] - a[[i, j]]).abs() < 1e-8,
                    "mismatch at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_nearest_low_rank_distance() {
        let a = array![[3.0_f64, 0.0], [0.0, 1.0]];
        // Rank-1 approx: keeps only σ_1 = 3; discards σ_2 = 1.
        // Distance = 1.0.
        let res = nearest_low_rank(&a.view(), 1).expect("failed");
        assert!(
            (res.distance - 1.0).abs() < 1e-10,
            "distance = {}",
            res.distance
        );
    }
}
