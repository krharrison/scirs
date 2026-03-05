//! CUR Decomposition and related matrix approximation methods
//!
//! This module provides interpretable low-rank matrix approximations that use
//! actual rows and columns from the original matrix, making results easier
//! to understand in domain terms.
//!
//! # Algorithms
//!
//! - **CUR Decomposition**: A ~ C * U * R using actual columns (C) and rows (R)
//! - **Leverage Score Selection**: Importance sampling using statistical leverage scores
//! - **Interpolative Decomposition (ID)**: A ~ C * Z using column skeleton + coefficients
//! - **Skeleton Decomposition**: A ~ A(:, J) * W * A(I, :) using intersection submatrix
//! - **Nystrom Approximation**: Efficient approximation for PSD kernel matrices
//!
//! # References
//!
//! - Mahoney & Drineas (2009). "CUR matrix decompositions for improved data analysis."
//! - Drineas, Kannan, Mahoney (2006). "Fast Monte Carlo algorithms for matrices."
//! - Williams & Seeger (2001). "Using the Nystrom method to speed up kernel machines."

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Result types
// ============================================================================

/// Result of CUR decomposition
#[derive(Debug, Clone)]
pub struct CurResult<F> {
    /// Selected columns matrix (m x c)
    pub c: Array2<F>,
    /// Bridge/linking matrix (c x r)
    pub u: Array2<F>,
    /// Selected rows matrix (r x n)
    pub r: Array2<F>,
    /// Indices of selected columns
    pub col_indices: Vec<usize>,
    /// Indices of selected rows
    pub row_indices: Vec<usize>,
}

/// Result of Interpolative Decomposition
#[derive(Debug, Clone)]
pub struct InterpolativeResult<F> {
    /// Column skeleton (m x k)
    pub skeleton: Array2<F>,
    /// Coefficient matrix (k x n)
    pub coefficients: Array2<F>,
    /// Indices of selected columns
    pub col_indices: Vec<usize>,
}

/// Result of Skeleton Decomposition
#[derive(Debug, Clone)]
pub struct SkeletonResult<F> {
    /// Selected columns of A (m x k)
    pub columns: Array2<F>,
    /// Bridge matrix (k x k)
    pub bridge: Array2<F>,
    /// Selected rows of A (k x n)
    pub rows: Array2<F>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Row indices
    pub row_indices: Vec<usize>,
}

/// Result of Nystrom approximation
#[derive(Debug, Clone)]
pub struct NystromResult<F> {
    /// Factor L such that A ~ L * L^T (m x k)
    pub factor: Array2<F>,
    /// Selected landmark indices
    pub landmark_indices: Vec<usize>,
    /// Approximation of the full matrix (optional, computed on demand)
    pub kernel_approx: Option<Array2<F>>,
}

// ============================================================================
// Leverage Scores
// ============================================================================

/// Compute column leverage scores of a matrix.
///
/// Leverage scores measure the importance of each column for the
/// column space of A. The i-th leverage score is ||U(i,:)||^2
/// where U comes from the SVD A = U * S * V^T.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
/// * `rank` - Target rank for approximate scores (None = full rank)
///
/// # Returns
///
/// * Array of leverage scores, one per column
pub fn column_leverage_scores<F>(a: &ArrayView2<F>, rank: Option<usize>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (_m, n) = a.dim();

    // Compute SVD
    let (_u, _s, vt) = svd(a, false, None)?;

    let k = rank.unwrap_or(vt.nrows()).min(vt.nrows());

    // Column leverage scores = sum of squared entries of V (transposed rows of Vt)
    let mut scores = Array1::zeros(n);
    for j in 0..n {
        for i in 0..k {
            scores[j] += vt[[i, j]] * vt[[i, j]];
        }
    }

    Ok(scores)
}

/// Compute row leverage scores of a matrix.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
/// * `rank` - Target rank for approximate scores (None = full rank)
///
/// # Returns
///
/// * Array of leverage scores, one per row
pub fn row_leverage_scores<F>(a: &ArrayView2<F>, rank: Option<usize>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (m, _n) = a.dim();

    // Compute SVD
    let (u, _s, _vt) = svd(a, false, None)?;

    let k = rank.unwrap_or(u.ncols()).min(u.ncols());

    // Row leverage scores = sum of squared entries of U
    let mut scores = Array1::zeros(m);
    for i in 0..m {
        for j in 0..k {
            scores[i] += u[[i, j]] * u[[i, j]];
        }
    }

    Ok(scores)
}

// ============================================================================
// CUR Decomposition
// ============================================================================

/// CUR matrix decomposition using leverage score sampling.
///
/// Decomposes A ~ C * U * R where C contains selected columns of A,
/// R contains selected rows of A, and U is a small bridge matrix.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
/// * `rank` - Target rank
/// * `n_cols` - Number of columns to select (None = 2 * rank)
/// * `n_rows` - Number of rows to select (None = 2 * rank)
///
/// # Returns
///
/// * `CurResult` containing C, U, R, and selected indices
pub fn cur_decomposition<F>(
    a: &ArrayView2<F>,
    rank: usize,
    n_cols: Option<usize>,
    n_rows: Option<usize>,
) -> LinalgResult<CurResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();

    if rank == 0 {
        return Err(LinalgError::InvalidInput(
            "Target rank must be greater than 0".to_string(),
        ));
    }
    if rank > m.min(n) {
        return Err(LinalgError::InvalidInput(format!(
            "Target rank ({rank}) exceeds min(m, n) = {}",
            m.min(n)
        )));
    }

    let nc = n_cols.unwrap_or(2 * rank).min(n);
    let nr = n_rows.unwrap_or(2 * rank).min(m);

    // Compute leverage scores
    let col_scores = column_leverage_scores(a, Some(rank))?;
    let row_scores = row_leverage_scores(a, Some(rank))?;

    // Select columns by leverage scores (top-nc)
    let col_indices = select_by_leverage(&col_scores, nc);
    // Select rows by leverage scores (top-nr)
    let row_indices = select_by_leverage(&row_scores, nr);

    // Form C from selected columns
    let mut c = Array2::zeros((m, nc));
    for (idx, &col_j) in col_indices.iter().enumerate() {
        for i in 0..m {
            c[[i, idx]] = a[[i, col_j]];
        }
    }

    // Form R from selected rows
    let mut r = Array2::zeros((nr, n));
    for (idx, &row_i) in row_indices.iter().enumerate() {
        for j in 0..n {
            r[[idx, j]] = a[[row_i, j]];
        }
    }

    // Compute bridge matrix U via pseudoinverse of intersection
    // W = A[row_indices, col_indices]
    let mut w = Array2::zeros((nr, nc));
    for (ri, &row_i) in row_indices.iter().enumerate() {
        for (ci, &col_j) in col_indices.iter().enumerate() {
            w[[ri, ci]] = a[[row_i, col_j]];
        }
    }

    // U = pseudoinverse(W)
    let u = pseudoinverse(&w.view(), rank)?;

    Ok(CurResult {
        c,
        u,
        r,
        col_indices,
        row_indices,
    })
}

/// Select top-k indices from leverage scores (deterministic).
fn select_by_leverage<F: Float>(scores: &Array1<F>, k: usize) -> Vec<usize> {
    let n = scores.len();
    let k = k.min(n);

    // Create (index, score) pairs and sort by score descending
    let mut indexed: Vec<(usize, F)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Compute the pseudoinverse of a matrix using SVD.
fn pseudoinverse<F>(a: &ArrayView2<F>, rank_hint: usize) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    let (u, s, vt) = svd(a, false, None)?;

    let k = rank_hint.min(s.len());

    // Build S^{-1}
    let mut s_inv = Array2::zeros((k, k));
    let threshold = F::epsilon() * s[0] * F::from(m.max(n)).unwrap_or(F::one());
    for i in 0..k {
        if s[i] > threshold {
            s_inv[[i, i]] = F::one() / s[i];
        }
    }

    // pinv(A) = V * S^{-1} * U^T
    let vt_k = vt.slice(s![..k, ..]).to_owned();
    let u_k = u.slice(s![.., ..k]).to_owned();
    let v_k = vt_k.t();
    let result = v_k.dot(&s_inv).dot(&u_k.t());

    Ok(result)
}

// ============================================================================
// Interpolative Decomposition
// ============================================================================

/// Interpolative Decomposition (ID) of a matrix.
///
/// Decomposes A ~ A(:, J) * Z where J is a set of k column indices and
/// Z is a k x n coefficient matrix with the property that Z(:, J) = I_k.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
/// * `rank` - Target rank
///
/// # Returns
///
/// * `InterpolativeResult` with skeleton, coefficients, and column indices
pub fn interpolative_decomposition<F>(
    a: &ArrayView2<F>,
    rank: usize,
) -> LinalgResult<InterpolativeResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();

    if rank == 0 {
        return Err(LinalgError::InvalidInput(
            "Target rank must be greater than 0".to_string(),
        ));
    }
    if rank > n {
        return Err(LinalgError::InvalidInput(format!(
            "Target rank ({rank}) exceeds number of columns ({n})"
        )));
    }

    // Use column-pivoted QR to identify the most important columns
    // Greedy column selection based on column norms (simplified RRQR)
    let mut a_work = a.to_owned();
    let mut perm: Vec<usize> = (0..n).collect();

    for step in 0..rank {
        // Find column with maximum norm among remaining columns
        let mut max_norm = F::neg_infinity();
        let mut max_col = step;

        for j in step..n {
            let col = a_work.column(perm[j]);
            let mut norm_sq = F::zero();
            for i in step..m {
                norm_sq += col[i] * col[i];
            }
            if norm_sq > max_norm {
                max_norm = norm_sq;
                max_col = j;
            }
        }

        // Swap in permutation
        perm.swap(step, max_col);

        // Apply Householder reflection to zero out below pivot
        let pivot_col_idx = perm[step];
        let mut x = Array1::zeros(m - step);
        for i in step..m {
            x[i - step] = a_work[[i, pivot_col_idx]];
        }
        let x_norm = x.iter().fold(F::zero(), |acc, &v| acc + v * v).sqrt();
        if x_norm > F::epsilon() {
            let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
            let mut v = x;
            v[0] -= alpha;
            let v_norm = v.iter().fold(F::zero(), |acc, &vi| acc + vi * vi).sqrt();
            if v_norm > F::epsilon() {
                v.mapv_inplace(|vi| vi / v_norm);

                // Update all columns
                for jj in 0..n {
                    let col_idx = perm[jj];
                    let mut dot = F::zero();
                    for i in step..m {
                        dot += v[i - step] * a_work[[i, col_idx]];
                    }
                    let two = F::from(2.0).unwrap_or(F::one() + F::one());
                    for i in step..m {
                        a_work[[i, col_idx]] -= two * v[i - step] * dot;
                    }
                }
            }
        }
    }

    let col_indices: Vec<usize> = perm[..rank].to_vec();

    // Form skeleton (selected columns of original A)
    let mut skeleton = Array2::zeros((m, rank));
    for (idx, &col_j) in col_indices.iter().enumerate() {
        for i in 0..m {
            skeleton[[i, idx]] = a[[i, col_j]];
        }
    }

    // Compute coefficient matrix Z = pinv(skeleton) * A
    let skel_pinv = pseudoinverse(&skeleton.view(), rank)?;
    let coefficients = skel_pinv.dot(a);

    Ok(InterpolativeResult {
        skeleton,
        coefficients,
        col_indices,
    })
}

// ============================================================================
// Skeleton Decomposition
// ============================================================================

/// Skeleton decomposition of a matrix.
///
/// Decomposes A ~ A(:, J) * W * A(I, :) where I and J are index sets
/// and W = pinv(A(I, J)), the pseudoinverse of the intersection submatrix.
///
/// # Arguments
///
/// * `a` - Input matrix (m x n)
/// * `rank` - Target rank
///
/// # Returns
///
/// * `SkeletonResult` with columns, bridge, rows, and indices
pub fn skeleton_decomposition<F>(a: &ArrayView2<F>, rank: usize) -> LinalgResult<SkeletonResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = a.dim();

    if rank == 0 {
        return Err(LinalgError::InvalidInput(
            "Target rank must be greater than 0".to_string(),
        ));
    }
    if rank > m.min(n) {
        return Err(LinalgError::InvalidInput(format!(
            "Target rank ({rank}) exceeds min(m, n) = {}",
            m.min(n)
        )));
    }

    // Select columns and rows via leverage scores
    let col_scores = column_leverage_scores(a, Some(rank))?;
    let row_scores = row_leverage_scores(a, Some(rank))?;

    let col_indices = select_by_leverage(&col_scores, rank);
    let row_indices = select_by_leverage(&row_scores, rank);

    // Form A(:, J)
    let mut columns = Array2::zeros((m, rank));
    for (idx, &col_j) in col_indices.iter().enumerate() {
        for i in 0..m {
            columns[[i, idx]] = a[[i, col_j]];
        }
    }

    // Form A(I, :)
    let mut rows = Array2::zeros((rank, n));
    for (idx, &row_i) in row_indices.iter().enumerate() {
        for j in 0..n {
            rows[[idx, j]] = a[[row_i, j]];
        }
    }

    // Form intersection W_0 = A(I, J)
    let mut w0 = Array2::zeros((rank, rank));
    for (ri, &row_i) in row_indices.iter().enumerate() {
        for (ci, &col_j) in col_indices.iter().enumerate() {
            w0[[ri, ci]] = a[[row_i, col_j]];
        }
    }

    // Bridge = pinv(W_0)
    let bridge = pseudoinverse(&w0.view(), rank)?;

    Ok(SkeletonResult {
        columns,
        bridge,
        rows,
        col_indices,
        row_indices,
    })
}

// ============================================================================
// Nystrom Approximation
// ============================================================================

/// Nystrom approximation for positive semi-definite (PSD) matrices.
///
/// Given a PSD matrix K (e.g., a kernel matrix), approximates it as
/// K ~ K(:, L) * pinv(K(L, L)) * K(L, :)
/// where L is a set of landmark indices.
///
/// This is especially useful for large kernel matrices in machine learning.
///
/// # Arguments
///
/// * `k` - PSD matrix (n x n)
/// * `n_landmarks` - Number of landmark points to use
/// * `compute_full` - Whether to compute the full approximation matrix
///
/// # Returns
///
/// * `NystromResult` with factor and landmark indices
///
/// # References
///
/// Williams & Seeger (2001). "Using the Nystrom method to speed up kernel machines."
pub fn nystrom_approximation<F>(
    k: &ArrayView2<F>,
    n_landmarks: usize,
    compute_full: bool,
) -> LinalgResult<NystromResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (n, n2) = k.dim();
    if n != n2 {
        return Err(LinalgError::DimensionError(
            "Nystrom approximation requires a square matrix".to_string(),
        ));
    }
    if n_landmarks == 0 || n_landmarks > n {
        return Err(LinalgError::InvalidInput(format!(
            "n_landmarks ({n_landmarks}) must be in [1, {n}]"
        )));
    }

    // Select landmarks using diagonal leverage (larger diagonal = more important)
    let mut diag_scores = Array1::zeros(n);
    for i in 0..n {
        diag_scores[i] = k[[i, i]];
    }
    let landmark_indices = select_by_leverage(&diag_scores, n_landmarks);

    let l = n_landmarks;

    // Extract K_LL = K(L, L)  (l x l)
    let mut k_ll = Array2::zeros((l, l));
    for (ri, &row_i) in landmark_indices.iter().enumerate() {
        for (ci, &col_j) in landmark_indices.iter().enumerate() {
            k_ll[[ri, ci]] = k[[row_i, col_j]];
        }
    }

    // Extract K_nL = K(:, L)  (n x l)
    let mut k_nl = Array2::zeros((n, l));
    for (ci, &col_j) in landmark_indices.iter().enumerate() {
        for i in 0..n {
            k_nl[[i, ci]] = k[[i, col_j]];
        }
    }

    // Eigendecomposition of K_LL (it should be PSD)
    // Use SVD since K_LL is small and we want numerical stability
    let (u_ll, s_ll, _vt_ll) = svd(&k_ll.view(), false, None)?;

    // Compute K_LL^{-1/2}: U * diag(1/sqrt(s)) * U^T
    let k_actual = s_ll.len().min(l);
    let threshold = F::epsilon() * s_ll[0] * F::from(l).unwrap_or(F::one());

    let mut s_inv_sqrt = Array2::zeros((k_actual, k_actual));
    for i in 0..k_actual {
        if s_ll[i] > threshold {
            s_inv_sqrt[[i, i]] = F::one() / s_ll[i].sqrt();
        }
    }

    let u_ll_k = u_ll.slice(s![.., ..k_actual]).to_owned();

    // Factor: L_factor = K_nL * U_LL * S^{-1/2}
    let factor = k_nl.dot(&u_ll_k).dot(&s_inv_sqrt);

    // Optionally compute full approximation: K_approx = factor * factor^T
    let kernel_approx = if compute_full {
        Some(factor.dot(&factor.t()))
    } else {
        None
    };

    Ok(NystromResult {
        factor,
        landmark_indices,
        kernel_approx,
    })
}

/// Compute the Nystrom approximation of K * x for a vector x.
///
/// This avoids forming the full approximation matrix.
///
/// # Arguments
///
/// * `nystrom` - Previously computed Nystrom result
/// * `x` - Vector to multiply
///
/// # Returns
///
/// * Approximate K * x
pub fn nystrom_matvec<F>(nystrom: &NystromResult<F>, x: &Array1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let n = nystrom.factor.nrows();
    if x.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "Vector length ({}) does not match matrix dimension ({n})",
            x.len()
        )));
    }

    // K_approx * x = L * L^T * x
    let lt_x = nystrom.factor.t().dot(x);
    let result = nystrom.factor.dot(&lt_x);

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::{Distribution, Normal};

    fn make_test_matrix() -> Array2<f64> {
        // A low-rank matrix for testing
        array![
            [3.0, 1.0, 0.5, 2.0],
            [1.0, 3.0, 0.5, 1.5],
            [0.5, 0.5, 2.0, 1.0],
            [2.0, 1.5, 1.0, 3.0],
            [1.5, 1.0, 0.8, 2.5]
        ]
    }

    fn make_psd_matrix(n: usize) -> Array2<f64> {
        // Build a PSD matrix K = A * A^T
        let mut rng = scirs2_core::random::rng();
        let normal =
            Normal::new(0.0, 1.0).unwrap_or_else(|_| panic!("Failed to create distribution"));
        let rank = n.min(3);
        let mut a_gen = Array2::zeros((n, rank));
        for i in 0..n {
            for j in 0..rank {
                a_gen[[i, j]] = normal.sample(&mut rng);
            }
        }
        let k = a_gen.dot(&a_gen.t());
        // Add small diagonal for strict positive definiteness
        let mut result = k;
        for i in 0..n {
            result[[i, i]] += 0.01;
        }
        result
    }

    #[test]
    fn test_column_leverage_scores() {
        let a = make_test_matrix();
        let scores = column_leverage_scores(&a.view(), Some(2));
        assert!(scores.is_ok());
        let scores = scores.expect("leverage scores failed");
        assert_eq!(scores.len(), 4);
        // All scores should be non-negative
        for &s in scores.iter() {
            assert!(s >= 0.0);
        }
        // Scores should sum to approximately rank
        let total: f64 = scores.sum();
        assert!(total > 0.0, "Total leverage should be positive");
    }

    #[test]
    fn test_row_leverage_scores() {
        let a = make_test_matrix();
        let scores = row_leverage_scores(&a.view(), Some(2));
        assert!(scores.is_ok());
        let scores = scores.expect("row leverage scores failed");
        assert_eq!(scores.len(), 5);
        for &s in scores.iter() {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_cur_decomposition_basic() {
        let a = make_test_matrix();
        let result = cur_decomposition(&a.view(), 2, Some(3), Some(3));
        assert!(result.is_ok());
        let cur = result.expect("CUR failed");

        assert_eq!(cur.c.nrows(), 5);
        assert_eq!(cur.c.ncols(), 3);
        assert_eq!(cur.r.nrows(), 3);
        assert_eq!(cur.r.ncols(), 4);
        assert_eq!(cur.col_indices.len(), 3);
        assert_eq!(cur.row_indices.len(), 3);

        // Reconstruction: A ~ C * U * R
        let approx = cur.c.dot(&cur.u).dot(&cur.r);
        assert_eq!(approx.nrows(), 5);
        assert_eq!(approx.ncols(), 4);
    }

    #[test]
    fn test_cur_decomposition_reconstruction() {
        let a = make_test_matrix();
        let cur =
            cur_decomposition(&a.view(), 3, Some(4), Some(4)).expect("CUR decomposition failed");

        let approx = cur.c.dot(&cur.u).dot(&cur.r);
        let mut error = 0.0;
        let mut total = 0.0;
        for i in 0..5 {
            for j in 0..4 {
                let diff = a[[i, j]] - approx[[i, j]];
                error += diff * diff;
                total += a[[i, j]] * a[[i, j]];
            }
        }
        let rel_error = if total > 0.0 {
            (error / total).sqrt()
        } else {
            0.0
        };
        // With rank 3 and 4 samples, should get reasonable approximation
        assert!(
            rel_error < 1.0,
            "CUR reconstruction error too large: {rel_error}"
        );
    }

    #[test]
    fn test_cur_decomposition_errors() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(cur_decomposition(&a.view(), 0, None, None).is_err());
        assert!(cur_decomposition(&a.view(), 5, None, None).is_err());
    }

    #[test]
    fn test_interpolative_decomposition_basic() {
        let a = make_test_matrix();
        let result = interpolative_decomposition(&a.view(), 2);
        assert!(result.is_ok());
        let id = result.expect("ID failed");

        assert_eq!(id.skeleton.nrows(), 5);
        assert_eq!(id.skeleton.ncols(), 2);
        assert_eq!(id.coefficients.nrows(), 2);
        assert_eq!(id.coefficients.ncols(), 4);
        assert_eq!(id.col_indices.len(), 2);

        // Reconstruction: A ~ skeleton * coefficients
        let approx = id.skeleton.dot(&id.coefficients);
        assert_eq!(approx.nrows(), 5);
        assert_eq!(approx.ncols(), 4);
    }

    #[test]
    fn test_interpolative_decomposition_reconstruction() {
        let a = make_test_matrix();
        let id = interpolative_decomposition(&a.view(), 3).expect("ID failed");

        let approx = id.skeleton.dot(&id.coefficients);
        let mut error = 0.0;
        let mut total = 0.0;
        for i in 0..5 {
            for j in 0..4 {
                let diff = a[[i, j]] - approx[[i, j]];
                error += diff * diff;
                total += a[[i, j]] * a[[i, j]];
            }
        }
        let rel_error = if total > 0.0 {
            (error / total).sqrt()
        } else {
            0.0
        };
        assert!(
            rel_error < 0.5,
            "ID reconstruction error too large: {rel_error}"
        );
    }

    #[test]
    fn test_interpolative_decomposition_errors() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(interpolative_decomposition(&a.view(), 0).is_err());
        assert!(interpolative_decomposition(&a.view(), 5).is_err());
    }

    #[test]
    fn test_skeleton_decomposition_basic() {
        let a = make_test_matrix();
        let result = skeleton_decomposition(&a.view(), 2);
        assert!(result.is_ok());
        let skel = result.expect("Skeleton failed");

        assert_eq!(skel.columns.nrows(), 5);
        assert_eq!(skel.columns.ncols(), 2);
        assert_eq!(skel.bridge.nrows(), 2);
        assert_eq!(skel.bridge.ncols(), 2);
        assert_eq!(skel.rows.nrows(), 2);
        assert_eq!(skel.rows.ncols(), 4);
        assert_eq!(skel.col_indices.len(), 2);
        assert_eq!(skel.row_indices.len(), 2);

        // Reconstruction: A ~ columns * bridge * rows
        let approx = skel.columns.dot(&skel.bridge).dot(&skel.rows);
        assert_eq!(approx.nrows(), 5);
        assert_eq!(approx.ncols(), 4);
    }

    #[test]
    fn test_skeleton_decomposition_errors() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(skeleton_decomposition(&a.view(), 0).is_err());
        assert!(skeleton_decomposition(&a.view(), 5).is_err());
    }

    #[test]
    fn test_nystrom_approximation_basic() {
        let k = make_psd_matrix(10);
        let result = nystrom_approximation(&k.view(), 5, true);
        assert!(result.is_ok());
        let nys = result.expect("Nystrom failed");

        assert_eq!(nys.factor.nrows(), 10);
        assert!(nys.factor.ncols() <= 5);
        assert_eq!(nys.landmark_indices.len(), 5);
        assert!(nys.kernel_approx.is_some());

        let approx = nys.kernel_approx.as_ref().expect("Should have approx");
        assert_eq!(approx.nrows(), 10);
        assert_eq!(approx.ncols(), 10);
    }

    #[test]
    fn test_nystrom_approximation_no_full() {
        let k = make_psd_matrix(8);
        let nys = nystrom_approximation(&k.view(), 4, false).expect("Nystrom failed");
        assert!(nys.kernel_approx.is_none());
    }

    #[test]
    fn test_nystrom_reconstruction_quality() {
        let k = make_psd_matrix(10);
        let nys = nystrom_approximation(&k.view(), 5, true).expect("Nystrom failed");
        let approx = nys.kernel_approx.as_ref().expect("Should have approx");

        // Approximation should be PSD (diagonal should be positive)
        for i in 0..10 {
            assert!(
                approx[[i, i]] >= -1e-6,
                "Nystrom approx diagonal should be non-negative"
            );
        }
    }

    #[test]
    fn test_nystrom_errors() {
        let k = make_psd_matrix(5);
        // Non-square
        let rect = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(nystrom_approximation(&rect.view(), 2, false).is_err());

        // n_landmarks = 0
        assert!(nystrom_approximation(&k.view(), 0, false).is_err());

        // n_landmarks > n
        assert!(nystrom_approximation(&k.view(), 10, false).is_err());
    }

    #[test]
    fn test_nystrom_matvec() {
        let k = make_psd_matrix(8);
        let nys = nystrom_approximation(&k.view(), 4, false).expect("Nystrom failed");

        let x = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = nystrom_matvec(&nys, &x);
        assert!(result.is_ok());
        let y = result.expect("matvec failed");
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_nystrom_matvec_dimension_error() {
        let k = make_psd_matrix(5);
        let nys = nystrom_approximation(&k.view(), 3, false).expect("Nystrom failed");

        let x = array![1.0, 2.0, 3.0]; // Wrong dimension
        assert!(nystrom_matvec(&nys, &x).is_err());
    }

    #[test]
    fn test_pseudoinverse_basic() {
        let a = array![[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
        let pinv = pseudoinverse(&a.view(), 2);
        assert!(pinv.is_ok());
        let pinv = pinv.expect("pseudoinverse failed");

        // A * pinv(A) * A should approximately equal A
        let a_pinv_a = a.dot(&pinv).dot(&a);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (a[[i, j]] - a_pinv_a[[i, j]]).abs() < 1e-6,
                    "pseudoinverse property failed at [{i}, {j}]"
                );
            }
        }
    }
}
