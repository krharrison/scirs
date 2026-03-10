//! Sparse QR factorization
//!
//! This module provides Householder-based sparse QR factorization with column
//! pivoting for rank-revealing decompositions and sparse least-squares solvers.
//!
//! # Features
//!
//! - **Householder QR**: Column-oriented Householder reflections on sparse matrices
//! - **Column pivoting**: Rank-revealing QR via column norm pivoting
//! - **Sparse least squares**: Solves min ||Ax - b||_2 via QR
//! - **Economy-size QR**: Thin Q factor for overdetermined systems (m > n)
//!
//! # Algorithm
//!
//! The factorization computes A*P = Q*R where:
//! - Q is m x m orthogonal (or m x n in economy mode)
//! - R is m x n (or n x n in economy mode) upper triangular
//! - P is a column permutation (for pivoted QR)
//!
//! Because sparse Q factors are expensive to store, this module stores
//! Householder vectors implicitly and provides apply/solve operations.
//!
//! # References
//!
//! - Golub, G.H. & Van Loan, C.F. (2013). "Matrix Computations". 4th ed.
//! - Davis, T.A. (2011). "Algorithm 915: SuiteSparseQR". ACM TOMS 38(1).

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for sparse QR factorization.
#[derive(Debug, Clone)]
pub struct SparseQrConfig {
    /// Whether to use column pivoting for rank-revealing QR.
    pub pivoting: bool,
    /// Tolerance for determining numerical rank (relative to max column norm).
    pub rank_tol: f64,
    /// Whether to compute economy-size factorization.
    pub economy: bool,
}

impl Default for SparseQrConfig {
    fn default() -> Self {
        Self {
            pivoting: true,
            rank_tol: 1e-12,
            economy: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of sparse QR factorization.
///
/// Stores the factorization in implicit Householder form:
/// - Householder vectors `v_k` and scalars `tau_k` define Q = I - V * T * V^T
/// - R is stored as a dense upper triangular matrix
/// - Column permutation for pivoted QR
#[derive(Debug, Clone)]
pub struct SparseQrResult<F> {
    /// Householder vectors, stored column-wise.
    /// `householder_v[k]` is the k-th Householder vector (length m).
    pub householder_v: Vec<Vec<F>>,
    /// Householder scalars `tau[k]` = 2 / (v^T v).
    pub tau: Vec<F>,
    /// R factor stored as dense row-major (min(m,n) x n).
    pub r_data: Vec<Vec<F>>,
    /// Column permutation (P): r_data columns correspond to original columns `col_perm[j]`.
    pub col_perm: Vec<usize>,
    /// Number of rows (m).
    pub m: usize,
    /// Number of columns (n).
    pub n: usize,
    /// Numerical rank determined during factorization.
    pub rank: usize,
}

/// Result of sparse least squares: min ||Ax - b||.
#[derive(Debug, Clone)]
pub struct SparseLeastSquaresResult<F> {
    /// Solution vector x.
    pub solution: Vec<F>,
    /// Residual norm ||Ax - b||_2.
    pub residual_norm: F,
    /// Numerical rank of A.
    pub rank: usize,
}

// ---------------------------------------------------------------------------
// Core factorization
// ---------------------------------------------------------------------------

/// Compute sparse QR factorization of an m x n CSR matrix.
///
/// Returns Q (implicitly via Householder vectors) and R (dense upper triangular).
/// With column pivoting enabled, also returns a permutation vector.
pub fn sparse_qr<F>(
    matrix: &CsrMatrix<F>,
    config: &SparseQrConfig,
) -> SparseResult<SparseQrResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let m = matrix.rows();
    let n = matrix.cols();

    if m == 0 || n == 0 {
        return Ok(SparseQrResult {
            householder_v: Vec::new(),
            tau: Vec::new(),
            r_data: Vec::new(),
            col_perm: (0..n).collect(),
            m,
            n,
            rank: 0,
        });
    }

    // Convert to dense column-major storage for Householder reflections.
    // For very large matrices this would be done in sparse format, but for
    // correctness we use dense storage with the sparse input.
    let mut cols_data = extract_columns(matrix, m, n);

    // Column permutation
    let mut col_perm: Vec<usize> = (0..n).collect();

    // Compute column norms for pivoting
    let mut col_norms: Vec<F> = (0..n).map(|j| column_norm(&cols_data[j])).collect();

    let k_max = m.min(n);
    let mut householder_v: Vec<Vec<F>> = Vec::with_capacity(k_max);
    let mut tau_vec: Vec<F> = Vec::with_capacity(k_max);
    let mut rank = k_max;

    let rank_tol = F::from(config.rank_tol).unwrap_or_else(|| F::epsilon());

    // Determine max norm for relative tolerance
    let max_norm = col_norms
        .iter()
        .copied()
        .fold(F::sparse_zero(), |a, b| if b > a { b } else { a });

    for k in 0..k_max {
        // Column pivoting: swap column with largest remaining norm
        if config.pivoting {
            let mut best_j = k;
            let mut best_norm = col_norms[k];
            for j in (k + 1)..n {
                if col_norms[j] > best_norm {
                    best_norm = col_norms[j];
                    best_j = j;
                }
            }
            if best_j != k {
                cols_data.swap(k, best_j);
                col_perm.swap(k, best_j);
                col_norms.swap(k, best_j);
            }

            // Check for rank deficiency
            if best_norm < rank_tol * max_norm {
                rank = k;
                // Fill remaining Householder vectors with identity reflections
                for _i in k..k_max {
                    let v = vec![F::sparse_zero(); m];
                    householder_v.push(v);
                    tau_vec.push(F::sparse_zero());
                }
                break;
            }
        }

        // Compute Householder vector for column k, rows k..m
        let (v, tau) = householder_vector(&cols_data[k], k, m);

        // Apply the Householder reflection to columns k..n
        // H = I - tau * v * v^T
        // H * A[:, j] = A[:, j] - tau * v * (v^T * A[:, j])
        for j in k..n {
            let dot = dot_from(m, &v, &cols_data[j], k);
            let scale = tau * dot;
            for i in k..m {
                cols_data[j][i] -= scale * v[i];
            }
        }

        // Update column norms (fast downdate)
        for j in (k + 1)..n {
            let r_kj = cols_data[j][k];
            let old_norm_sq = col_norms[j] * col_norms[j];
            let new_norm_sq = old_norm_sq - r_kj * r_kj;
            col_norms[j] = if new_norm_sq > F::sparse_zero() {
                new_norm_sq.sqrt()
            } else {
                // Recompute from scratch to avoid negative sqrt
                column_norm_from(&cols_data[j], k + 1, m)
            };
        }

        householder_v.push(v);
        tau_vec.push(tau);
    }

    // Extract R from the modified columns
    let r_rows = if config.economy { rank } else { m };
    let mut r_data = vec![vec![F::sparse_zero(); n]; r_rows];
    for j in 0..n {
        for i in 0..r_rows.min(j + 1) {
            r_data[i][j] = cols_data[j][i];
        }
    }

    Ok(SparseQrResult {
        householder_v,
        tau: tau_vec,
        r_data,
        col_perm,
        m,
        n,
        rank,
    })
}

// ---------------------------------------------------------------------------
// Sparse least squares
// ---------------------------------------------------------------------------

/// Solve the sparse least squares problem min ||Ax - b||_2.
///
/// Uses QR factorization with column pivoting. For overdetermined systems
/// (m > n), this gives the minimum-norm least-squares solution.
pub fn sparse_least_squares<F>(
    matrix: &CsrMatrix<F>,
    b: &[F],
    config: Option<&SparseQrConfig>,
) -> SparseResult<SparseLeastSquaresResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let m = matrix.rows();
    let n = matrix.cols();

    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    let default_config = SparseQrConfig::default();
    let cfg = config.unwrap_or(&default_config);

    // Factorize
    let qr = sparse_qr(matrix, cfg)?;

    // Compute Q^T * b
    let qt_b = apply_qt(b, &qr.householder_v, &qr.tau, m)?;

    let rank = qr.rank;
    if rank == 0 {
        return Ok(SparseLeastSquaresResult {
            solution: vec![F::sparse_zero(); n],
            residual_norm: vector_norm(b),
            rank: 0,
        });
    }

    // Back-substitution: R(1:rank, 1:rank) * y(1:rank) = (Q^T * b)(1:rank)
    let mut y = vec![F::sparse_zero(); n];
    for i in (0..rank).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..rank {
            sum -= qr.r_data[i][j] * y[j];
        }
        let diag = qr.r_data[i][i];
        if diag.abs() < F::epsilon() {
            return Err(SparseError::SingularMatrix(format!(
                "Zero diagonal in R at position {i}"
            )));
        }
        y[i] = sum / diag;
    }

    // Apply column permutation: x[col_perm[j]] = y[j]
    let mut x = vec![F::sparse_zero(); n];
    for j in 0..n {
        x[qr.col_perm[j]] = y[j];
    }

    // Compute residual norm: ||qt_b[rank:]||
    let residual_norm = if rank < m {
        let mut sum_sq = F::sparse_zero();
        for i in rank..m {
            sum_sq += qt_b[i] * qt_b[i];
        }
        sum_sq.sqrt()
    } else {
        F::sparse_zero()
    };

    Ok(SparseLeastSquaresResult {
        solution: x,
        residual_norm,
        rank,
    })
}

// ---------------------------------------------------------------------------
// Q operations
// ---------------------------------------------------------------------------

/// Apply Q^T to a vector b: result = Q^T * b.
///
/// Q is represented as a product of Householder reflections:
/// Q = H_0 * H_1 * ... * H_{k-1}
/// Q^T = H_{k-1} * ... * H_1 * H_0
///
/// But since each H_i is its own inverse (H_i^T = H_i), we just apply them
/// in forward order.
pub fn apply_qt<F>(b: &[F], householder_v: &[Vec<F>], tau: &[F], m: usize) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    let mut result = b.to_vec();
    let k = householder_v.len();

    for i in 0..k {
        if tau[i] == F::sparse_zero() {
            continue;
        }
        let v = &householder_v[i];
        let dot: F = (0..m).map(|row| v[row] * result[row]).sum();
        let scale = tau[i] * dot;
        for row in 0..m {
            result[row] -= scale * v[row];
        }
    }

    Ok(result)
}

/// Apply Q to a vector b: result = Q * b.
///
/// Applies Householder reflections in reverse order.
pub fn apply_q<F>(b: &[F], householder_v: &[Vec<F>], tau: &[F], m: usize) -> SparseResult<Vec<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    let mut result = b.to_vec();
    let k = householder_v.len();

    for i in (0..k).rev() {
        if tau[i] == F::sparse_zero() {
            continue;
        }
        let v = &householder_v[i];
        let dot: F = (0..m).map(|row| v[row] * result[row]).sum();
        let scale = tau[i] * dot;
        for row in 0..m {
            result[row] -= scale * v[row];
        }
    }

    Ok(result)
}

/// Extract the explicit dense Q matrix (m x rank) from Householder vectors.
///
/// This is the economy-size Q factor.
pub fn extract_q_dense<F>(
    householder_v: &[Vec<F>],
    tau: &[F],
    m: usize,
    rank: usize,
) -> SparseResult<Vec<Vec<F>>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let mut q = vec![vec![F::sparse_zero(); rank]; m];

    // Build Q column by column: Q[:, j] = Q * e_j
    for j in 0..rank {
        let mut ej = vec![F::sparse_zero(); m];
        if j < m {
            ej[j] = F::sparse_one();
        }
        let col = apply_q(&ej, householder_v, tau, m)?;
        for i in 0..m {
            q[i][j] = col[i];
        }
    }

    Ok(q)
}

// ---------------------------------------------------------------------------
// Rank-revealing QR analysis
// ---------------------------------------------------------------------------

/// Determine the numerical rank of a sparse matrix using QR with column pivoting.
///
/// Returns the rank (number of R diagonal elements above the tolerance).
pub fn numerical_rank<F>(matrix: &CsrMatrix<F>, tol: Option<f64>) -> SparseResult<usize>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let config = SparseQrConfig {
        pivoting: true,
        rank_tol: tol.unwrap_or(1e-12),
        economy: true,
    };
    let qr = sparse_qr(matrix, &config)?;
    Ok(qr.rank)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract columns of a CSR matrix into dense column-major storage.
fn extract_columns<F>(matrix: &CsrMatrix<F>, m: usize, n: usize) -> Vec<Vec<F>>
where
    F: Float + SparseElement + Debug + 'static,
{
    let mut cols = vec![vec![F::sparse_zero(); m]; n];
    for i in 0..m {
        let start = matrix.indptr[i];
        let end = matrix.indptr[i + 1];
        for idx in start..end {
            let j = matrix.indices[idx];
            cols[j][i] = matrix.data[idx];
        }
    }
    cols
}

/// Compute the Householder vector for a column, zeroing entries below position k.
///
/// Given x = col[k..m], computes v and tau such that H*x = ||x|| e_1
/// where H = I - tau * v * v^T.
///
/// Returns (v, tau) where v is length m (entries 0..k are zero).
fn householder_vector<F>(col: &[F], k: usize, m: usize) -> (Vec<F>, F)
where
    F: Float + NumAssign + SparseElement + Debug + 'static,
{
    let mut v = vec![F::sparse_zero(); m];

    // Compute norm of col[k..m]
    let mut sigma_sq = F::sparse_zero();
    for i in k..m {
        sigma_sq += col[i] * col[i];
    }
    let sigma = sigma_sq.sqrt();

    if sigma < F::epsilon() {
        return (v, F::sparse_zero());
    }

    // Choose sign to avoid cancellation
    let alpha = if col[k] >= F::sparse_zero() {
        -sigma
    } else {
        sigma
    };

    // v[k] = col[k] - alpha
    v[k] = col[k] - alpha;
    v[(k + 1)..m].copy_from_slice(&col[(k + 1)..m]);

    // Normalize v
    let mut v_norm_sq = F::sparse_zero();
    for i in k..m {
        v_norm_sq += v[i] * v[i];
    }

    if v_norm_sq < F::epsilon() {
        return (v, F::sparse_zero());
    }

    let tau = F::from(2.0).unwrap_or_else(|| F::sparse_one() + F::sparse_one()) / v_norm_sq;

    (v, tau)
}

/// Dot product of two vectors from index `start` to their lengths.
fn dot_from<F: Float + SparseElement>(m: usize, a: &[F], b: &[F], start: usize) -> F {
    let mut sum = F::sparse_zero();
    for i in start..m {
        sum = sum + a[i] * b[i];
    }
    sum
}

/// Compute the 2-norm of a full vector.
fn column_norm<F: Float + SparseElement>(col: &[F]) -> F {
    let mut sum_sq = F::sparse_zero();
    for &v in col {
        sum_sq = sum_sq + v * v;
    }
    sum_sq.sqrt()
}

/// Compute the 2-norm of col[start..end].
fn column_norm_from<F: Float + SparseElement>(col: &[F], start: usize, end: usize) -> F {
    let mut sum_sq = F::sparse_zero();
    for i in start..end {
        sum_sq = sum_sq + col[i] * col[i];
    }
    sum_sq.sqrt()
}

/// Compute the 2-norm of a vector.
fn vector_norm<F: Float + SparseElement>(v: &[F]) -> F {
    let mut sum_sq = F::sparse_zero();
    for &x in v {
        sum_sq = sum_sq + x * x;
    }
    sum_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a 3x3 well-conditioned matrix
    fn create_3x3() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed")
    }

    /// Create a 4x2 overdetermined system
    fn create_overdetermined() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        CsrMatrix::new(data, rows, cols, (4, 2)).expect("Failed")
    }

    /// Create a rank-deficient 3x3 matrix
    fn create_rank_deficient() -> CsrMatrix<f64> {
        // Row 2 = Row 0 + Row 1
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed")
    }

    #[test]
    fn test_sparse_qr_basic() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");
        assert_eq!(qr.m, 3);
        assert_eq!(qr.n, 3);
        assert_eq!(qr.rank, 3);
        assert_eq!(qr.householder_v.len(), 3);
        assert_eq!(qr.r_data.len(), 3);
    }

    #[test]
    fn test_sparse_qr_empty() {
        let mat = CsrMatrix::<f64>::new(vec![], vec![], vec![], (0, 0)).expect("Failed");
        let config = SparseQrConfig::default();
        let qr = sparse_qr(&mat, &config).expect("QR on empty failed");
        assert_eq!(qr.rank, 0);
    }

    #[test]
    fn test_qr_orthogonality() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");
        let q = extract_q_dense(&qr.householder_v, &qr.tau, 3, 3).expect("Q extraction failed");

        // Check Q^T * Q ≈ I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[k][i] * q[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T Q[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_qr_factorization_accuracy() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");
        let q = extract_q_dense(&qr.householder_v, &qr.tau, 3, 3).expect("Q extraction failed");

        // Check Q * R ≈ A * P
        let dense = mat.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let mut qr_val = 0.0;
                for k in 0..3 {
                    qr_val += q[i][k] * qr.r_data[k][j];
                }
                let orig_col = qr.col_perm[j];
                let a_val = dense[i][orig_col];
                assert!(
                    (qr_val - a_val).abs() < 1e-10,
                    "QR[{i},{j}] = {qr_val}, A*P[{i},{j}] = {a_val}"
                );
            }
        }
    }

    #[test]
    fn test_least_squares_square() {
        let mat = create_3x3();
        // A = [[1,2,3],[4,5,6],[7,8,10]]
        // b = [6, 15, 25] => check residual ~ 0
        let b = vec![6.0, 15.0, 25.0];
        let result = sparse_least_squares(&mat, &b, None).expect("LS failed");
        assert_eq!(result.solution.len(), 3);
        assert_eq!(result.rank, 3);

        // Verify Ax ≈ b
        let dense = mat.to_dense();
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += dense[i][j] * result.solution[j];
            }
            assert!(
                (sum - b[i]).abs() < 1e-8,
                "Row {i}: residual {}",
                (sum - b[i]).abs()
            );
        }
    }

    #[test]
    fn test_least_squares_overdetermined() {
        let mat = create_overdetermined();
        // A = [[1,0],[1,1],[0,1],[1,1]]
        // b = [1, 2, 1, 2] => least squares
        let b = vec![1.0, 2.0, 1.0, 2.0];
        let result = sparse_least_squares(&mat, &b, None).expect("LS overdetermined failed");
        assert_eq!(result.solution.len(), 2);
        assert_eq!(result.rank, 2);

        // Solution should minimize ||Ax - b||
        // Verify by checking normal equations: A^T A x ≈ A^T b
        let dense = mat.to_dense();
        let mut ata = vec![vec![0.0; 2]; 2];
        let mut atb = [0.0; 2];
        for i in 0..4 {
            for j in 0..2 {
                atb[j] += dense[i][j] * b[i];
                for k in 0..2 {
                    ata[j][k] += dense[i][j] * dense[i][k];
                }
            }
        }
        for j in 0..2 {
            let mut sum = 0.0;
            for k in 0..2 {
                sum += ata[j][k] * result.solution[k];
            }
            assert!(
                (sum - atb[j]).abs() < 1e-8,
                "Normal eq {j}: {sum} vs {}",
                atb[j]
            );
        }
    }

    #[test]
    fn test_least_squares_dimension_mismatch() {
        let mat = create_3x3();
        let result = sparse_least_squares(&mat, &[1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pivoted_qr_rank_deficient() {
        let mat = create_rank_deficient();
        let config = SparseQrConfig {
            pivoting: true,
            rank_tol: 1e-10,
            economy: true,
        };
        let qr = sparse_qr(&mat, &config).expect("QR rank deficient failed");
        assert!(qr.rank <= 2, "Expected rank <= 2, got {}", qr.rank);
    }

    #[test]
    fn test_numerical_rank() {
        let mat = create_3x3();
        let rank = numerical_rank(&mat, None).expect("Rank computation failed");
        assert_eq!(rank, 3);

        let mat2 = create_rank_deficient();
        let rank2 = numerical_rank(&mat2, Some(1e-10)).expect("Rank computation failed");
        assert!(rank2 <= 2, "Expected rank <= 2, got {rank2}");
    }

    #[test]
    fn test_apply_q_qt_inverse() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");

        let b = vec![1.0, 2.0, 3.0];
        let qt_b = apply_qt(&b, &qr.householder_v, &qr.tau, 3).expect("Q^T failed");
        let q_qt_b = apply_q(&qt_b, &qr.householder_v, &qr.tau, 3).expect("Q failed");

        // Q * Q^T * b ≈ b (since Q is orthogonal)
        for i in 0..3 {
            assert!(
                (q_qt_b[i] - b[i]).abs() < 1e-10,
                "Q*Q^T*b[{i}] = {}, expected {}",
                q_qt_b[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_qr_r_upper_triangular() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");

        // Check R is upper triangular
        for i in 0..qr.r_data.len() {
            for j in 0..i {
                assert!(
                    qr.r_data[i][j].abs() < 1e-10,
                    "R[{i},{j}] = {} should be zero",
                    qr.r_data[i][j]
                );
            }
        }
    }

    #[test]
    fn test_sparse_qr_single_element() {
        let mat = CsrMatrix::new(vec![5.0], vec![0], vec![0], (1, 1)).expect("Failed");
        let config = SparseQrConfig::default();
        let qr = sparse_qr(&mat, &config).expect("QR single failed");
        assert_eq!(qr.rank, 1);
        assert!((qr.r_data[0][0].abs() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_least_squares_identity() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let data = vec![1.0, 1.0, 1.0];
        let mat = CsrMatrix::new(data, rows, cols, (3, 3)).expect("Failed");
        let b = vec![1.0, 2.0, 3.0];
        let result = sparse_least_squares(&mat, &b, None).expect("LS identity failed");
        for i in 0..3 {
            assert!(
                (result.solution[i] - b[i]).abs() < 1e-10,
                "x[{i}] = {}, expected {}",
                result.solution[i],
                b[i]
            );
        }
        assert!(result.residual_norm < 1e-10);
    }

    #[test]
    fn test_least_squares_tall_skinny() {
        // 5x2 matrix: simple least-squares regression
        let rows = vec![0, 1, 1, 2, 2, 3, 3, 4, 4];
        let cols = vec![0, 0, 1, 0, 1, 0, 1, 0, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        let mat = CsrMatrix::new(data, rows, cols, (5, 2)).expect("Failed");
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sparse_least_squares(&mat, &b, None).expect("LS tall failed");
        assert_eq!(result.solution.len(), 2);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn test_householder_vector_zero() {
        let col = vec![0.0, 0.0, 0.0];
        let (v, tau) = householder_vector(&col, 0, 3);
        assert!((tau).abs() < 1e-15);
        for &vi in &v {
            assert!((vi).abs() < 1e-15);
        }
    }

    #[test]
    fn test_col_perm_valid() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");
        let mut sorted = qr.col_perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_extract_q_dense_columns_orthonormal() {
        let mat = create_3x3();
        let config = SparseQrConfig {
            pivoting: false,
            economy: true,
            ..Default::default()
        };
        let qr = sparse_qr(&mat, &config).expect("QR failed");
        let q = extract_q_dense(&qr.householder_v, &qr.tau, 3, 3).expect("Q failed");

        // Each column should have unit norm
        for j in 0..3 {
            let mut norm_sq = 0.0;
            for i in 0..3 {
                norm_sq += q[i][j] * q[i][j];
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-10,
                "Column {j} norm = {}",
                norm_sq.sqrt()
            );
        }
    }
}
