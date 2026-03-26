//! Rank-1 Cholesky update via Givens rotations.
//!
//! Given a lower-triangular Cholesky factor L such that `L L^T = A`, this
//! module computes the updated factor L' satisfying
//! `L' L'^T = A + alpha * u * u^T`
//! using the method of Gill, Golub, Murray, and Saunders (1974).
//!
//! # Algorithm
//!
//! 1. Solve `L w = sqrt(alpha) * u` (forward substitution).
//! 2. Walk w from bottom to top, applying Givens rotations to zero out
//!    each element while updating the corresponding column of L.
//!
//! This is O(n²) — the same cost as a dense Cholesky factorization — but
//! avoids re-factorizing from scratch.
//!
//! # References
//!
//! - Gill, P.E., Golub, G.H., Murray, W., and Saunders, M.A. (1974).
//!   "Methods for modifying matrix factorizations."
//!   *Mathematics of Computation*, 28(126), 505–535.

use crate::error::{SparseError, SparseResult};

use super::types::{estimate_condition, CholUpdateConfig, CholUpdateResult};

/// Perform a dense Cholesky factorization of a symmetric positive-definite
/// matrix A, returning the lower-triangular factor L such that L L^T = A.
///
/// This is a straightforward column-by-column implementation used mainly
/// for testing and as a reference.  For production sparse matrices the
/// sparse Cholesky in `linalg::decomposition` should be preferred.
///
/// # Errors
///
/// Returns [`SparseError::ComputationError`] if A is not positive definite.
///
/// # Example
///
/// ```
/// use scirs2_sparse::cholesky_update::cholesky_factorize;
///
/// let a = vec![
///     vec![4.0, 2.0],
///     vec![2.0, 5.0],
/// ];
/// let l = cholesky_factorize(&a).expect("factorize");
/// // l is lower triangular: l[0][1] == 0
/// assert!((l[0][1]).abs() < 1e-14);
/// ```
pub fn cholesky_factorize(a: &[Vec<f64>]) -> SparseResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Validate square matrix
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(SparseError::ComputationError(format!(
                "Row {} has length {} but matrix dimension is {}",
                i,
                row.len(),
                n
            )));
        }
    }

    let mut l = vec![vec![0.0; n]; n];

    for j in 0..n {
        // Diagonal element
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag <= 0.0 {
            return Err(SparseError::ComputationError(format!(
                "Matrix is not positive definite: diagonal element at ({},{}) is {:.6e}",
                j, j, diag
            )));
        }
        l[j][j] = diag.sqrt();

        // Off-diagonal elements in column j
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - s) / l[j][j];
        }
    }

    Ok(l)
}

/// Compute the rank-1 Cholesky update L' such that
/// `L' L'^T = L L^T + alpha * u * u^T`
/// where L is a lower-triangular Cholesky factor, u is a vector, and alpha > 0.
///
/// The algorithm uses Givens rotations following Gill–Golub–Murray–Saunders.
///
/// # Arguments
///
/// * `l`     — Lower-triangular factor (n × n, row-major dense).
/// * `u`     — Update vector of length n.
/// * `alpha` — Positive scalar weight.
///
/// # Errors
///
/// * `SparseError::ValueError` if α ≤ 0 or dimensions mismatch.
/// * `SparseError::ComputationError` if numerical breakdown occurs.
///
/// # Example
///
/// ```
/// use scirs2_sparse::cholesky_update::{cholesky_factorize, cholesky_rank1_update};
///
/// let a = vec![
///     vec![4.0, 2.0],
///     vec![2.0, 5.0],
/// ];
/// let l = cholesky_factorize(&a).expect("factor");
/// let u = vec![1.0, 0.5];
/// let l_new = cholesky_rank1_update(&l, &u, 1.0).expect("update");
/// // Verify: l_new * l_new^T ≈ a + u*u^T
/// let n = 2;
/// for i in 0..n {
///     for j in 0..n {
///         let mut val = 0.0;
///         for k in 0..n {
///             val += l_new[i][k] * l_new[j][k];
///         }
///         let expected = a[i][j] + u[i] * u[j];
///         assert!((val - expected).abs() < 1e-10);
///     }
/// }
/// ```
pub fn cholesky_rank1_update(l: &[Vec<f64>], u: &[f64], alpha: f64) -> SparseResult<Vec<Vec<f64>>> {
    cholesky_rank1_update_with_config(l, u, alpha, &CholUpdateConfig::default())
}

/// Rank-1 Cholesky update with explicit configuration.
///
/// See [`cholesky_rank1_update`] for the mathematical details.
pub fn cholesky_rank1_update_with_config(
    l: &[Vec<f64>],
    u: &[f64],
    alpha: f64,
    config: &CholUpdateConfig,
) -> SparseResult<Vec<Vec<f64>>> {
    let n = l.len();

    // ---- Validation --------------------------------------------------------
    if alpha <= 0.0 {
        return Err(SparseError::ValueError(
            "alpha must be positive for a rank-1 update".into(),
        ));
    }
    if u.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: u.len(),
        });
    }
    for (i, row) in l.iter().enumerate() {
        if row.len() != n {
            return Err(SparseError::ComputationError(format!(
                "Factor row {} has length {} but expected {}",
                i,
                row.len(),
                n
            )));
        }
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    // ---- Gill–Golub–Murray–Saunders method ---------------------------------
    //
    // We maintain a working copy of L and a vector w = √α u.
    // For each column j = 0..n-1 we apply a Givens rotation that
    // incorporates w[j] into L[j][j], then propagate the effect through
    // the remaining elements of column j.

    let mut l_new = l.to_vec();
    let sqrt_alpha = alpha.sqrt();
    let mut w: Vec<f64> = u.iter().map(|&v| sqrt_alpha * v).collect();

    for j in 0..n {
        // Current diagonal element and the component to incorporate
        let lj = l_new[j][j];
        let wj = w[j];

        // Compute the updated diagonal via Givens rotation
        let r = (lj * lj + wj * wj).sqrt();
        if r < config.tolerance {
            return Err(SparseError::ComputationError(format!(
                "Numerical breakdown at column {}: r = {:.6e}",
                j, r
            )));
        }
        let c = lj / r; // cosine
        let s = wj / r; // sine

        l_new[j][j] = r;

        // Update the remaining rows in column j
        for i in (j + 1)..n {
            let li = l_new[i][j];
            let wi = w[i];
            l_new[i][j] = c * li + s * wi;
            w[i] = -s * li + c * wi;
        }
    }

    // ---- Post-checks -------------------------------------------------------
    if config.check_positive_definite {
        for j in 0..n {
            if l_new[j][j] <= config.tolerance {
                return Err(SparseError::ComputationError(format!(
                    "Updated factor is not positive definite at diagonal ({},{}): {:.6e}",
                    j, j, l_new[j][j]
                )));
            }
        }
    }

    Ok(l_new)
}

/// Rank-1 update returning a full [`CholUpdateResult`].
///
/// Identical to [`cholesky_rank1_update_with_config`] but wraps the output in
/// a result struct that also carries a condition estimate.
pub fn cholesky_rank1_update_result(
    l: &[Vec<f64>],
    u: &[f64],
    alpha: f64,
    config: &CholUpdateConfig,
) -> CholUpdateResult {
    match cholesky_rank1_update_with_config(l, u, alpha, config) {
        Ok(factor) => {
            let cond = estimate_condition(&factor, config.tolerance);
            CholUpdateResult {
                factor,
                successful: true,
                condition_estimate: cond,
            }
        }
        Err(_) => CholUpdateResult {
            factor: l.to_vec(),
            successful: false,
            condition_estimate: f64::INFINITY,
        },
    }
}

/// Multiply L L^T and return the dense symmetric matrix.
///
/// Utility used in tests; not part of the public API.
pub(crate) fn llt(l: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = l.len();
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += l[i][k] * l[j][k];
            }
            a[i][j] = s;
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small SPD matrix.
    fn spd_3x3() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let a = vec![
            vec![4.0, 2.0, 1.0],
            vec![2.0, 5.0, 3.0],
            vec![1.0, 3.0, 6.0],
        ];
        let l = cholesky_factorize(&a).expect("factorize");
        (a, l)
    }

    fn assert_matrices_close(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].len(), b[i].len());
            for j in 0..a[i].len() {
                assert!(
                    (a[i][j] - b[i][j]).abs() < tol,
                    "Mismatch at ({},{}): {} vs {} (diff {})",
                    i,
                    j,
                    a[i][j],
                    b[i][j],
                    (a[i][j] - b[i][j]).abs()
                );
            }
        }
    }

    #[test]
    fn test_cholesky_factorize_basic() {
        let (a, l) = spd_3x3();
        let reconstructed = llt(&l);
        assert_matrices_close(&a, &reconstructed, 1e-12);
    }

    #[test]
    fn test_factorize_not_spd() {
        let a = vec![vec![1.0, 0.0], vec![0.0, -1.0]];
        assert!(cholesky_factorize(&a).is_err());
    }

    #[test]
    fn test_rank1_update_correctness() {
        let (a, l) = spd_3x3();
        let u = vec![1.0, 0.5, -0.3];
        let alpha = 2.0;

        let l_new = cholesky_rank1_update(&l, &u, alpha).expect("update");

        // Expected: A' = A + alpha * u * u^T
        let n = a.len();
        let mut a_prime = a.clone();
        for i in 0..n {
            for j in 0..n {
                a_prime[i][j] += alpha * u[i] * u[j];
            }
        }

        let reconstructed = llt(&l_new);
        assert_matrices_close(&a_prime, &reconstructed, 1e-10);
    }

    #[test]
    fn test_rank1_update_identity() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u = vec![0.0, 1.0];
        let alpha = 1.0;

        let l_new = cholesky_rank1_update(&l, &u, alpha).expect("update");
        let a_new = llt(&l_new);
        // Expected: I + e2*e2^T = diag(1, 2)
        assert!((a_new[0][0] - 1.0).abs() < 1e-12);
        assert!((a_new[1][1] - 2.0).abs() < 1e-12);
        assert!((a_new[0][1]).abs() < 1e-12);
    }

    #[test]
    fn test_rank1_update_preserves_lower_triangular() {
        let (_, l) = spd_3x3();
        let u = vec![1.0, 2.0, 3.0];
        let l_new = cholesky_rank1_update(&l, &u, 1.0).expect("update");

        for i in 0..l_new.len() {
            for j in (i + 1)..l_new[i].len() {
                assert!(
                    l_new[i][j].abs() < 1e-14,
                    "Non-zero upper triangle at ({},{}): {}",
                    i,
                    j,
                    l_new[i][j]
                );
            }
        }
    }

    #[test]
    fn test_rank1_update_dimension_mismatch() {
        let l = vec![vec![1.0, 0.0], vec![0.5, 1.0]];
        let u = vec![1.0, 2.0, 3.0]; // wrong length
        assert!(cholesky_rank1_update(&l, &u, 1.0).is_err());
    }

    #[test]
    fn test_rank1_update_negative_alpha() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u = vec![1.0, 0.0];
        assert!(cholesky_rank1_update(&l, &u, -1.0).is_err());
    }

    #[test]
    fn test_rank1_update_result_struct() {
        let (_, l) = spd_3x3();
        let u = vec![0.5, 0.5, 0.5];
        let res = cholesky_rank1_update_result(&l, &u, 1.0, &CholUpdateConfig::default());
        assert!(res.successful);
        assert!(res.condition_estimate > 0.0);
        assert!(res.condition_estimate < 1e6);
    }

    #[test]
    fn test_rank1_update_large_20x20() {
        // Build a 20x20 diagonally dominant SPD matrix
        let n = 20;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = (n as f64) + 1.0;
            for j in 0..n {
                if i != j {
                    a[i][j] = 1.0 / ((1 + i).abs_diff(1 + j) as f64 + 1.0);
                    a[j][i] = a[i][j];
                }
            }
        }

        let l = cholesky_factorize(&a).expect("factorize 20x20");

        let u: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.1).collect();
        let alpha = 0.5;

        let l_new = cholesky_rank1_update(&l, &u, alpha).expect("update 20x20");

        let mut a_prime = a.clone();
        for i in 0..n {
            for j in 0..n {
                a_prime[i][j] += alpha * u[i] * u[j];
            }
        }

        let reconstructed = llt(&l_new);
        assert_matrices_close(&a_prime, &reconstructed, 1e-8);
    }
}
