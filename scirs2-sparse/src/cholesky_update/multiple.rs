//! Multiple rank-1 and low-rank Cholesky updates.
//!
//! Provides convenience functions that apply several rank-1 modifications
//! in sequence, or a single low-rank modification A' = A + W D W^T.

use crate::error::{SparseError, SparseResult};

use super::downdate::cholesky_rank1_downdate_with_config;
use super::rank1::cholesky_rank1_update_with_config;
use super::types::CholUpdateConfig;

/// Apply multiple rank-1 updates and/or downdates to a Cholesky factor.
///
/// Each entry in `vectors` is paired with the corresponding entry in
/// `weights`.  A positive weight produces an update (A + w_i v_i v_i^T),
/// and a negative weight produces a downdate (A − |w_i| v_i v_i^T).
///
/// The modifications are applied sequentially in the order given, so the
/// result is `L' L'^T = L L^T + sum_i w_i v_i v_i^T`.
///
/// # Arguments
///
/// * `l`       — Lower-triangular Cholesky factor (n × n dense).
/// * `vectors` — Slice of update/downdate vectors, each of length n.
/// * `weights` — Scalar weight for each vector.
///
/// # Errors
///
/// * `SparseError::DimensionMismatch` if `vectors.len() != weights.len()`.
/// * Any error from an individual rank-1 update or downdate.
///
/// # Example
///
/// ```
/// use scirs2_sparse::cholesky_update::{cholesky_factorize, cholesky_rank_k_update};
///
/// let a = vec![
///     vec![10.0, 1.0],
///     vec![ 1.0, 10.0],
/// ];
/// let l = cholesky_factorize(&a).expect("factor");
/// let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let weights = vec![0.5, 0.5];
///
/// let l_new = cholesky_rank_k_update(&l, &vecs, &weights).expect("k-update");
/// ```
pub fn cholesky_rank_k_update(
    l: &[Vec<f64>],
    vectors: &[Vec<f64>],
    weights: &[f64],
) -> SparseResult<Vec<Vec<f64>>> {
    cholesky_rank_k_update_with_config(l, vectors, weights, &CholUpdateConfig::default())
}

/// Multiple rank-1 updates/downdates with explicit configuration.
///
/// See [`cholesky_rank_k_update`] for details.
pub fn cholesky_rank_k_update_with_config(
    l: &[Vec<f64>],
    vectors: &[Vec<f64>],
    weights: &[f64],
    config: &CholUpdateConfig,
) -> SparseResult<Vec<Vec<f64>>> {
    if vectors.len() != weights.len() {
        return Err(SparseError::DimensionMismatch {
            expected: vectors.len(),
            found: weights.len(),
        });
    }

    let n = l.len();
    let mut current = l.to_vec();

    for (idx, (v, &w)) in vectors.iter().zip(weights.iter()).enumerate() {
        if v.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: v.len(),
            });
        }

        if w.abs() < config.tolerance {
            // Near-zero weight — skip
            continue;
        }

        if w > 0.0 {
            current = cholesky_rank1_update_with_config(&current, v, w, config).map_err(|e| {
                SparseError::ComputationError(format!(
                    "Rank-1 update {} (weight={:.6e}) failed: {}",
                    idx, w, e
                ))
            })?;
        } else {
            current =
                cholesky_rank1_downdate_with_config(&current, v, -w, config).map_err(|e| {
                    SparseError::ComputationError(format!(
                        "Rank-1 downdate {} (weight={:.6e}) failed: {}",
                        idx, w, e
                    ))
                })?;
        }
    }

    Ok(current)
}

/// Low-rank Cholesky update: A' = A + W D W^T.
///
/// `w` is a collection of column vectors (each of length n) and `d` holds
/// the diagonal entries of the diagonal matrix D.  A positive `d[i]`
/// produces an update and a negative `d[i]` a downdate, exactly like
/// [`cholesky_rank_k_update`] with `weights = d`.
///
/// This is mathematically identical to calling [`cholesky_rank_k_update`],
/// but the name makes the intention clearer when the modification is
/// specified as W D W^T.
///
/// # Arguments
///
/// * `l` — Lower-triangular Cholesky factor (n × n dense).
/// * `w` — Column vectors of the low-rank term (each of length n).
/// * `d` — Diagonal entries of D (same length as `w`).
///
/// # Errors
///
/// Same as [`cholesky_rank_k_update`].
///
/// # Example
///
/// ```
/// use scirs2_sparse::cholesky_update::{cholesky_factorize, cholesky_low_rank_update};
///
/// let a = vec![
///     vec![10.0, 1.0, 0.5],
///     vec![ 1.0, 10.0, 1.0],
///     vec![ 0.5,  1.0, 10.0],
/// ];
/// let l = cholesky_factorize(&a).expect("factor");
///
/// // W = [[1, 0], [0, 1], [0, 0]], D = diag(0.5, 0.3)
/// let w = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
/// let d = vec![0.5, 0.3];
///
/// let l_new = cholesky_low_rank_update(&l, &w, &d).expect("low-rank update");
/// ```
pub fn cholesky_low_rank_update(
    l: &[Vec<f64>],
    w: &[Vec<f64>],
    d: &[f64],
) -> SparseResult<Vec<Vec<f64>>> {
    cholesky_rank_k_update_with_config(l, w, d, &CholUpdateConfig::default())
}

/// Low-rank Cholesky update with explicit configuration.
///
/// See [`cholesky_low_rank_update`] for details.
pub fn cholesky_low_rank_update_with_config(
    l: &[Vec<f64>],
    w: &[Vec<f64>],
    d: &[f64],
    config: &CholUpdateConfig,
) -> SparseResult<Vec<Vec<f64>>> {
    cholesky_rank_k_update_with_config(l, w, d, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cholesky_update::rank1::{cholesky_factorize, cholesky_rank1_update, llt};

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
    fn test_rank_k_update_equals_sequential() {
        let a = vec![
            vec![10.0, 1.0, 0.5],
            vec![1.0, 10.0, 1.0],
            vec![0.5, 1.0, 10.0],
        ];
        let l = cholesky_factorize(&a).expect("factor");

        let v1 = vec![0.5, 0.3, 0.1];
        let v2 = vec![0.1, 0.4, 0.2];
        let w1 = 1.0;
        let w2 = 0.5;

        // Sequential manual updates
        let l_step1 = cholesky_rank1_update(&l, &v1, w1).expect("step1");
        let l_step2 = cholesky_rank1_update(&l_step1, &v2, w2).expect("step2");

        // Batch
        let l_batch = cholesky_rank_k_update(&l, &[v1, v2], &[w1, w2]).expect("batch");

        let a_seq = llt(&l_step2);
        let a_batch = llt(&l_batch);
        assert_matrices_close(&a_seq, &a_batch, 1e-12);
    }

    #[test]
    fn test_rank_k_mixed_update_downdate() {
        let a = vec![vec![20.0, 1.0], vec![1.0, 20.0]];
        let l = cholesky_factorize(&a).expect("factor");

        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 0.3]; // downdate

        let l_new =
            cholesky_rank_k_update(&l, &[v1.clone(), v2.clone()], &[2.0, -1.0]).expect("mixed");
        let a_new = llt(&l_new);

        let n = 2;
        let mut a_expected = a.clone();
        for i in 0..n {
            for j in 0..n {
                a_expected[i][j] += 2.0 * v1[i] * v1[j];
                a_expected[i][j] -= 1.0 * v2[i] * v2[j];
            }
        }
        assert_matrices_close(&a_expected, &a_new, 1e-10);
    }

    #[test]
    fn test_low_rank_update_wdwt() {
        let a = vec![
            vec![10.0, 1.0, 0.5],
            vec![1.0, 10.0, 1.0],
            vec![0.5, 1.0, 10.0],
        ];
        let l = cholesky_factorize(&a).expect("factor");

        let w1 = vec![1.0, 0.0, 0.0];
        let w2 = vec![0.0, 1.0, 0.0];
        let d = vec![0.5, 0.3];

        let l_new = cholesky_low_rank_update(&l, &[w1.clone(), w2.clone()], &d).expect("low-rank");
        let a_new = llt(&l_new);

        let n = 3;
        let mut a_expected = a.clone();
        for i in 0..n {
            for j in 0..n {
                a_expected[i][j] += d[0] * w1[i] * w1[j];
                a_expected[i][j] += d[1] * w2[i] * w2[j];
            }
        }
        assert_matrices_close(&a_expected, &a_new, 1e-10);
    }

    #[test]
    fn test_rank_k_dimension_mismatch() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0]];
        let w = vec![1.0, 2.0]; // length mismatch with v
        assert!(cholesky_rank_k_update(&l, &v, &w).is_err());
    }

    #[test]
    fn test_rank_k_vector_dimension_mismatch() {
        let l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0, 0.0]]; // vector too long
        let w = vec![1.0];
        assert!(cholesky_rank_k_update(&l, &v, &w).is_err());
    }

    #[test]
    fn test_rank_k_skip_zero_weight() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 4.0]];
        let l = cholesky_factorize(&a).expect("factor");
        let v = vec![vec![1.0, 0.0]];
        let w = vec![0.0]; // zero weight => no change

        let l_new = cholesky_rank_k_update(&l, &v, &w).expect("skip");
        let a_new = llt(&l_new);
        assert_matrices_close(&a, &a_new, 1e-14);
    }

    #[test]
    fn test_round_trip_update_then_downdate() {
        let a = vec![
            vec![10.0, 2.0, 1.0],
            vec![2.0, 10.0, 3.0],
            vec![1.0, 3.0, 10.0],
        ];
        let l = cholesky_factorize(&a).expect("factor");
        let u = vec![0.5, 0.3, 0.1];
        let alpha = 1.0;

        // Update then downdate should recover original
        let l_up = cholesky_rank1_update(&l, &u, alpha).expect("update");
        let l_back = cholesky_rank_k_update(&l_up, &[u], &[-alpha]).expect("downdate back");

        let a_recovered = llt(&l_back);
        assert_matrices_close(&a, &a_recovered, 1e-9);
    }
}
