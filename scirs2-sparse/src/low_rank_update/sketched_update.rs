//! Sketched / randomized low-rank updates (Nystrom, randomized range).
//!
//! This module provides approximate factorization updates using randomized
//! linear algebra techniques, which trade exactness for computational
//! efficiency on large matrices.
//!
//! # References
//!
//! - Halko, Martinsson, Tropp (2011). "Finding structure with randomness:
//!   Probabilistic algorithms for constructing approximate matrix decompositions."
//!   *SIAM Review* 53(2), 217-288.

use crate::error::{SparseError, SparseResult};

use super::types::SketchConfig;

/// Type alias for a pair of dense matrices (e.g., Q and R factors).
type DenseMatrixPair = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Simple linear congruential generator (no external deps).
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate approximate standard normal via Box-Muller transform.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
            let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
            if u1 > 1e-15 {
                let r = (-2.0 * u1.ln()).sqrt();
                return r * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }
}

/// Dense matrix-vector multiply y = A * x.
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi).sum())
        .collect()
}

/// Dense matrix-matrix multiply C = A * B.
fn mat_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let p = b.first().map_or(0, |r| r.len());
    let k = b.len();
    let mut c = vec![vec![0.0; p]; m];
    for i in 0..m {
        for j in 0..p {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i][t] * b[t][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Transpose a dense matrix.
fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut at = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            at[j][i] = a[i][j];
        }
    }
    at
}

/// Thin QR factorization using modified Gram-Schmidt.
/// Returns (Q, R) where Q is m x k and R is k x k, with k = min(m, n).
fn thin_qr(a: &[Vec<f64>]) -> SparseResult<DenseMatrixPair> {
    let m = a.len();
    if m == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = a[0].len();
    let k = m.min(n);

    // Work with columns
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();

    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut r = vec![vec![0.0; n]; k];

    for j in 0..k {
        let mut v = cols[j].clone();

        // Orthogonalize against previous columns
        for (qi, q_col) in q_cols.iter().enumerate() {
            let dot: f64 = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[qi][j] = dot;
            for (vi, &qi_val) in v.iter_mut().zip(q_col.iter()) {
                *vi -= dot * qi_val;
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            r[j][j] = 0.0;
            q_cols.push(vec![0.0; m]);
        } else {
            r[j][j] = norm;
            for vi in &mut v {
                *vi /= norm;
            }
            q_cols.push(v);
        }

        // Project remaining columns
        for jj in (j + 1)..n {
            let dot: f64 = cols[jj]
                .iter()
                .zip(q_cols[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            r[j][jj] = dot;
            for (ci, &qi) in cols[jj].iter_mut().zip(q_cols[j].iter()) {
                *ci -= dot * qi;
            }
        }
    }

    // Convert Q from column-major to row-major
    let mut q = vec![vec![0.0; k]; m];
    for i in 0..m {
        for j in 0..k {
            q[i][j] = q_cols[j][i];
        }
    }

    // Truncate R to k x n
    let r_out: Vec<Vec<f64>> = r.into_iter().take(k).collect();

    Ok((q, r_out))
}

/// Approximate A + UV^T using a Nystrom sketch.
///
/// Returns `(Q, B)` where `A + UV^T ≈ Q B` is a low-rank approximation.
///
/// # Arguments
///
/// * `a` - The original matrix (m x n).
/// * `e_u` - Left factor of the perturbation (m x r).
/// * `e_v` - Right factor of the perturbation (n x r), so perturbation = e_u * e_v^T.
/// * `rank` - Desired rank of the approximation.
/// * `config` - Sketch configuration.
///
/// # Errors
///
/// Returns errors on dimension mismatch.
pub fn nystrom_update(
    a: &[Vec<f64>],
    e_u: &[Vec<f64>],
    e_v: &[Vec<f64>],
    rank: usize,
    config: &SketchConfig,
) -> SparseResult<DenseMatrixPair> {
    let m = a.len();
    if m == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let n = a[0].len();

    // Validate e_u is m x r
    if e_u.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: e_u.len(),
        });
    }
    // Validate e_v is n x r
    if e_v.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: e_v.len(),
        });
    }

    let sketch_dim = if config.sketch_dimension == 0 {
        // Auto: use rank + oversampling
        (rank + 5).min(n).min(m)
    } else {
        config.sketch_dimension.min(n).min(m)
    };

    let mut rng = LcgRng::new(config.seed);

    // Generate random sampling matrix Omega (n x sketch_dim)
    let mut omega = vec![vec![0.0; sketch_dim]; n];
    for i in 0..n {
        for j in 0..sketch_dim {
            omega[i][j] = rng.next_normal();
        }
    }

    // Compute Y = (A + e_u * e_v^T) * Omega  (m x sketch_dim)
    // = A * Omega + e_u * (e_v^T * Omega)
    let a_omega = mat_mat_mul(a, &omega);
    let evt = transpose(e_v);
    let evt_omega = mat_mat_mul(&evt, &omega); // r x sketch_dim
    let u_evtomega = mat_mat_mul(e_u, &evt_omega); // m x sketch_dim

    let mut y = vec![vec![0.0; sketch_dim]; m];
    for i in 0..m {
        for j in 0..sketch_dim {
            y[i][j] = a_omega[i][j] + u_evtomega[i][j];
        }
    }

    // QR factorization of Y to get orthonormal basis Q
    let (q, _r) = thin_qr(&y)?;

    // Compute B = Q^T * (A + UV^T)  (sketch_dim x n)
    let qt = transpose(&q);
    let qt_a = mat_mat_mul(&qt, a); // sketch_dim x n
    let qt_u = mat_mat_mul(&qt, e_u); // sketch_dim x r
    let qt_u_evt = mat_mat_mul(&qt_u, &evt); // sketch_dim x n

    let actual_k = q[0].len();
    let mut b = vec![vec![0.0; n]; actual_k];
    for i in 0..actual_k {
        for j in 0..n {
            b[i][j] = qt_a[i][j] + qt_u_evt[i][j];
        }
    }

    Ok((q, b))
}

/// Approximate (A + UV^T) using a randomized range finder.
///
/// Draws a random Gaussian matrix, computes the product, and uses QR
/// to find an approximate range, then projects.
///
/// # Arguments
///
/// * `a` - The original matrix (m x n).
/// * `u_mat` - Left factor of the perturbation (m x r).
/// * `v_mat` - Right factor of the perturbation (n x r), perturbation = u_mat * v_mat^T.
/// * `sketch_dim` - Dimension of the random sketch.
/// * `seed` - Random seed.
///
/// # Errors
///
/// Returns errors on dimension mismatch.
pub fn randomized_low_rank_update(
    a: &[Vec<f64>],
    u_mat: &[Vec<f64>],
    v_mat: &[Vec<f64>],
    sketch_dim: usize,
    seed: u64,
) -> SparseResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Ok(Vec::new());
    }
    let n = a[0].len();

    if u_mat.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: u_mat.len(),
        });
    }
    if v_mat.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: v_mat.len(),
        });
    }

    let k = sketch_dim.min(n).min(m);
    if k == 0 {
        return Ok(vec![vec![0.0; n]; m]);
    }

    let mut rng = LcgRng::new(seed);

    // Draw random Gaussian matrix Omega (n x k)
    let mut omega = vec![vec![0.0; k]; n];
    for i in 0..n {
        for j in 0..k {
            omega[i][j] = rng.next_normal();
        }
    }

    // Compute Y = (A + U V^T) Omega  (m x k)
    let a_omega = mat_mat_mul(a, &omega);
    let vt = transpose(v_mat);
    let vt_omega = mat_mat_mul(&vt, &omega);
    let u_vt_omega = mat_mat_mul(u_mat, &vt_omega);

    let mut y = vec![vec![0.0; k]; m];
    for i in 0..m {
        for j in 0..k {
            y[i][j] = a_omega[i][j] + u_vt_omega[i][j];
        }
    }

    // QR of Y
    let (q, _r) = thin_qr(&y)?;

    // Project: result ≈ Q Q^T (A + UV^T)
    let qt = transpose(&q);

    // Q^T A (actual_k x n)
    let qt_a = mat_mat_mul(&qt, a);
    // Q^T U (actual_k x r)
    let qt_u = mat_mat_mul(&qt, u_mat);
    // Q^T U V^T (actual_k x n)
    let qt_u_vt = mat_mat_mul(&qt_u, &vt);

    let actual_k = q.first().map_or(0, |row| row.len());
    let mut qt_total = vec![vec![0.0; n]; actual_k];
    for i in 0..actual_k {
        for j in 0..n {
            qt_total[i][j] = qt_a[i][j] + qt_u_vt[i][j];
        }
    }

    // Q * (Q^T * total) => m x n
    let result = mat_mat_mul(&q, &qt_total);

    Ok(result)
}

/// Compute the Frobenius norm of the difference between two matrices.
///
/// Returns ||original - approx||_F.
///
/// # Panics
///
/// Does not panic; returns 0.0 for empty matrices. Handles dimension
/// mismatches gracefully by treating missing elements as zero.
pub fn update_error_bound(original: &[Vec<f64>], approx: &[Vec<f64>]) -> f64 {
    let m = original.len().max(approx.len());
    let mut sum_sq = 0.0;

    for i in 0..m {
        let orig_row = original.get(i);
        let appr_row = approx.get(i);
        let n = orig_row
            .map_or(0, |r| r.len())
            .max(appr_row.map_or(0, |r| r.len()));

        for j in 0..n {
            let o = orig_row.and_then(|r| r.get(j)).copied().unwrap_or(0.0);
            let a = appr_row.and_then(|r| r.get(j)).copied().unwrap_or(0.0);
            sum_sq += (o - a) * (o - a);
        }
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_bound_identical() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let err = update_error_bound(&a, &b);
        assert!(err < 1e-14);
    }

    #[test]
    fn test_error_bound_known() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let err = update_error_bound(&a, &b);
        // ||I||_F = sqrt(2)
        assert!((err - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_nystrom_low_rank_perturbation() {
        // A = I(3), perturbation = [1;0;0] * [1,0,0] (rank-1)
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let e_u = vec![vec![1.0], vec![0.0], vec![0.0]];
        let e_v = vec![vec![1.0], vec![0.0], vec![0.0]];
        let config = SketchConfig {
            sketch_dimension: 3,
            seed: 42,
            error_tolerance: 1e-6,
        };

        let (q, b) = nystrom_update(&a, &e_u, &e_v, 1, &config).expect("nystrom");

        // Q*B should approximate A + UV^T = diag(2,1,1)
        let approx = mat_mat_mul(&q, &b);
        let exact = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let err = update_error_bound(&exact, &approx);
        assert!(
            err < 0.5,
            "Nystrom error {} should be small for exact sketch dim",
            err
        );
    }

    #[test]
    fn test_randomized_update_rank1() {
        // A = I(2), U = [1;0], V = [0;1], so A + UV^T = [[1,1],[0,1]]
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0], vec![0.0]];
        let v_mat = vec![vec![0.0], vec![1.0]];

        let result =
            randomized_low_rank_update(&a, &u_mat, &v_mat, 2, 42).expect("randomized update");

        let exact = vec![vec![1.0, 1.0], vec![0.0, 1.0]];
        let err = update_error_bound(&exact, &result);
        assert!(
            err < 1.0,
            "Randomized error {} should be reasonable for full sketch",
            err
        );
    }

    #[test]
    fn test_nystrom_dimension_mismatch() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let e_u = vec![vec![1.0]]; // wrong: should be 2x1
        let e_v = vec![vec![1.0], vec![0.0]];
        let config = SketchConfig::default();

        let result = nystrom_update(&a, &e_u, &e_v, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_randomized_dimension_mismatch() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let u_mat = vec![vec![1.0]]; // wrong
        let v_mat = vec![vec![0.0], vec![1.0]];

        let result = randomized_low_rank_update(&a, &u_mat, &v_mat, 2, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_bound_empty() {
        let err = update_error_bound(&[], &[]);
        assert!(err.abs() < 1e-14);
    }
}
