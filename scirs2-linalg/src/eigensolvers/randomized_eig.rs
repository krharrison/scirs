//! Randomized eigensolver via Nyström approximation.
//!
//! This implements the randomized SVD/eigenvalue algorithm from Halko, Martinsson &
//! Tropp (2011), specialized for symmetric positive semidefinite matrices via the
//! Nyström method (Williams & Seeger 2001).
//!
//! For a symmetric matrix A of size n×n, the algorithm:
//! 1. Draws a Gaussian random matrix Ω (n × k) where k = rank + oversampling.
//! 2. Applies power iteration: Y = (AAᵀ)^q A Ω to sharpen the approximation.
//! 3. Computes a thin QR decomposition: Y = Q R.
//! 4. Projects onto the k-dimensional subspace: B = Qᵀ A Q  (k × k symmetric).
//! 5. Solves the small eigenproblem B C = C Λ.
//! 6. Reconstructs approximate eigenvectors: V ≈ Q C.
//!
//! This gives the top-`rank` eigenvalues/eigenvectors in O(n k log n + k² n) time,
//! far cheaper than dense eigendecomposition for k ≪ n.
//!
//! # References
//!
//! - Halko, N., Martinsson, P.G. & Tropp, J.A. (2011). Finding structure with
//!   randomness: Probabilistic algorithms for constructing approximate matrix
//!   decompositions. *SIAM Review*, 53(2), 217–288.
//! - Williams, C.K.I. & Seeger, M. (2001). Using the Nyström method to speed up
//!   kernel machines. *NIPS 2000*.

use crate::error::{LinalgError, LinalgResult};

/// Configuration for the randomized eigensolver.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RandomizedEigConfig {
    /// Number of eigenvalues/eigenvectors to compute. Default: 20.
    pub rank: usize,
    /// Additional oversampling dimension to improve accuracy. Default: 10.
    pub n_oversampling: usize,
    /// Number of power iteration steps to sharpen the subspace. Default: 2.
    pub n_power_iter: usize,
    /// Random seed for reproducibility. Default: 42.
    pub seed: u64,
}

impl Default for RandomizedEigConfig {
    fn default() -> Self {
        Self {
            rank: 20,
            n_oversampling: 10,
            n_power_iter: 2,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute approximate top-`config.rank` eigenvalues and eigenvectors of a real
/// symmetric matrix using the randomized Nyström method.
///
/// # Arguments
///
/// * `a` — n×n real symmetric matrix (row-major).
/// * `config` — randomized eigensolver configuration.
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` sorted in descending order of eigenvalue magnitude.
/// The eigenvectors are returned as columns: `result.1[j][i]` is the i-th component
/// of the j-th eigenvector.
///
/// # Errors
///
/// Returns `LinalgError` if dimensions are inconsistent or rank exceeds matrix size.
pub fn randomized_eig_symmetric(
    a: &[Vec<f64>],
    config: &RandomizedEigConfig,
) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = a.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    for row in a {
        if row.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "Matrix row has {} elements, expected {n}",
                row.len()
            )));
        }
    }

    let rank = config.rank.min(n);
    let k = (rank + config.n_oversampling).min(n);

    // Step 1: Find approximate range of A via power iteration
    let q = randomized_range_finder(a, k, config.n_power_iter, config.seed, n)?;

    // Step 2: Project B = Q^T A Q  (k × k symmetric)
    let b = project_symmetric(a, &q, k, n);

    // Step 3: Solve small eigenproblem B C = C Λ via Jacobi
    let (b_evals, b_evecs) = jacobi_eig_small(&b, k)?;

    // Step 4: Reconstruct eigenvectors: v_j = Q * c_j
    // q is stored as: q[col_idx][row_idx] — q[j] is the j-th basis vector of length n
    let eigvecs: Vec<Vec<f64>> = b_evecs
        .iter()
        .map(|c| {
            let mut v = vec![0.0f64; n];
            for l in 0..k {
                for i in 0..n {
                    v[i] += q[l][i] * c[l];
                }
            }
            v
        })
        .collect();

    // Step 5: Return top-rank pairs sorted descending by eigenvalue
    let mut pairs: Vec<(f64, Vec<f64>)> = b_evals.into_iter().zip(eigvecs).collect();

    // Sort descending by eigenvalue magnitude
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to rank
    pairs.truncate(rank);

    let evals: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let evecs: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok((evals, evecs))
}

// ---------------------------------------------------------------------------
// Randomized range finder with power iteration
// ---------------------------------------------------------------------------

/// Compute an orthonormal basis Q (k columns, each of length n) for the approximate
/// range of A using Gaussian random projections and power iteration.
///
/// # Arguments
///
/// * `a` — n×n matrix.
/// * `k` — number of basis vectors (rank + oversampling).
/// * `q` — number of power iteration steps.
/// * `seed` — random seed.
/// * `n` — dimension.
pub fn randomized_range_finder(
    a: &[Vec<f64>],
    k: usize,
    q: usize,
    seed: u64,
    n: usize,
) -> LinalgResult<Vec<Vec<f64>>> {
    // Draw Gaussian random matrix Ω: n × k
    let mut lcg = Lcg::new(seed);
    let mut omega: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..n).map(|_| lcg.next_normal()).collect())
        .collect();

    // Y = A Ω  (store as k columns, each of length n)
    let mut y: Vec<Vec<f64>> = omega.iter().map(|col| matvec_sym(a, col, n)).collect();

    // Power iteration: Y = (A Aᵀ)^q * Y  but for symmetric A, Aᵀ = A, so:
    // Y = A^{2q} * A Ω  →  at each step, Y = A (A Y)
    for _ in 0..q {
        // Orthogonalize Y for numerical stability
        y = qr_cols(y, n)?;
        // Y = A (A Y)
        let ay: Vec<Vec<f64>> = y.iter().map(|col| matvec_sym(a, col, n)).collect();
        y = ay.iter().map(|col| matvec_sym(a, col, n)).collect();
    }

    // Final QR: Q = orth(Y)
    let q_basis = qr_cols(y, n)?;

    let _ = &mut omega; // suppress lint
    Ok(q_basis)
}

/// Compute Y = A x for symmetric matrix A.
fn matvec_sym(a: &[Vec<f64>], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

/// QR-orthogonalize a set of columns (each of length n) using modified Gram-Schmidt.
fn qr_cols(cols: Vec<Vec<f64>>, n: usize) -> LinalgResult<Vec<Vec<f64>>> {
    let k = cols.len();
    let mut q = cols;

    for j in 0..k {
        // Normalize column j
        let norm: f64 = q[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            // Replace with a canonical basis vector orthogonal to previous columns
            let mut found = false;
            for candidate in 0..n {
                let mut e = vec![0.0f64; n];
                e[candidate] = 1.0;
                // Orthogonalize against q[0..j]
                for ql in q.iter().take(j) {
                    let dot: f64 = ql.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
                    for (ei, qli) in e.iter_mut().zip(ql.iter()) {
                        *ei -= dot * qli;
                    }
                }
                let e_norm: f64 = e.iter().map(|x| x * x).sum::<f64>().sqrt();
                if e_norm > 1e-10 {
                    for xi in &mut e {
                        *xi /= e_norm;
                    }
                    q[j] = e;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(LinalgError::ComputationError(
                    "Could not find orthogonal complement in QR".to_string(),
                ));
            }
        } else {
            for xi in &mut q[j] {
                *xi /= norm;
            }
        }
        // Orthogonalize subsequent columns
        let qj_clone = q[j].clone();
        for ql in q.iter_mut().take(k).skip(j + 1) {
            let dot: f64 = qj_clone.iter().zip(ql.iter()).map(|(a, b)| a * b).sum();
            for (qli, qji) in ql.iter_mut().zip(qj_clone.iter()) {
                *qli -= dot * qji;
            }
        }
    }

    // One reorthogonalization pass for numerical stability
    for j in 0..k {
        for l in 0..j {
            let ql_clone = q[l].clone();
            let dot: f64 = ql_clone.iter().zip(q[j].iter()).map(|(a, b)| a * b).sum();
            for (qji, qli) in q[j].iter_mut().zip(ql_clone.iter()) {
                *qji -= dot * qli;
            }
        }
        let norm: f64 = q[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for xi in &mut q[j] {
                *xi /= norm;
            }
        }
    }

    Ok(q)
}

// ---------------------------------------------------------------------------
// Projection: B = Q^T A Q
// ---------------------------------------------------------------------------

fn project_symmetric(a: &[Vec<f64>], q: &[Vec<f64>], k: usize, n: usize) -> Vec<Vec<f64>> {
    // Compute AQ: for each column q[j], compute A * q[j]
    let aq: Vec<Vec<f64>> = (0..k).map(|j| matvec_sym(a, &q[j], n)).collect();

    // B[i][j] = q[i]^T * (AQ)[j]  = sum_r q[i][r] * aq[j][r]
    let mut b = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let dot: f64 = q[i].iter().zip(aq[j].iter()).map(|(a, b)| a * b).sum();
            b[i][j] = dot;
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Small symmetric eigensolver: Jacobi
// ---------------------------------------------------------------------------

fn jacobi_eig_small(a: &[Vec<f64>], n: usize) -> LinalgResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let max_sweeps = 200;
    let tol = 1e-13;

    let mut mat: Vec<Vec<f64>> = a.to_vec();
    // evecs[j] = j-th column of eigenvector matrix, length n
    let mut evecs: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            let mut col = vec![0.0f64; n];
            col[j] = 1.0;
            col
        })
        .collect();

    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0f64;
        let mut p = 0usize;
        let mut q_idx = 1usize;
        for (i, mat_i) in mat.iter().enumerate().take(n) {
            for (j, &mat_ij) in mat_i.iter().enumerate().take(n).skip(i + 1) {
                let v = mat_ij.abs();
                if v > max_off {
                    max_off = v;
                    p = i;
                    q_idx = j;
                }
            }
        }
        if max_off < tol {
            break;
        }

        let theta = (mat[q_idx][q_idx] - mat[p][p]) / (2.0 * mat[p][q_idx]);
        let sign_t = if theta >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        let t = sign_t / (theta.abs() + (1.0 + theta * theta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let app = mat[p][p];
        let aqq = mat[q_idx][q_idx];
        let apq = mat[p][q_idx];

        mat[p][p] = app - t * apq;
        mat[q_idx][q_idx] = aqq + t * apq;
        mat[p][q_idx] = 0.0;
        mat[q_idx][p] = 0.0;

        for (r, mat_r) in mat.iter_mut().enumerate().take(n) {
            if r != p && r != q_idx {
                let arp = mat_r[p];
                let arq = mat_r[q_idx];
                mat_r[p] = c * arp - s * arq;
                mat_r[q_idx] = s * arp + c * arq;
            }
        }
        // Symmetrize
        {
            let col_p: Vec<f64> = (0..n).map(|r| mat[r][p]).collect();
            let col_q: Vec<f64> = (0..n).map(|r| mat[r][q_idx]).collect();
            for r in 0..n {
                if r != p && r != q_idx {
                    mat[p][r] = col_p[r];
                    mat[q_idx][r] = col_q[r];
                }
            }
        }

        // Update eigenvectors
        {
            let (left, right) = evecs.split_at_mut(q_idx);
            let ep = &mut left[p];
            let eq = &mut right[0];
            for (vp, vq) in ep.iter_mut().zip(eq.iter_mut()) {
                let old_p = *vp;
                let old_q = *vq;
                *vp = c * old_p - s * old_q;
                *vq = s * old_p + c * old_q;
            }
        }
    }

    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n).map(|i| (mat[i][i], evecs[i].clone())).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let evals: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let evecs_sorted: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    Ok((evals, evecs_sorted))
}

// ---------------------------------------------------------------------------
// Simple LCG random number generator
// ---------------------------------------------------------------------------

/// Linear congruential generator for reproducible random numbers.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate the next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        // Knuth's multiplier + Newlib addend
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a uniform float in (0, 1).
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Generate a standard normal variate using Box-Muller transform.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that eigenvectors are approximately orthonormal.
    fn check_orthonormal(evecs: &[Vec<f64>], tol: f64) -> bool {
        let m = evecs.len();
        for i in 0..m {
            let dot_ii: f64 = evecs[i].iter().map(|x| x * x).sum();
            if (dot_ii - 1.0).abs() > tol {
                return false;
            }
            for j in i + 1..m {
                let dot_ij: f64 = evecs[i]
                    .iter()
                    .zip(evecs[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                if dot_ij.abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_randomized_eig_rank3() {
        // Construct a rank-3 symmetric PSD matrix: A = V * diag(10, 5, 2) * V^T
        // using 3 orthonormal vectors.
        let n = 8;
        // Simple orthonormal vectors: scaled standard basis
        let vecs: Vec<Vec<f64>> = vec![
            {
                let mut v = vec![0.0f64; n];
                v[0] = 1.0;
                v
            },
            {
                let mut v = vec![0.0f64; n];
                v[1] = 1.0;
                v
            },
            {
                let mut v = vec![0.0f64; n];
                v[2] = 1.0;
                v
            },
        ];
        let lambdas = [10.0f64, 5.0, 2.0];

        let mut a = vec![vec![0.0f64; n]; n];
        for (k, lam) in lambdas.iter().enumerate() {
            let v = &vecs[k];
            for i in 0..n {
                for j in 0..n {
                    a[i][j] += lam * v[i] * v[j];
                }
            }
        }

        let config = RandomizedEigConfig {
            rank: 3,
            n_oversampling: 5,
            n_power_iter: 2,
            seed: 123,
        };

        let (evals, evecs) = randomized_eig_symmetric(&a, &config).expect("Randomized eig failed");

        assert_eq!(evals.len(), 3);
        assert_eq!(evecs.len(), 3);

        // Top eigenvalues should be close to 10, 5, 2 (sorted descending)
        assert!(
            (evals[0] - 10.0).abs() < 0.1,
            "Expected λ_1 ≈ 10, got {}",
            evals[0]
        );
        assert!(
            (evals[1] - 5.0).abs() < 0.1,
            "Expected λ_2 ≈ 5, got {}",
            evals[1]
        );
        assert!(
            (evals[2] - 2.0).abs() < 0.1,
            "Expected λ_3 ≈ 2, got {}",
            evals[2]
        );

        // Eigenvectors should be approximately orthonormal
        assert!(
            check_orthonormal(&evecs, 1e-6),
            "Eigenvectors not orthonormal"
        );
    }

    #[test]
    fn test_lcg_statistics() {
        // Basic sanity check: mean of many normals ≈ 0
        let mut lcg = Lcg::new(0);
        let n = 10000;
        let sum: f64 = (0..n).map(|_| lcg.next_normal()).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05, "LCG normal mean too far from 0: {mean}");
    }

    #[test]
    fn test_range_finder_orthonormality() {
        let n = 6;
        let mut a = vec![vec![0.0f64; n]; n];
        // Symmetric matrix: tridiagonal
        for i in 0..n {
            a[i][i] = (i + 1) as f64;
            if i + 1 < n {
                a[i][i + 1] = 0.5;
                a[i + 1][i] = 0.5;
            }
        }
        let q = randomized_range_finder(&a, 4, 1, 42, n).expect("Range finder failed");

        // Check orthonormality of Q columns
        let m = q.len();
        for i in 0..m {
            let norm: f64 = q[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Q column {i} not normalized: {norm}"
            );
            for j in i + 1..m {
                let dot: f64 = q[i].iter().zip(q[j].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-10, "Q columns {i},{j} not orthogonal: {dot}");
            }
        }
    }

    #[test]
    fn test_randomized_eig_symmetric_tridiag() {
        // Small symmetric tridiagonal — full spectrum should be recoverable
        let n = 5;
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = 2.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }

        let config = RandomizedEigConfig {
            rank: 3,
            n_oversampling: 2,
            n_power_iter: 3,
            seed: 7,
        };

        let (evals, evecs) = randomized_eig_symmetric(&a, &config).expect("Randomized eig failed");

        assert_eq!(evals.len(), 3);

        // All eigenvalues of this 5×5 tridiagonal are positive (it's PD).
        // The largest eigenvalue should be close to 2 + 2*cos(π/6) ≈ 3.73
        // and the smallest ≈ 2 - 2*cos(π/6) ≈ 0.27. Top 3 are ≈ 3.73, 3.0, ~2.0.
        for &ev in &evals {
            assert!(ev > 0.0, "Expected positive eigenvalue, got {ev}");
            assert!(ev < 5.0, "Eigenvalue {ev} unexpectedly large");
        }

        // Top eigenvalue should be the largest
        assert!(evals[0] >= evals[1], "Eigenvalues not sorted descending");
        if evals.len() > 2 {
            assert!(evals[1] >= evals[2], "Eigenvalues not sorted descending");
        }

        // Eigenvectors orthonormal
        assert!(
            check_orthonormal(&evecs, 1e-6),
            "Eigenvectors not orthonormal"
        );
    }
}
