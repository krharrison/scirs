//! Low-rank matrix approximation via truncated SVD.
//!
//! Uses a randomized SVD algorithm (power-iteration + QR) that works entirely
//! in safe Rust without any LAPACK/BLAS C dependencies.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─────────────────────────────────────────────────────────────────────────────
// Data structure
// ─────────────────────────────────────────────────────────────────────────────

/// Thin SVD factorisation `A ≈ U Σ Vᵀ` truncated to `rank` components.
///
/// - `U`  : `(m, rank)` matrix with orthonormal columns.
/// - `sigma`: length-`rank` singular values in descending order.
/// - `vt` : `(rank, n)` matrix with orthonormal rows.
#[derive(Debug, Clone)]
pub struct LowRankApprox {
    /// Number of retained singular values.
    pub rank: usize,
    /// Left singular vectors `(m, rank)`.
    pub u: Array2<f32>,
    /// Singular values `(rank,)`.
    pub sigma: Array1<f32>,
    /// Right singular vectors transposed `(rank, n)`.
    pub vt: Array2<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a rank-`rank` approximation via randomized SVD with power iteration.
///
/// Uses the algorithm from Halko, Martinsson & Tropp (2011) with `n_power_iter = 4`
/// power iterations, which gives very accurate results for moderate ranks.
///
/// # Errors
/// Returns an error if `rank == 0`, `rank > min(m, n)`, or if the matrix is empty.
pub fn truncated_svd(matrix: &Array2<f32>, rank: usize) -> Result<LowRankApprox> {
    let (m, n) = (matrix.nrows(), matrix.ncols());
    if m == 0 || n == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "truncated_svd: matrix must not be empty".into(),
        ));
    }
    if rank == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "truncated_svd: rank must be > 0".into(),
        ));
    }
    let max_rank = m.min(n);
    if rank > max_rank {
        return Err(NeuralError::InvalidArchitecture(format!(
            "truncated_svd: rank {rank} > min(m={m}, n={n}) = {max_rank}"
        )));
    }

    // ── Step 1: randomized range finder ──────────────────────────────────────
    // Sketch: Y = A * Omega  where Omega is (n, k), k = rank + oversampling.
    let oversampling = 10.min(max_rank - rank);
    let k = rank + oversampling;
    // Deterministic pseudo-random seed (linear congruential generator).
    let omega = lcg_normal_matrix(n, k, 42u64);

    // Y = A @ Omega  (m x k)
    let mut y = mat_mul(matrix, &omega)?;

    // Power iteration: Y = (A Aᵀ)^q A Omega to improve accuracy.
    let n_power_iter = 4usize;
    for _ in 0..n_power_iter {
        // Y = A Aᵀ Y
        let aty = mat_mul_t(matrix, &y)?; // (n x k) = Aᵀ (m x k)
        y = mat_mul(matrix, &aty)?; // (m x k) = A (n x k)
    }

    // ── Step 2: QR decomposition of Y ────────────────────────────────────────
    let q = qr_thin(&y)?; // (m, k), orthonormal columns

    // ── Step 3: Project A onto Q-subspace ────────────────────────────────────
    // B = Qᵀ A  (k x n)
    let b = mat_mul_t_left(&q, matrix)?;

    // ── Step 4: Full SVD of the small matrix B ────────────────────────────────
    // Using Jacobi one-sided SVD for B (k x n), where k = rank + oversampling.
    let (ub, sb, vb_t) = svd_small(&b)?; // ub:(k,k), sb:(k,), vb_t:(k,n)

    // ── Step 5: Recover left singular vectors of A ───────────────────────────
    // U_full = Q * U_B  (m x k)
    let u_full = mat_mul(&q, &ub)?;

    // Truncate to `rank`
    let u = u_full.slice(scirs2_core::ndarray::s![.., ..rank]).to_owned();
    let sigma = sb.slice(scirs2_core::ndarray::s![..rank]).to_owned();
    let vt = vb_t.slice(scirs2_core::ndarray::s![..rank, ..]).to_owned();

    Ok(LowRankApprox { rank, u, sigma, vt })
}

/// Reconstruct the approximate matrix `A ≈ U Σ Vᵀ`.
///
/// # Errors
/// Returns an error if dimensions are inconsistent.
pub fn reconstruct(approx: &LowRankApprox) -> Result<Array2<f32>> {
    // U_scaled = U * diag(sigma)  (m x rank)
    let m = approx.u.nrows();
    let n = approx.vt.ncols();
    let r = approx.rank;

    if approx.u.ncols() != r || approx.sigma.len() != r || approx.vt.nrows() != r {
        return Err(NeuralError::InvalidArchitecture(format!(
            "reconstruct: inconsistent dimensions u={:?} sigma={} vt={:?}",
            approx.u.shape(),
            approx.sigma.len(),
            approx.vt.shape()
        )));
    }

    // Scale columns of U by sigma.
    let mut u_scaled = approx.u.clone();
    for j in 0..r {
        let s = approx.sigma[j];
        for i in 0..m {
            u_scaled[(i, j)] *= s;
        }
    }
    // A ≈ U_scaled @ Vt
    mat_mul(&u_scaled, &approx.vt)
}

/// Frobenius-norm relative approximation error:
/// `‖A - Â‖_F / ‖A‖_F`.
///
/// Returns 0 if `original` is the zero matrix.
///
/// # Errors
/// Returns an error if reconstruction fails or shapes mismatch.
pub fn approximate_error(original: &Array2<f32>, approx: &LowRankApprox) -> Result<f32> {
    let reconstructed = reconstruct(approx)?;
    if original.shape() != reconstructed.shape() {
        return Err(NeuralError::InvalidArchitecture(format!(
            "approximate_error: original shape {:?} != reconstructed shape {:?}",
            original.shape(),
            reconstructed.shape()
        )));
    }
    let orig_norm = frobenius_norm(original);
    if orig_norm == 0.0 {
        return Ok(0.0);
    }
    let diff_norm = frobenius_norm(&(original - &reconstructed));
    Ok(diff_norm / orig_norm)
}

/// Compress a weight matrix to achieve an approximate `compression_ratio`.
///
/// `compression_ratio` is defined as `original_params / compressed_params`,
/// where compressed parameters for a `(m, n)` matrix at rank `r` is
/// `r * (m + n + 1)` (U, V, and sigma).
///
/// The rank is chosen as the largest `r` satisfying the ratio constraint.
///
/// # Errors
/// Returns an error if `compression_ratio < 1` or no valid rank exists.
pub fn compress_layer(weights: &Array2<f32>, compression_ratio: f64) -> Result<LowRankApprox> {
    if compression_ratio < 1.0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "compress_layer: compression_ratio must be >= 1, got {compression_ratio}"
        )));
    }
    let (m, n) = (weights.nrows(), weights.ncols());
    let original_params = (m * n) as f64;
    // rank * (m + n + 1) <= original_params / compression_ratio
    let target_params = original_params / compression_ratio;
    let max_r = m.min(n);
    let rank = (1..=max_r)
        .filter(|&r| (r * (m + n + 1)) as f64 <= target_params)
        .max()
        .ok_or_else(|| {
            NeuralError::InvalidArchitecture(format!(
                "compress_layer: no valid rank for ({m}×{n}) matrix at ratio {compression_ratio}"
            ))
        })?;
    truncated_svd(weights, rank)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal numerical routines
// ─────────────────────────────────────────────────────────────────────────────

/// Frobenius norm of a matrix.
fn frobenius_norm(a: &Array2<f32>) -> f32 {
    a.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Matrix multiply `C = A @ B`.
fn mat_mul(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
    let (m, k1) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k1 != k2 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "mat_mul: ({m}×{k1}) @ ({k2}×{n}) — inner dims mismatch"
        )));
    }
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for l in 0..k1 {
                s += a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = s;
        }
    }
    Ok(c)
}

/// `C = A @ Bᵀ`  where `B` is `(k, m)` → `C` is `(rows_a, k)`.
fn mat_mul_t(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
    // Implements Aᵀ @ b where a has shape (m, k) and b has shape (m, p)
    // → result (k, p)
    let (m1, k) = (a.nrows(), a.ncols());
    let (m2, p) = (b.nrows(), b.ncols());
    if m1 != m2 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "mat_mul_t: A({m1}×{k})ᵀ incompatible with B({m2}×{p})"
        )));
    }
    let mut c = Array2::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            let mut s = 0.0_f32;
            for l in 0..m1 {
                s += a[(l, i)] * b[(l, j)];
            }
            c[(i, j)] = s;
        }
    }
    Ok(c)
}

/// `C = Aᵀ @ B`.
fn mat_mul_t_left(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
    // a: (m, k), b: (m, n) → C: (k, n)
    let (m1, k) = (a.nrows(), a.ncols());
    let (m2, n) = (b.nrows(), b.ncols());
    if m1 != m2 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "mat_mul_t_left: A({m1}×{k}) vs B({m2}×{n}) row mismatch"
        )));
    }
    let mut c = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            let mut s = 0.0_f32;
            for l in 0..m1 {
                s += a[(l, i)] * b[(l, j)];
            }
            c[(i, j)] = s;
        }
    }
    Ok(c)
}

/// Generate an `(m, n)` matrix of pseudo-standard-normal random values using
/// a Linear Congruential Generator (no external RNG dependency).
fn lcg_normal_matrix(m: usize, n: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let mut flat = Vec::with_capacity(m * n);
    for _ in 0..m * n {
        // Box-Muller needs two uniform samples; we generate them in pairs.
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let u1 = (state >> 33) as f32 / (u32::MAX as f32) + 1e-10;
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let u2 = (state >> 33) as f32 / (u32::MAX as f32);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        flat.push(z);
    }
    Array2::from_shape_vec((m, n), flat).expect("lcg_normal_matrix shape error")
}

/// Thin QR decomposition via modified Gram-Schmidt.
///
/// Returns `Q` of shape `(m, k)` with orthonormal columns.
fn qr_thin(a: &Array2<f32>) -> Result<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let mut q_cols: Vec<Vec<f32>> = Vec::with_capacity(k);

    for j in 0..k {
        let mut col: Vec<f32> = (0..m).map(|i| a[(i, j)]).collect();

        // Subtract projections onto already-computed orthonormal vectors.
        for prev in &q_cols {
            let dot: f32 = col.iter().zip(prev.iter()).map(|(&c, &p)| c * p).sum();
            for (c, &p) in col.iter_mut().zip(prev.iter()) {
                *c -= dot * p;
            }
        }

        // Normalise.
        let norm: f32 = col.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if norm < 1e-10 {
            // Nearly linearly dependent column; fill with zeros.
            q_cols.push(vec![0.0; m]);
        } else {
            q_cols.push(col.into_iter().map(|v| v / norm).collect());
        }
    }

    // Build Q matrix column-by-column.
    let mut q = Array2::zeros((m, k));
    for (j, col) in q_cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            q[(i, j)] = v;
        }
    }
    Ok(q)
}

/// Compute full SVD of a small matrix `B` (shape `(k, n)`) via Jacobi iterations
/// on the symmetric eigenproblem `Bᵀ B`.
///
/// Returns `(U, S, Vᵀ)` where:
/// - `U` : `(k, k)` orthogonal
/// - `S` : `(k,)` singular values in descending order
/// - `Vᵀ`: `(k, n)` matrix (rows are right singular vectors)
fn svd_small(b: &Array2<f32>) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>)> {
    let (k, n) = (b.nrows(), b.ncols());

    // Build BᵀB (n×n) or BBᵀ (k×k) — use the smaller one.
    if n >= k {
        // Eigen-decompose BBᵀ (k×k) → get U and S; derive V.
        let bbt = mat_mul(b, &b.t().to_owned())?;
        let (eigenvalues, eigenvectors) = jacobi_eigen_sym(&bbt)?; // eigenvalues desc

        let n_sv = k.min(n);
        let s: Array1<f32> = eigenvalues.iter().take(n_sv).map(|&e| e.max(0.0).sqrt()).collect();
        let u = eigenvectors.slice(scirs2_core::ndarray::s![.., ..n_sv]).to_owned(); // (k, n_sv)

        // V = B^T U S^{-1}  → Vᵀ = (B^T U S^{-1})^T = S^{-1} U^T B
        // Vᵀ shape: (n_sv, n)
        let mut vt = Array2::zeros((n_sv, n));
        for j in 0..n_sv {
            let sv = s[j];
            if sv < 1e-10 {
                continue;
            }
            // v_j = (1/sv) * B^T u_j
            for ci in 0..n {
                let mut acc = 0.0_f32;
                for ri in 0..k {
                    acc += b[(ri, ci)] * u[(ri, j)];
                }
                vt[(j, ci)] = acc / sv;
            }
        }
        Ok((u, s, vt))
    } else {
        // Eigen-decompose BᵀB (n×n) → get V and S; derive U.
        let btb = mat_mul(&b.t().to_owned(), b)?;
        let (eigenvalues, eigenvectors) = jacobi_eigen_sym(&btb)?; // eigenvalues desc

        let n_sv = k.min(n);
        let s: Array1<f32> = eigenvalues.iter().take(n_sv).map(|&e| e.max(0.0).sqrt()).collect();
        let v = eigenvectors.slice(scirs2_core::ndarray::s![.., ..n_sv]).to_owned(); // (n, n_sv)
        let vt = v.t().to_owned(); // (n_sv, n)

        // U = B V S^{-1}
        let mut u = Array2::zeros((k, n_sv));
        for j in 0..n_sv {
            let sv = s[j];
            if sv < 1e-10 {
                continue;
            }
            for ri in 0..k {
                let mut acc = 0.0_f32;
                for ci in 0..n {
                    acc += b[(ri, ci)] * v[(ci, j)];
                }
                u[(ri, j)] = acc / sv;
            }
        }
        Ok((u, s, vt))
    }
}

/// Jacobi eigendecomposition of a symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` sorted in *descending* order of eigenvalue.
/// `eigenvectors` is `(n, n)` with eigenvectors as columns.
fn jacobi_eigen_sym(a_in: &Array2<f32>) -> Result<(Vec<f32>, Array2<f32>)> {
    let n = a_in.nrows();
    if n != a_in.ncols() {
        return Err(NeuralError::InvalidArchitecture(
            "jacobi_eigen_sym: matrix must be square".into(),
        ));
    }

    let mut a: Vec<f32> = a_in.iter().cloned().collect();
    let mut v_flat: Vec<f32> = vec![0.0; n * n];
    // Initialize V = I.
    for i in 0..n {
        v_flat[i * n + i] = 1.0;
    }

    let max_iter = 200 * n * n;
    for _ in 0..max_iter {
        // Find off-diagonal element with largest absolute value.
        let mut max_off = 0.0_f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }

        // Compute rotation angle.
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (aqq - app).abs() < 1e-14 {
            std::f32::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * apq) / (aqq - app)).atan()
        };
        let (s_val, c_val) = theta.sin_cos();

        // Update A in place (Jacobi rotation).
        // Save affected rows/cols.
        let old_a: Vec<f32> = a.clone();
        for r in 0..n {
            if r != p && r != q {
                a[r * n + p] = c_val * old_a[r * n + p] - s_val * old_a[r * n + q];
                a[p * n + r] = a[r * n + p];
                a[r * n + q] = s_val * old_a[r * n + p] + c_val * old_a[r * n + q];
                a[q * n + r] = a[r * n + q];
            }
        }
        a[p * n + p] = c_val * c_val * old_a[p * n + p] + s_val * s_val * old_a[q * n + q]
            - 2.0 * s_val * c_val * old_a[p * n + q];
        a[q * n + q] = s_val * s_val * old_a[p * n + p] + c_val * c_val * old_a[q * n + q]
            + 2.0 * s_val * c_val * old_a[p * n + q];
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors.
        for r in 0..n {
            let vp = v_flat[r * n + p];
            let vq = v_flat[r * n + q];
            v_flat[r * n + p] = c_val * vp - s_val * vq;
            v_flat[r * n + q] = s_val * vp + c_val * vq;
        }
    }

    // Extract diagonal (eigenvalues).
    let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();

    // Sort by descending eigenvalue.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_eigenvalues: Vec<f32> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut eigenvectors = Array2::zeros((n, n));
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            eigenvectors[(row, new_col)] = v_flat[row * n + old_col];
        }
    }

    Ok((sorted_eigenvalues, eigenvectors))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a rank-`r` matrix from known factors.
    fn rank_r_matrix(m: usize, n: usize, r: usize) -> Array2<f32> {
        let u = lcg_normal_matrix(m, r, 1);
        let v = lcg_normal_matrix(n, r, 2);
        mat_mul(&u, &v.t().to_owned()).expect("rank_r_matrix failed")
    }

    #[test]
    fn test_truncated_svd_rank1_exact() {
        // A rank-1 matrix should be exactly recovered at rank=1.
        let a = rank_r_matrix(8, 6, 1);
        let approx = truncated_svd(&a, 1).expect("truncated_svd failed");
        let err = approximate_error(&a, &approx).expect("approximate_error failed");
        // Should reconstruct perfectly (within float precision).
        assert!(err < 0.01, "rank-1 error should be tiny, got {err}");
    }

    #[test]
    fn test_truncated_svd_rank_full() {
        // A full-rank 4×4 matrix approximated at rank 4 should have near-zero error.
        let a = lcg_normal_matrix(4, 4, 99);
        let approx = truncated_svd(&a, 4).expect("truncated_svd rank-4 failed");
        let err = approximate_error(&a, &approx).expect("approximate_error failed");
        assert!(err < 0.05, "full-rank error should be small, got {err}");
    }

    #[test]
    fn test_truncated_svd_singular_values_descending() {
        let a = lcg_normal_matrix(10, 8, 7);
        let approx = truncated_svd(&a, 4).expect("truncated_svd failed");
        for i in 0..(approx.rank - 1) {
            assert!(
                approx.sigma[i] >= approx.sigma[i + 1] - 1e-4,
                "sigma not descending at {i}: {} < {}",
                approx.sigma[i],
                approx.sigma[i + 1]
            );
        }
    }

    #[test]
    fn test_truncated_svd_invalid_rank() {
        let a = lcg_normal_matrix(4, 3, 5);
        assert!(truncated_svd(&a, 0).is_err());
        assert!(truncated_svd(&a, 5).is_err()); // rank > min(4,3) = 3
    }

    #[test]
    fn test_reconstruct_shape() {
        let a = lcg_normal_matrix(6, 8, 3);
        let approx = truncated_svd(&a, 2).expect("failed");
        let recon = reconstruct(&approx).expect("reconstruct failed");
        assert_eq!(recon.shape(), a.shape());
    }

    #[test]
    fn test_approximate_error_zero_matrix() {
        let a = Array2::zeros((4, 4));
        let approx = truncated_svd(&a, 1).expect("failed");
        let err = approximate_error(&a, &approx).expect("error failed");
        assert_eq!(err, 0.0);
    }

    #[test]
    fn test_compress_layer_basic() {
        let weights = lcg_normal_matrix(32, 32, 11);
        // With ratio=4, target_params = 1024/4 = 256.
        // rank*(32+32+1) <= 256 → rank <= 3.
        let approx = compress_layer(&weights, 4.0).expect("compress_layer failed");
        assert!(approx.rank >= 1);
        let err = approximate_error(&weights, &approx).expect("error failed");
        // Large compression → some loss acceptable.
        assert!(err < 1.1, "error should be < 1.1 for rank-approx, got {err}");
    }

    #[test]
    fn test_compress_layer_invalid_ratio() {
        let weights = lcg_normal_matrix(4, 4, 0);
        assert!(compress_layer(&weights, 0.5).is_err());
    }

    #[test]
    fn test_frobenius_norm() {
        let a = Array2::from_shape_vec((2, 2), vec![3.0_f32, 4.0, 0.0, 0.0]).expect("shape");
        let norm = frobenius_norm(&a);
        assert!((norm - 5.0).abs() < 1e-5, "expected 5, got {norm}");
    }
}
