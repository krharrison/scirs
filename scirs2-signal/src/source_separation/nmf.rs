//! Non-negative Matrix Factorization (NMF) for audio/spectral source separation.
//!
//! Implements Frobenius, KL-divergence, and ALS variants.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// NMF result storing factor matrices.
#[derive(Debug, Clone)]
pub struct NMF {
    /// Basis matrix W (n_features × rank).
    pub w: Vec<Vec<f64>>,
    /// Activation matrix H (rank × n_samples).
    pub h: Vec<Vec<f64>>,
    /// Factorization rank.
    pub rank: usize,
}

/// Configuration for NMF algorithms.
#[derive(Debug, Clone)]
pub struct NMFConfig {
    /// Factorization rank.
    pub rank: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative change in objective).
    pub tol: f64,
    /// Regularization strength.
    pub alpha: f64,
    /// Beta exponent for beta-divergence (unused in basic variants; reserved).
    pub beta: f64,
    /// L1 ratio (0 = L2, 1 = L1) for regularization.
    pub l1_ratio: f64,
}

impl Default for NMFConfig {
    fn default() -> Self {
        Self {
            rank: 2,
            max_iter: 200,
            tol: 1e-4,
            alpha: 0.0,
            beta: 2.0,
            l1_ratio: 0.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

const EPS: f64 = 1e-10;

/// Convert `Vec<Vec<f64>>` to `Array2<f64>`.
fn vv_to_arr2(data: &[Vec<f64>]) -> SignalResult<Array2<f64>> {
    let rows = data.len();
    if rows == 0 {
        return Err(SignalError::ValueError("Empty input matrix".into()));
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(SignalError::ValueError("Empty row in input matrix".into()));
    }
    let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| SignalError::ShapeMismatch(e.to_string()))
}

/// Convert `Array2<f64>` to `Vec<Vec<f64>>`.
fn arr2_to_vv(a: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..a.nrows()).map(|i| a.row(i).to_vec()).collect()
}

/// Random non-negative initialisation via a simple LCG.
fn random_nonneg(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed.wrapping_add(1);
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let val = ((state >> 33) as f64) / (u32::MAX as f64) + EPS;
        data.push(val);
    }
    Array2::from_shape_vec((rows, cols), data).expect("shape ok")
}

/// Matrix-matrix product (m×k) × (k×n) → (m×n) using ndarray dot.
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    if a.ncols() != b.nrows() {
        return Err(SignalError::DimensionMismatch(format!(
            "matmul: {}×{} cannot multiply {}×{}",
            a.nrows(), a.ncols(), b.nrows(), b.ncols()
        )));
    }
    Ok(a.dot(b))
}

/// Frobenius norm ||A - B||_F^2.
fn frob_sq(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum()
}

/// KL divergence D_KL(V || W*H) = sum(V*log(V/WH) - V + WH).
fn kl_div(v: &Array2<f64>, wh: &Array2<f64>) -> f64 {
    v.iter()
        .zip(wh.iter())
        .map(|(&vi, &whi)| {
            if vi < EPS {
                whi
            } else {
                vi * (vi / (whi + EPS)).ln() - vi + whi
            }
        })
        .sum()
}

/// Validate that V has non-negative entries.
fn validate_nonneg(v: &[Vec<f64>], name: &str) -> SignalResult<()> {
    for (i, row) in v.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val < 0.0 {
                return Err(SignalError::ValueError(format!(
                    "{name}[{i}][{j}] = {val} is negative; NMF requires non-negative input"
                )));
            }
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// NMF implementation
// ──────────────────────────────────────────────────────────────────────────────

impl NMF {
    /// Reconstruct V ≈ W * H.
    pub fn reconstruct(&self) -> SignalResult<Vec<Vec<f64>>> {
        let w = vv_to_arr2(&self.w)?;
        let h = vv_to_arr2(&self.h)?;
        let wh = matmul(&w, &h)?;
        Ok(arr2_to_vv(&wh))
    }

    /// Return H (activation matrix) as `fit_transform` result.
    pub fn fit_transform(&self, _v: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
        Ok(self.h.clone())
    }

    /// Reconstruction error ||V - W*H||_F.
    pub fn reconstruction_error(&self, v: &[Vec<f64>]) -> SignalResult<f64> {
        let v_arr = vv_to_arr2(v)?;
        let wh_vv = self.reconstruct()?;
        let wh_arr = vv_to_arr2(&wh_vv)?;
        Ok(frob_sq(&v_arr, &wh_arr).sqrt())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Frobenius NMF (Lee-Seung 2001)
// ──────────────────────────────────────────────────────────────────────────────

/// NMF with Frobenius-norm objective (multiplicative updates).
///
/// Update rules:
/// - H ← H ⊙ (W^T V) / (W^T W H + ε)
/// - W ← W ⊙ (V H^T) / (W H H^T + ε)
pub fn multiplicative_updates_frobenius(
    v: &[Vec<f64>],
    rank: usize,
    config: &NMFConfig,
) -> SignalResult<NMF> {
    validate_nonneg(v, "V")?;
    let v_arr = vv_to_arr2(v)?;
    let (m, n) = v_arr.dim();
    if rank == 0 || rank > m.min(n) {
        return Err(SignalError::InvalidArgument(format!(
            "rank must be in 1..=min({m},{n})"
        )));
    }

    let mut w = random_nonneg(m, rank, 42);
    let mut h = random_nonneg(rank, n, 137);

    let mut prev_obj = f64::INFINITY;

    for _iter in 0..config.max_iter {
        // H ← H ⊙ (W^T V) / (W^T W H + ε)
        let wt_v = w.t().dot(&v_arr);
        let wt_w = w.t().dot(&w);
        let wt_wh = wt_w.dot(&h);
        for i in 0..rank {
            for j in 0..n {
                let num = wt_v[[i, j]] + EPS;
                let den = wt_wh[[i, j]] + EPS;
                h[[i, j]] *= num / den;
                h[[i, j]] = h[[i, j]].max(EPS);
            }
        }

        // W ← W ⊙ (V H^T) / (W H H^T + ε)
        let v_ht = v_arr.dot(&h.t());
        let h_ht = h.dot(&h.t());
        let whht = w.dot(&h_ht);
        for i in 0..m {
            for j in 0..rank {
                let num = v_ht[[i, j]] + EPS;
                let den = whht[[i, j]] + EPS;
                w[[i, j]] *= num / den;
                w[[i, j]] = w[[i, j]].max(EPS);
            }
        }

        // Apply regularization (L1 + L2 on H)
        if config.alpha > 0.0 {
            let l1 = config.alpha * config.l1_ratio;
            let l2 = config.alpha * (1.0 - config.l1_ratio);
            for i in 0..rank {
                for j in 0..n {
                    h[[i, j]] = (h[[i, j]] - l1).max(0.0) / (1.0 + l2);
                    h[[i, j]] = h[[i, j]].max(EPS);
                }
            }
        }

        // Check convergence
        let wh = matmul(&w, &h)?;
        let obj = frob_sq(&v_arr, &wh);
        if (prev_obj - obj).abs() / (prev_obj + EPS) < config.tol {
            break;
        }
        prev_obj = obj;
    }

    Ok(NMF {
        w: arr2_to_vv(&w),
        h: arr2_to_vv(&h),
        rank,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// KL-divergence NMF
// ──────────────────────────────────────────────────────────────────────────────

/// NMF with Kullback-Leibler divergence objective (multiplicative updates).
///
/// Update rules:
/// - H ← H ⊙ (W^T (V / W H)) / (W^T 1)
/// - W ← W ⊙ ((V / W H) H^T) / (1 H^T)
pub fn multiplicative_updates_kl(
    v: &[Vec<f64>],
    rank: usize,
    config: &NMFConfig,
) -> SignalResult<NMF> {
    validate_nonneg(v, "V")?;
    let v_arr = vv_to_arr2(v)?;
    let (m, n) = v_arr.dim();
    if rank == 0 || rank > m.min(n) {
        return Err(SignalError::InvalidArgument(format!(
            "rank must be in 1..=min({m},{n})"
        )));
    }

    let mut w = random_nonneg(m, rank, 99);
    let mut h = random_nonneg(rank, n, 201);

    let ones_m = Array2::<f64>::ones((m, n));
    let mut prev_obj = f64::INFINITY;

    for _iter in 0..config.max_iter {
        let wh = matmul(&w, &h)?;

        // V / (W H + ε)
        let v_over_wh = Array2::from_shape_fn((m, n), |(i, j)| {
            v_arr[[i, j]] / (wh[[i, j]] + EPS)
        });

        // H ← H ⊙ (W^T (V/WH)) / (W^T 1)
        let wt_vwh = w.t().dot(&v_over_wh);
        let wt_ones = w.t().dot(&ones_m);
        for i in 0..rank {
            for j in 0..n {
                let num = wt_vwh[[i, j]] + EPS;
                let den = wt_ones[[i, j]] + EPS;
                h[[i, j]] *= num / den;
                h[[i, j]] = h[[i, j]].max(EPS);
            }
        }

        // Recompute WH after H update
        let wh2 = matmul(&w, &h)?;
        let v_over_wh2 = Array2::from_shape_fn((m, n), |(i, j)| {
            v_arr[[i, j]] / (wh2[[i, j]] + EPS)
        });

        // W ← W ⊙ ((V/WH) H^T) / (1 H^T)
        let vwh_ht = v_over_wh2.dot(&h.t());
        let ones_ht = ones_m.dot(&h.t());
        for i in 0..m {
            for j in 0..rank {
                let num = vwh_ht[[i, j]] + EPS;
                let den = ones_ht[[i, j]] + EPS;
                w[[i, j]] *= num / den;
                w[[i, j]] = w[[i, j]].max(EPS);
            }
        }

        // Convergence check
        let wh_final = matmul(&w, &h)?;
        let obj = kl_div(&v_arr, &wh_final);
        if (prev_obj - obj).abs() / (prev_obj.abs() + EPS) < config.tol {
            break;
        }
        prev_obj = obj;
    }

    Ok(NMF {
        w: arr2_to_vv(&w),
        h: arr2_to_vv(&h),
        rank,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Alternating Least Squares NMF
// ──────────────────────────────────────────────────────────────────────────────

/// NMF via Alternating Least Squares with non-negativity projection.
///
/// - H ← max(0, (W^T W)^{-1} W^T V)
/// - W ← max(0, V H^T (H H^T)^{-1})
pub fn als_nmf(
    v: &[Vec<f64>],
    rank: usize,
    config: &NMFConfig,
) -> SignalResult<NMF> {
    validate_nonneg(v, "V")?;
    let v_arr = vv_to_arr2(v)?;
    let (m, n) = v_arr.dim();
    if rank == 0 || rank > m.min(n) {
        return Err(SignalError::InvalidArgument(format!(
            "rank must be in 1..=min({m},{n})"
        )));
    }

    let mut w = random_nonneg(m, rank, 7);
    let mut h = random_nonneg(rank, n, 13);

    let mut prev_obj = f64::INFINITY;

    for _iter in 0..config.max_iter {
        // H step: solve (W^T W) H = W^T V  then project to >=0
        let wtw = w.t().dot(&w);
        let wtv = w.t().dot(&v_arr);
        // Solve via Cholesky-like inverse using simple LU (small system, rank×rank)
        match solve_small(&wtw, &wtv) {
            Ok(h_new) => {
                h = Array2::from_shape_fn((rank, n), |(i, j)| h_new[[i, j]].max(EPS));
            }
            Err(_) => {
                // Fall back to multiplicative update step
                let wt_v = w.t().dot(&v_arr);
                let wt_wh = w.t().dot(&w).dot(&h);
                for i in 0..rank {
                    for j in 0..n {
                        let num = (wt_v[[i, j]] + EPS).max(0.0);
                        let den = wt_wh[[i, j]] + EPS;
                        h[[i, j]] = (h[[i, j]] * num / den).max(EPS);
                    }
                }
            }
        }

        // W step: solve W (H H^T) = V H^T  then project to >=0
        let hht = h.dot(&h.t());
        let vht = v_arr.dot(&h.t());
        // W^T (H H^T) = V H^T  →  (H H^T)^T W^T = V H^T
        // equivalent: solve hht^T * W^T = vht^T
        match solve_small(&hht.t().to_owned(), &vht.t().to_owned()) {
            Ok(wt_new) => {
                w = Array2::from_shape_fn((m, rank), |(i, j)| wt_new[[j, i]].max(EPS));
            }
            Err(_) => {
                let v_ht = v_arr.dot(&h.t());
                let whht = w.dot(&hht);
                for i in 0..m {
                    for j in 0..rank {
                        let num = (v_ht[[i, j]] + EPS).max(0.0);
                        let den = whht[[i, j]] + EPS;
                        w[[i, j]] = (w[[i, j]] * num / den).max(EPS);
                    }
                }
            }
        }

        let wh = matmul(&w, &h)?;
        let obj = frob_sq(&v_arr, &wh);
        if (prev_obj - obj).abs() / (prev_obj + EPS) < config.tol {
            break;
        }
        prev_obj = obj;
    }

    Ok(NMF {
        w: arr2_to_vv(&w),
        h: arr2_to_vv(&h),
        rank,
    })
}

/// Solve the square system `A x = B` using Gaussian elimination with partial
/// pivoting. A is (k×k) and B is (k×n).
fn solve_small(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let k = a.nrows();
    if a.ncols() != k {
        return Err(SignalError::DimensionMismatch(
            "solve_small: A must be square".into(),
        ));
    }
    if b.nrows() != k {
        return Err(SignalError::DimensionMismatch(
            "solve_small: B rows must match A rows".into(),
        ));
    }
    let n = b.ncols();

    // Augmented matrix [A | B]  (k × (k+n))
    let mut aug = Array2::<f64>::zeros((k, k + n));
    for i in 0..k {
        for j in 0..k {
            aug[[i, j]] = a[[i, j]];
        }
        for j in 0..n {
            aug[[i, k + j]] = b[[i, j]];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..k {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "solve_small: singular matrix".into(),
            ));
        }
        // Swap rows
        if max_row != col {
            for j in 0..(k + n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in (col + 1)..k {
            let factor = aug[[row, col]] / pivot;
            for j in col..(k + n) {
                let sub = factor * aug[[col, j]];
                aug[[row, j]] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = Array2::<f64>::zeros((k, n));
    for rhs in 0..n {
        for row in (0..k).rev() {
            let mut val = aug[[row, k + rhs]];
            for col in (row + 1)..k {
                val -= aug[[row, col]] * x[[col, rhs]];
            }
            x[[row, rhs]] = val / aug[[row, row]];
        }
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random non-negative matrix V = W_true * H_true + noise.
    fn synthetic_nmf(m: usize, n: usize, r: usize) -> Vec<Vec<f64>> {
        // Use deterministic LCG
        let mut state: u64 = 1234;
        let rand_val = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };

        let mut w_true = vec![vec![0.0; r]; m];
        let mut h_true = vec![vec![0.0; n]; r];
        for i in 0..m { for j in 0..r { w_true[i][j] = rand_val(&mut state); } }
        for i in 0..r { for j in 0..n { h_true[i][j] = rand_val(&mut state); } }

        let mut v = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                for k in 0..r {
                    v[i][j] += w_true[i][k] * h_true[k][j];
                }
                // small noise
                v[i][j] += 0.001 * rand_val(&mut state);
            }
        }
        v
    }

    fn relative_error(v: &[Vec<f64>], nmf: &NMF) -> f64 {
        let v_arr = vv_to_arr2(v).expect("v arr");
        let wh_vv = nmf.reconstruct().expect("reconstruct");
        let wh_arr = vv_to_arr2(&wh_vv).expect("wh arr");
        let v_norm: f64 = v_arr.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-12 { return 0.0; }
        frob_sq(&v_arr, &wh_arr).sqrt() / v_norm
    }

    #[test]
    fn test_frobenius_nmf_convergence() {
        let v = synthetic_nmf(10, 20, 3);
        let config = NMFConfig { rank: 3, max_iter: 500, tol: 1e-6, ..Default::default() };
        let nmf = multiplicative_updates_frobenius(&v, 3, &config).expect("nmf_frob");
        let err = relative_error(&v, &nmf);
        assert!(err < 0.1, "Frobenius NMF relative error {err:.4} should be < 0.1");
    }

    #[test]
    fn test_kl_nmf_convergence() {
        let v = synthetic_nmf(8, 16, 2);
        let config = NMFConfig { rank: 2, max_iter: 500, tol: 1e-6, ..Default::default() };
        let nmf = multiplicative_updates_kl(&v, 2, &config).expect("nmf_kl");
        let err = relative_error(&v, &nmf);
        assert!(err < 0.15, "KL NMF relative error {err:.4} should be < 0.15");
    }

    #[test]
    fn test_als_nmf_convergence() {
        let v = synthetic_nmf(10, 20, 3);
        let config = NMFConfig { rank: 3, max_iter: 200, tol: 1e-5, ..Default::default() };
        let nmf = als_nmf(&v, 3, &config).expect("nmf_als");
        let err = relative_error(&v, &nmf);
        assert!(err < 0.1, "ALS NMF relative error {err:.4} should be < 0.1");
    }

    #[test]
    fn test_nmf_shapes() {
        let v = synthetic_nmf(6, 12, 2);
        let config = NMFConfig { rank: 2, ..Default::default() };
        let nmf = multiplicative_updates_frobenius(&v, 2, &config).expect("nmf");
        assert_eq!(nmf.w.len(), 6);
        assert_eq!(nmf.w[0].len(), 2);
        assert_eq!(nmf.h.len(), 2);
        assert_eq!(nmf.h[0].len(), 12);
    }

    #[test]
    fn test_nmf_nonneg_input_validation() {
        let mut v = synthetic_nmf(4, 8, 2);
        v[1][2] = -0.5;
        let config = NMFConfig::default();
        assert!(multiplicative_updates_frobenius(&v, 2, &config).is_err());
    }

    #[test]
    fn test_nmf_reconstruct() {
        let v = synthetic_nmf(5, 10, 2);
        let config = NMFConfig { rank: 2, max_iter: 300, ..Default::default() };
        let nmf = multiplicative_updates_frobenius(&v, 2, &config).expect("nmf");
        let recon = nmf.reconstruct().expect("reconstruct");
        assert_eq!(recon.len(), 5);
        assert_eq!(recon[0].len(), 10);
        // All entries non-negative
        for row in &recon {
            for &val in row {
                assert!(val >= 0.0, "Reconstructed value {val} should be non-negative");
            }
        }
    }

    #[test]
    fn test_fit_transform() {
        let v = synthetic_nmf(5, 10, 2);
        let config = NMFConfig { rank: 2, max_iter: 100, ..Default::default() };
        let nmf = multiplicative_updates_frobenius(&v, 2, &config).expect("nmf");
        let h = nmf.fit_transform(&v).expect("fit_transform");
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].len(), 10);
    }
}
