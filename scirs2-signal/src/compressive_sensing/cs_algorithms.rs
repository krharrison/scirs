//! Compressed sensing recovery algorithms.
//!
//! Provides CoSaMP, Iterative Hard Thresholding (IHT), and threshold utilities.

use crate::error::{SignalError, SignalResult};

const EPS: f64 = 1e-12;

// ──────────────────────────────────────────────────────────────────────────────
// Threshold utilities
// ──────────────────────────────────────────────────────────────────────────────

/// Keep the `k` largest-magnitude entries, zero out the rest.
pub fn hard_threshold(v: &[f64], k: usize) -> Vec<f64> {
    if k == 0 || v.is_empty() {
        return vec![0.0; v.len()];
    }
    let k = k.min(v.len());
    let mut magnitudes: Vec<(usize, f64)> = v.iter().enumerate().map(|(i, &x)| (i, x.abs())).collect();
    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = magnitudes[k - 1].1;

    v.iter()
        .map(|&x| if x.abs() >= threshold - EPS { x } else { 0.0 })
        .collect()
}

/// Soft-thresholding operator: `S_λ(x) = sign(x) * max(|x| - λ, 0)`.
pub fn soft_threshold(v: &[f64], lambda: f64) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            let abs = x.abs();
            if abs <= lambda { 0.0 } else { x.signum() * (abs - lambda) }
        })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn matvec_t(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return vec![];
    }
    let n = a[0].len();
    let mut u = vec![0.0_f64; n];
    for (i, row) in a.iter().enumerate() {
        for (j, &aij) in row.iter().enumerate() {
            u[j] += aij * v[i];
        }
    }
    u
}

fn validate_dims(a: &[Vec<f64>], b: &[f64]) -> SignalResult<(usize, usize)> {
    let m = a.len();
    if m == 0 {
        return Err(SignalError::InvalidArgument("A has no rows".into()));
    }
    if b.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "A has {m} rows but b has {} entries",
            b.len()
        )));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(SignalError::InvalidArgument("A has no columns".into()));
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "Row {i} has {} columns, expected {n}",
                row.len()
            )));
        }
    }
    Ok((m, n))
}

/// Extract the support (non-zero indices) of a sparse vector.
fn support(x: &[f64]) -> Vec<usize> {
    x.iter().enumerate().filter(|&(_, &v)| v.abs() > EPS).map(|(i, _)| i).collect()
}

/// Union of two sorted index sets.
fn union_support(s1: &[usize], s2: &[usize]) -> Vec<usize> {
    let mut result: Vec<usize> = s1.to_vec();
    for &idx in s2 {
        if !result.contains(&idx) {
            result.push(idx);
        }
    }
    result.sort_unstable();
    result
}

/// Restricted least squares: min ||A_S x_S - b|| for support S.
/// Returns the full-length solution (zero outside S).
fn restricted_ls(a: &[Vec<f64>], b: &[f64], s: &[usize]) -> SignalResult<Vec<f64>> {
    let k = s.len();
    let m = a.len();
    let n = if m > 0 { a[0].len() } else { 0 };

    if k == 0 {
        return Ok(vec![0.0_f64; n]);
    }

    // Build A_S  (m × k)
    let a_s: Vec<Vec<f64>> = (0..m)
        .map(|i| s.iter().map(|&j| a[i][j]).collect())
        .collect();

    // Normal equations: A_S^T A_S c = A_S^T b  (k × k system)
    let at_a: Vec<Vec<f64>> = (0..k)
        .map(|i| (0..k).map(|j| {
            (0..m).map(|r| a_s[r][i] * a_s[r][j]).sum()
        }).collect())
        .collect();
    let at_b: Vec<f64> = (0..k).map(|i| (0..m).map(|r| a_s[r][i] * b[r]).sum()).collect();

    // Solve via Gaussian elimination with partial pivoting
    let coeff = gauss_solve(&at_a, &at_b)?;

    let mut x = vec![0.0_f64; n];
    for (pos, &idx) in s.iter().enumerate() {
        x[idx] = coeff[pos];
    }
    Ok(x)
}

/// Gaussian elimination with partial pivoting for square system (k × k).
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let k = a.len();
    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..k {
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Singular matrix in restricted LS".into(),
            ));
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for row in (col + 1)..k {
            let factor = aug[row][col] / pivot;
            for j in col..=k {
                let sub = factor * aug[col][j];
                aug[row][j] -= sub;
            }
        }
    }

    let mut x = vec![0.0_f64; k];
    for row in (0..k).rev() {
        let mut val = aug[row][k];
        for j in (row + 1)..k {
            val -= aug[row][j] * x[j];
        }
        x[row] = val / aug[row][row];
    }
    Ok(x)
}

// ──────────────────────────────────────────────────────────────────────────────
// CoSaMP
// ──────────────────────────────────────────────────────────────────────────────

/// CoSaMP (Compressive Sampling Matching Pursuit) recovery algorithm.
///
/// Greedy algorithm for recovering s-sparse signals from compressed measurements.
pub struct CoSaMP {
    /// Number of iterations.
    pub n_iters: usize,
    /// Target sparsity.
    pub sparsity: usize,
}

impl CoSaMP {
    /// Create a new CoSaMP instance.
    pub fn new(n_iters: usize, sparsity: usize) -> Self {
        Self { n_iters, sparsity }
    }

    /// Recover signal x from measurements b = Ax.
    ///
    /// # Arguments
    /// * `a` — measurement matrix (m × n)
    /// * `b` — measurement vector (m)
    ///
    /// # Returns
    /// Recovered signal estimate x̂ (n)
    pub fn recover(&self, a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
        let (_, n) = validate_dims(a, b)?;
        if self.sparsity == 0 {
            return Err(SignalError::InvalidArgument("sparsity must be > 0".into()));
        }
        let s = self.sparsity;

        let mut x = vec![0.0_f64; n];

        for _iter in 0..self.n_iters {
            // Step 1: Compute residual r = b - Ax
            let ax = matvec(a, &x);
            let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();

            // Step 2: Identify 2s largest components of A^T r
            let at_r = matvec_t(a, &residual);
            let omega = {
                let ht = hard_threshold(&at_r, 2 * s);
                support(&ht)
            };

            if omega.is_empty() {
                break;
            }

            // Step 3: Merge support of x with omega
            let current_support = support(&x);
            let merged = union_support(&current_support, &omega);

            // Step 4: Least squares on merged support
            let x_ls = restricted_ls(a, b, &merged)?;

            // Step 5: Keep s largest components
            x = hard_threshold(&x_ls, s);

            // Check convergence
            let ax_new = matvec(a, &x);
            let res_norm: f64 = b.iter().zip(ax_new.iter()).map(|(&bi, &axi)| (bi - axi).powi(2)).sum::<f64>().sqrt();
            if res_norm < EPS {
                break;
            }
        }

        Ok(x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// IHT — Iterative Hard Thresholding
// ──────────────────────────────────────────────────────────────────────────────

/// Iterative Hard Thresholding (IHT) recovery algorithm.
///
/// Update: `x_{t+1} = H_s(x_t + μ A^T(b - Ax_t))`
pub struct IHT {
    /// Gradient step size μ (use 1/||A||_2^2 for guaranteed convergence).
    pub step_size: f64,
    /// Target sparsity s.
    pub sparsity: usize,
    /// Maximum iterations.
    pub max_iter: usize,
}

impl IHT {
    /// Create a new IHT instance.
    pub fn new(step_size: f64, sparsity: usize, max_iter: usize) -> Self {
        Self { step_size, sparsity, max_iter }
    }

    /// Estimate a safe step size as 1/||A||_F^2 × m.
    pub fn auto_step_size(a: &[Vec<f64>]) -> f64 {
        let frob_sq: f64 = a.iter()
            .flat_map(|row| row.iter())
            .map(|&v| v * v)
            .sum();
        if frob_sq < EPS { 1.0 } else { 1.0 / frob_sq * a.len() as f64 }
    }

    /// Recover signal x from measurements b = Ax.
    pub fn recover(&self, a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
        let (_, n) = validate_dims(a, b)?;
        if self.sparsity == 0 {
            return Err(SignalError::InvalidArgument("sparsity must be > 0".into()));
        }

        let mut x = vec![0.0_f64; n];
        let mut prev_norm = f64::INFINITY;

        for _iter in 0..self.max_iter {
            // Gradient: A^T(b - Ax)
            let ax = matvec(a, &x);
            let residual: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();
            let at_r = matvec_t(a, &residual);

            // Gradient step
            let x_grad: Vec<f64> = x.iter().zip(at_r.iter()).map(|(&xi, &gi)| xi + self.step_size * gi).collect();

            // Hard threshold
            x = hard_threshold(&x_grad, self.sparsity);

            // Convergence check
            let cur_norm = norm2(&x);
            if (cur_norm - prev_norm).abs() < EPS * (1.0 + cur_norm) {
                break;
            }
            prev_norm = cur_norm;
        }

        Ok(x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build identity measurement problem with k-sparse signal.
    fn identity_problem(n: usize, nonzeros: &[(usize, f64)]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut x_true = vec![0.0_f64; n];
        for &(idx, val) in nonzeros {
            x_true[idx] = val;
        }
        let b = matvec(&a, &x_true);
        (a, b, x_true)
    }

    /// Build a random Gaussian measurement problem (seeded).
    fn gaussian_problem(m: usize, n: usize, nonzeros: &[(usize, f64)]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        use crate::compressive_sensing::measurements::GaussianMatrix;
        let a = GaussianMatrix::new(m, n).expect("gaussian");
        let mut x_true = vec![0.0_f64; n];
        for &(idx, val) in nonzeros {
            x_true[idx] = val;
        }
        let b = matvec(&a, &x_true);
        (a, b, x_true)
    }

    #[test]
    fn test_hard_threshold_basic() {
        let v = vec![1.0, -5.0, 2.0, -3.0, 4.0];
        let ht = hard_threshold(&v, 2);
        let nnz: Vec<usize> = ht.iter().enumerate().filter(|&(_, &x)| x.abs() > 1e-10).map(|(i, _)| i).collect();
        assert_eq!(nnz.len(), 2, "got {nnz:?}");
        // -5 and 4 should be the two largest
        assert!(ht[1].abs() > 1e-10);
        assert!(ht[4].abs() > 1e-10);
    }

    #[test]
    fn test_soft_threshold_basic() {
        let v = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let st = soft_threshold(&v, 2.0);
        assert!((st[0] - (-1.0)).abs() < 1e-10);
        assert_eq!(st[1], 0.0);
        assert_eq!(st[2], 0.0);
        assert_eq!(st[3], 0.0);
        assert!((st[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosamp_identity_recovery() {
        let (a, b, x_true) = identity_problem(16, &[(3, 2.5), (9, -1.5)]);
        let cosamp = CoSaMP::new(10, 2);
        let x_rec = cosamp.recover(&a, &b).expect("cosamp");
        let err: f64 = x_rec.iter().zip(x_true.iter()).map(|(&r, &t)| (r - t).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-8, "CoSaMP recovery error {err:.2e}");
    }

    #[test]
    fn test_iht_identity_recovery() {
        let (a, b, x_true) = identity_problem(16, &[(5, 3.0), (12, -2.0)]);
        let step = IHT::auto_step_size(&a);
        let iht = IHT::new(step, 2, 50);
        let x_rec = iht.recover(&a, &b).expect("iht");
        let err: f64 = x_rec.iter().zip(x_true.iter()).map(|(&r, &t)| (r - t).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-6, "IHT recovery error {err:.2e}");
    }

    #[test]
    fn test_cosamp_gaussian_measurement() {
        // m=30, n=64, 3-sparse signal
        let (a, b, x_true) = gaussian_problem(30, 64, &[(7, 2.0), (20, -1.5), (50, 3.0)]);
        let cosamp = CoSaMP::new(20, 3);
        let x_rec = cosamp.recover(&a, &b).expect("cosamp gaussian");
        let x_norm: f64 = x_true.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let err: f64 = x_rec.iter().zip(x_true.iter()).map(|(&r, &t)| (r - t).powi(2)).sum::<f64>().sqrt();
        assert!(err / x_norm < 0.1, "CoSaMP Gaussian error {:.4}", err / x_norm);
    }

    #[test]
    fn test_iht_gaussian_measurement() {
        let (a, b, x_true) = gaussian_problem(40, 64, &[(10, 3.0), (35, -2.0)]);
        let step = IHT::auto_step_size(&a);
        let iht = IHT::new(step, 2, 100);
        let x_rec = iht.recover(&a, &b).expect("iht gaussian");
        let x_norm: f64 = x_true.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let err: f64 = x_rec.iter().zip(x_true.iter()).map(|(&r, &t)| (r - t).powi(2)).sum::<f64>().sqrt();
        assert!(err / x_norm < 0.1, "IHT Gaussian error {:.4}", err / x_norm);
    }

    #[test]
    fn test_cosamp_invalid_sparsity() {
        let a: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 0.0];
        let cosamp = CoSaMP::new(5, 0);
        assert!(cosamp.recover(&a, &b).is_err());
    }

    #[test]
    fn test_iht_invalid_sparsity() {
        let a: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 0.0];
        let iht = IHT::new(1.0, 0, 10);
        assert!(iht.recover(&a, &b).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let a: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 0.0, 0.0];
        let cosamp = CoSaMP::new(5, 1);
        assert!(cosamp.recover(&a, &b).is_err());
        let iht = IHT::new(1.0, 1, 10);
        assert!(iht.recover(&a, &b).is_err());
    }

    #[test]
    fn test_hard_threshold_k_zero() {
        let v = vec![1.0, 2.0, 3.0];
        let ht = hard_threshold(&v, 0);
        assert!(ht.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_hard_threshold_k_all() {
        let v = vec![1.0, -2.0, 3.0];
        let ht = hard_threshold(&v, 10); // k > len
        assert_eq!(ht, v);
    }
}
