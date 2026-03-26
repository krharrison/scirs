//! Parametric linear smoother trained via gradient descent.
//!
//! The smoother applies the update rule
//!
//!   x_{k+1} = x_k + W · r_k,   where r_k = b − A x_k
//!
//! and W is a diagonal weight matrix initialised as ω D⁻¹ (weighted Jacobi).
//! Training minimises ‖x_{k+1} − x_exact‖² via gradient descent on W.

use crate::error::{SparseError, SparseResult};
use crate::learned_smoother::types::Smoother;

// ---------------------------------------------------------------------------
// CSR helpers (raw arrays, no CsrMatrix import)
// ---------------------------------------------------------------------------

/// Compute y = A * x where A is given in CSR format.
fn csr_matvec(a_values: &[f64], a_row_ptr: &[usize], a_col_idx: &[usize], x: &[f64]) -> Vec<f64> {
    let n = a_row_ptr.len().saturating_sub(1);
    let mut y = vec![0.0; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        let mut sum = 0.0;
        for pos in start..end {
            sum += a_values[pos] * x[a_col_idx[pos]];
        }
        y[i] = sum;
    }
    y
}

/// Compute residual r = b - A*x.
fn compute_residual(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    x: &[f64],
    b: &[f64],
) -> Vec<f64> {
    let ax = csr_matvec(a_values, a_row_ptr, a_col_idx, x);
    let n = b.len();
    let mut r = vec![0.0; n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    r
}

/// Extract the diagonal of A (CSR).
fn extract_diagonal(
    a_values: &[f64],
    a_row_ptr: &[usize],
    a_col_idx: &[usize],
    n: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; n];
    for i in 0..n {
        let start = a_row_ptr[i];
        let end = a_row_ptr[i + 1];
        for pos in start..end {
            if a_col_idx[pos] == i {
                diag[i] = a_values[pos];
                break;
            }
        }
    }
    diag
}

/// Euclidean norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// LinearSmoother
// ---------------------------------------------------------------------------

/// A parametric linear smoother with trainable diagonal weights.
///
/// The smoother computes x_{k+1} = x_k + W · r_k where W = diag(w) is a
/// diagonal matrix initialised from the weighted Jacobi smoother ω D⁻¹.
///
/// Training minimises the error ‖x_{k+1} − x*‖² by gradient descent on w.
#[derive(Debug, Clone)]
pub struct LinearSmoother {
    /// Diagonal weight vector (length n).
    weights: Vec<f64>,
    /// Problem dimension.
    n: usize,
    /// Initial relaxation parameter used for weight initialisation.
    omega: f64,
}

impl LinearSmoother {
    /// Create a new linear smoother initialised as ω D⁻¹.
    ///
    /// # Arguments
    /// - `a_diag`: diagonal entries of A
    /// - `n`: problem dimension
    /// - `omega`: relaxation parameter (typically 2/3)
    pub fn new(a_diag: &[f64], n: usize, omega: f64) -> Self {
        let weights: Vec<f64> = a_diag
            .iter()
            .map(|&d| {
                if d.abs() > f64::EPSILON {
                    omega / d
                } else {
                    omega
                }
            })
            .collect();
        Self { weights, n, omega }
    }

    /// Create from raw CSR data, extracting the diagonal internally.
    pub fn from_csr(
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        omega: f64,
    ) -> Self {
        let n = a_row_ptr.len().saturating_sub(1);
        let diag = extract_diagonal(a_values, a_row_ptr, a_col_idx, n);
        Self::new(&diag, n, omega)
    }

    /// Apply one smoothing sweep: x += W · r.
    fn apply_one_sweep(
        &self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
    ) {
        let r = compute_residual(a_values, a_row_ptr, a_col_idx, x, b);
        for i in 0..self.n {
            x[i] += self.weights[i] * r[i];
        }
    }

    /// Estimate the spectral radius of the error propagation operator (I − W A)
    /// using power iteration.
    ///
    /// Returns a value in [0, 1) for a convergent smoother.
    pub fn spectral_radius_estimate(
        &self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
    ) -> f64 {
        let n = self.n;
        if n == 0 {
            return 0.0;
        }

        // Start with a random-ish vector
        let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let mut nrm = vec_norm(&v);
        if nrm < f64::EPSILON {
            return 0.0;
        }
        for x in v.iter_mut() {
            *x /= nrm;
        }

        let max_iter = 100;
        let mut lambda = 0.0;

        for _ in 0..max_iter {
            // Compute w = (I - W A) v
            let av = csr_matvec(a_values, a_row_ptr, a_col_idx, &v);
            let mut w = vec![0.0; n];
            for i in 0..n {
                w[i] = v[i] - self.weights[i] * av[i];
            }

            nrm = vec_norm(&w);
            if nrm < f64::EPSILON {
                return 0.0;
            }

            let new_lambda = nrm;
            for x in w.iter_mut() {
                *x /= nrm;
            }
            v = w;

            if (new_lambda - lambda).abs() < 1e-10 {
                return new_lambda;
            }
            lambda = new_lambda;
        }

        lambda
    }

    /// Read-only access to the weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Problem dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The omega used for initialisation.
    pub fn omega(&self) -> f64 {
        self.omega
    }
}

impl Smoother for LinearSmoother {
    fn smooth(
        &self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        n_sweeps: usize,
    ) -> SparseResult<()> {
        let n = a_row_ptr.len().saturating_sub(1);
        if x.len() != n || b.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }
        for _ in 0..n_sweeps {
            self.apply_one_sweep(a_values, a_row_ptr, a_col_idx, x, b);
        }
        Ok(())
    }

    fn train_step(
        &mut self,
        a_values: &[f64],
        a_row_ptr: &[usize],
        a_col_idx: &[usize],
        x: &mut [f64],
        b: &[f64],
        x_exact: &[f64],
        lr: f64,
    ) -> SparseResult<f64> {
        let n = self.n;
        if x.len() != n || b.len() != n || x_exact.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }

        // Compute residual r = b - A x
        let r = compute_residual(a_values, a_row_ptr, a_col_idx, x, b);

        // Proposed update: x_new = x + W r
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            x_new[i] = x[i] + self.weights[i] * r[i];
        }

        // Error: e = x_new - x_exact
        let mut error = vec![0.0; n];
        for i in 0..n {
            error[i] = x_new[i] - x_exact[i];
        }

        // Loss = ‖e‖²
        let loss: f64 = error.iter().map(|e| e * e).sum();

        // Gradient: ∂loss/∂w_i = 2 * e_i * r_i
        for i in 0..n {
            let grad = 2.0 * error[i] * r[i];
            self.weights[i] -= lr * grad;
        }

        // Apply the update to x
        x[..n].copy_from_slice(&x_new[..n]);

        Ok(loss)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 3x3 SPD tridiagonal: [2 -1 0; -1 2 -1; 0 -1 2]
    fn tridiag_3() -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
        let row_ptr = vec![0, 2, 5, 7];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
        (values, row_ptr, col_idx)
    }

    #[test]
    fn test_linear_smoother_reduces_residual() {
        let (vals, rp, ci) = tridiag_3();
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);

        let b = vec![1.0, 0.0, 1.0];
        let mut x = vec![0.0; 3];

        let r0 = compute_residual(&vals, &rp, &ci, &x, &b);
        let norm0 = vec_norm(&r0);

        smoother
            .smooth(&vals, &rp, &ci, &mut x, &b, 10)
            .expect("smooth failed");

        let r1 = compute_residual(&vals, &rp, &ci, &x, &b);
        let norm1 = vec_norm(&r1);

        assert!(norm1 < norm0, "Residual should decrease after smoothing");
    }

    #[test]
    fn test_linear_smoother_training_reduces_loss() {
        let (vals, rp, ci) = tridiag_3();
        let mut smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);

        let b = vec![1.0, 0.0, 1.0];
        let x_exact = vec![1.0, 1.0, 1.0]; // exact solution of 2x-y=1, -x+2y-z=0, -y+2z=1

        let mut losses = Vec::new();
        for _ in 0..20 {
            let mut x = vec![0.0; 3];
            let loss = smoother
                .train_step(&vals, &rp, &ci, &mut x, &b, &x_exact, 0.01)
                .expect("train failed");
            losses.push(loss);
        }

        // Loss should generally decrease
        let first_losses: f64 = losses[..5].iter().sum::<f64>() / 5.0;
        let last_losses: f64 = losses[15..].iter().sum::<f64>() / 5.0;
        assert!(
            last_losses < first_losses,
            "Training should reduce average loss: first={first_losses}, last={last_losses}"
        );
    }

    #[test]
    fn test_spectral_radius_less_than_one() {
        let (vals, rp, ci) = tridiag_3();
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let rho = smoother.spectral_radius_estimate(&vals, &rp, &ci);
        assert!(
            rho < 1.0,
            "Spectral radius should be < 1 for convergent smoother, got {rho}"
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let (vals, rp, ci) = tridiag_3();
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 2.0 / 3.0);
        let b = vec![1.0, 0.0]; // wrong size
        let mut x = vec![0.0; 3];
        let result = smoother.smooth(&vals, &rp, &ci, &mut x, &b, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dimension_smoother() {
        let vals: Vec<f64> = vec![];
        let rp = vec![0usize];
        let ci: Vec<usize> = vec![];
        let smoother = LinearSmoother::from_csr(&vals, &rp, &ci, 0.5);
        assert_eq!(smoother.dim(), 0);
        let rho = smoother.spectral_radius_estimate(&vals, &rp, &ci);
        assert!((rho - 0.0).abs() < f64::EPSILON);
    }
}
