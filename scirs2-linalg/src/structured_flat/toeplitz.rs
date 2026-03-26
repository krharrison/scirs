//! Toeplitz matrix operations: O(N log N) matvec and O(N²) Levinson-Durbin solve.
//!
//! The matvec uses the classic circulant embedding trick:
//! embed the n×n Toeplitz T into a 2n×2n circulant C, then apply FFT-based
//! circulant matvec and extract the first n components.
//!
//! The solve uses the Levinson-Durbin algorithm which is O(N²) but exact.

use super::types::{FlatCirculant, FlatToeplitz};
use crate::error::{LinalgError, LinalgResult};

impl FlatToeplitz {
    /// Matrix-vector product y = T * x.
    ///
    /// Uses the circulant embedding trick: O(N log N).
    pub fn matvec(&self, x: &[f64]) -> LinalgResult<Vec<f64>> {
        let n = self.n;
        if x.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "x length {} ≠ matrix dim {}",
                x.len(),
                n
            )));
        }

        // Build the first *column* of the 2n circulant embedding
        // using the convention C[i,j] = c[(i-j+2n)%(2n)]:
        //   c[0..n]   = first_col[0..n]   (column 0 of T)
        //   c[n]      = 0                  (padding)
        //   c[n+1..2n] = first_row[n-1..1] reversed (anti-diagonal wrap)
        let two_n = 2 * n;
        let mut c_circ = vec![0.0_f64; two_n];
        // First n entries: first column of T.
        c_circ[..n].copy_from_slice(&self.first_col[..n]);
        // c_circ[n] = 0 (already zero).
        // Entries n+1 .. 2n-1: reversed first row (excluding diagonal).
        for i in 1..n {
            c_circ[two_n - i] = self.first_row[i];
        }

        // Build the circulant embedding and apply it.
        let circ = FlatCirculant::new(c_circ)?;

        // Pad x to length 2n.
        let mut x_pad = vec![0.0_f64; two_n];
        x_pad[..n].copy_from_slice(&x[..n]);

        let y_full = circ.matvec(&x_pad)?;
        // Extract first n elements.
        Ok(y_full[..n].to_vec())
    }

    /// Solve T * x = b using the Levinson-Durbin algorithm.
    ///
    /// Requires T to be a symmetric Toeplitz matrix (i.e., `first_row == first_col`).
    /// Complexity: O(N²).
    ///
    /// # Errors
    /// Returns [`LinalgError::SingularMatrixError`] if the algorithm detects a zero pivot.
    pub fn solve_symmetric(&self, b: &[f64]) -> LinalgResult<Vec<f64>> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "b length {} ≠ matrix dim {}",
                b.len(),
                n
            )));
        }
        levinson_durbin(&self.first_row, &self.first_col, b)
    }
}

/// Levinson-Durbin algorithm for solving a symmetric Toeplitz system T * x = b.
///
/// The Toeplitz matrix is symmetric, so `T[i,j] = t_row[|i-j|]`.
/// Implements Golub-Van Loan Algorithm 4.7.1 — the Levinson-Trench-Zohar
/// recursion. Complexity: O(N²).
///
/// # Errors
/// Returns [`LinalgError::SingularMatrixError`] if a near-zero pivot is encountered.
pub fn levinson_durbin(t_row: &[f64], _t_col: &[f64], b: &[f64]) -> LinalgResult<Vec<f64>> {
    levinson_clean(t_row, b)
}

/// Core symmetric Toeplitz solver (Golub-Van Loan Algorithm 4.7.1).
fn levinson_clean(t: &[f64], b: &[f64]) -> LinalgResult<Vec<f64>> {
    let n = t.len();

    if t[0].abs() < 1e-15 {
        return Err(LinalgError::SingularMatrixError(
            "Levinson: zero diagonal".into(),
        ));
    }

    // x is the solution; a is the auxiliary (autoregressive) vector.
    let mut x = vec![b[0] / t[0]];
    // a[j] is the reflection/predictor for the k-step system.
    // Initial a for k=1: empty (0-length auxiliary before first extension).
    let mut a: Vec<f64> = Vec::new();
    let mut alpha = t[0]; // current leading minor "determinant factor"

    for k in 0..n - 1 {
        // Step 1: rho_k = -(t[k+1] + sum_{j=0}^{k-1} t[j+1] * a[k-1-j]) / alpha
        let inner: f64 = (0..k).map(|j| t[j + 1] * a[k - 1 - j]).sum();
        let rho = -(t[k + 1] + inner) / alpha;

        // Step 2: alpha_{k+1} = alpha_k * (1 - rho_k^2)
        let alpha_new = alpha * (1.0 - rho * rho);
        if alpha_new.abs() < 1e-15 {
            return Err(LinalgError::SingularMatrixError(format!(
                "Levinson: near-zero pivot at step {}",
                k
            )));
        }

        // Step 3: a_new[j] = a[j] + rho * a[k-1-j] for j=0..k, then append rho.
        let a_old = a.clone();
        let mut a_new: Vec<f64> = (0..k).map(|j| a_old[j] + rho * a_old[k - 1 - j]).collect();
        a_new.push(rho);

        alpha = alpha_new;

        // Step 4: compute lambda_k = b[k+1] - sum_{j=0}^{k} t[k+1-j] * x[j]
        //                          = b[k+1] - sum_{j=0}^{k} t[|k+1-j|] * x[j]
        let lambda: f64 = b[k + 1] - (0..=k).map(|j| t[k + 1 - j] * x[j]).sum::<f64>();

        // Step 5: mu = lambda / alpha_new
        let mu = lambda / alpha_new;

        // Step 6: x_new[j] = x[j] + mu * a_new[k-j] for j=0..k, then append mu.
        let x_old = x.clone();
        let mut x_new: Vec<f64> = (0..=k).map(|j| x_old[j] + mu * a_new[k - j]).collect();
        x_new.push(mu);

        a = a_new;
        x = x_new;
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn mat_vec_explicit(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i * n + j] * x[j];
            }
        }
        y
    }

    #[test]
    fn test_toeplitz_matvec_vs_dense() {
        // Symmetric Toeplitz with first_row = [4,3,2,1].
        let row = vec![4.0_f64, 3.0, 2.0, 1.0];
        let col = row.clone(); // symmetric
        let t = FlatToeplitz::new(row.clone(), col).expect("failed");
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let y_struct = t.matvec(&x).expect("matvec failed");
        let dense = t.to_dense();
        let y_dense = mat_vec_explicit(&dense, 4, &x);
        for i in 0..4 {
            assert_relative_eq!(y_struct[i], y_dense[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_toeplitz_matvec_asymmetric() {
        // Non-symmetric Toeplitz.
        let row = vec![1.0_f64, 2.0, 3.0];
        let col = vec![1.0_f64, 4.0, 5.0];
        let t = FlatToeplitz::new(row, col).expect("failed");
        let x = vec![1.0_f64, 0.0, 0.0];
        let y = t.matvec(&x).expect("matvec failed");
        // First column of T = [1, 4, 5].
        assert_relative_eq!(y[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(y[1], 4.0, epsilon = 1e-8);
        assert_relative_eq!(y[2], 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_levinson_durbin_roundtrip() {
        // Solve a symmetric 4×4 Toeplitz system and verify T*x ≈ b.
        let row = vec![4.0_f64, 1.0, 0.5, 0.1];
        let col = row.clone();
        let t = FlatToeplitz::new(row.clone(), col.clone()).expect("failed");
        let b = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = t.solve_symmetric(&b).expect("solve failed");
        let y = t.matvec(&x).expect("matvec failed");
        for i in 0..4 {
            assert_relative_eq!(y[i], b[i], epsilon = 1e-7);
        }
    }

    #[test]
    fn test_levinson_durbin_1x1() {
        let row = vec![2.0_f64];
        let col = vec![2.0_f64];
        let b = vec![6.0_f64];
        let x = levinson_durbin(&row, &col, &b).expect("solve failed");
        assert_relative_eq!(x[0], 3.0, epsilon = 1e-12);
    }
}
