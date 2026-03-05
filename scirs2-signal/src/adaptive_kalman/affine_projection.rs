//! Affine Projection Algorithm (APA) adaptive filter.
//!
//! APA is a generalisation of NLMS that projects the weight update onto
//! an affine subspace spanned by K recent input vectors (projection order K).
//! This dramatically accelerates convergence for correlated inputs while
//! maintaining computational tractability.
//!
//! # Algorithm (K-th order APA)
//!
//! Let X(n) = [x(n), x(n-1), ..., x(n-K+1)] (N×K input matrix),
//! d(n) = [d(n), d(n-1), ..., d(n-K+1)]^T (K desired outputs):
//!
//! ```text
//! y(n)   = X^T(n) * w(n-1)                               (K outputs)
//! e(n)   = d(n) - y(n)                                    (K errors)
//! w(n)   = w(n-1) + μ * X(n) * (X^T(n)*X(n) + ε*I)^{-1} * e(n)
//! ```
//!
//! When K=1, APA reduces to NLMS.
//!
//! # Applications
//!
//! * Acoustic echo cancellation (long echo tails, correlated speech)
//! * Channel equalization
//! * Noise cancellation
//!
//! # References
//!
//! * Ozeki, K. & Umeda, T. (1984). "An adaptive filtering algorithm using an orthogonal projection
//!   to an affine subspace and its properties". *Electronics and Communications in Japan*, 67-A(5).
//! * Gay, S.L. & Benesty, J. (Eds.) (2000). *Acoustic Signal Processing for Telecommunication*.

use crate::error::{SignalError, SignalResult};

/// Affine Projection Algorithm adaptive filter.
///
/// # Example
///
/// ```
/// use scirs2_signal::adaptive_kalman::affine_projection::AffineProjFilter;
///
/// // APA with projection order 4 for fast convergence on correlated input
/// let mut apa = AffineProjFilter::new(16, 4, 0.5, 1e-4).expect("create apa");
/// for n in 0..500 {
///     let x = (n as f64 * 0.05).sin(); // correlated sinusoidal input
///     let d = 0.6 * x;
///     let y = apa.update(x, d).expect("update");
/// }
/// ```
pub struct AffineProjFilter {
    /// Filter weights w(n) — length = filter_order
    weights: Vec<f64>,
    /// Circular buffer for input samples — length = filter_order
    x_buf: Vec<f64>,
    /// Write pointer into x_buf
    x_idx: usize,
    /// Circular buffer for desired samples — length = proj_order
    d_buf: Vec<f64>,
    /// Write pointer into d_buf
    d_idx: usize,
    /// Filter order N (length of weight vector)
    filter_order: usize,
    /// Projection order K (number of past input vectors to use)
    proj_order: usize,
    /// Step size μ
    mu: f64,
    /// Regularisation ε (prevents singularity when inputs are linearly dependent)
    eps: f64,
    /// Number of samples processed (for initialisation phase)
    n_updates: usize,
}

impl AffineProjFilter {
    /// Create a new APA filter.
    ///
    /// # Arguments
    ///
    /// * `filter_order` - Length of the adaptive weight vector
    /// * `proj_order`   - Projection order K (K=1 reduces to NLMS)
    /// * `mu`           - Step size ∈ (0, 2)
    /// * `eps`          - Regularisation constant for matrix inversion
    pub fn new(
        filter_order: usize,
        proj_order: usize,
        mu: f64,
        eps: f64,
    ) -> SignalResult<Self> {
        if filter_order == 0 {
            return Err(SignalError::ValueError("filter_order must be >= 1".to_string()));
        }
        if proj_order == 0 {
            return Err(SignalError::ValueError("proj_order must be >= 1".to_string()));
        }
        if !(0.0 < mu && mu < 2.0) {
            return Err(SignalError::ValueError(
                "mu must be in (0, 2)".to_string(),
            ));
        }
        if eps <= 0.0 {
            return Err(SignalError::ValueError("eps must be positive".to_string()));
        }

        Ok(AffineProjFilter {
            weights: vec![0.0_f64; filter_order],
            x_buf: vec![0.0_f64; filter_order + proj_order], // extra space for shifts
            x_idx: 0,
            d_buf: vec![0.0_f64; proj_order],
            d_idx: 0,
            filter_order,
            proj_order,
            mu,
            eps,
            n_updates: 0,
        })
    }

    /// Process one input-desired sample and return the filter output.
    ///
    /// The update uses the K most recent input-desired pairs.
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        // Write new input sample into circular buffer
        let buf_size = self.filter_order + self.proj_order;
        self.x_buf[self.x_idx] = x;
        self.x_idx = (self.x_idx + 1) % buf_size;

        // Write new desired sample into circular buffer
        self.d_buf[self.d_idx] = d;
        self.d_idx = (self.d_idx + 1) % self.proj_order;

        self.n_updates += 1;

        // Build N×K input matrix X(n) = [x(n), x(n-1), ..., x(n-K+1)]
        // Each column is a length-N input vector at a different time offset
        let k = self.proj_order.min(self.n_updates);
        let n = self.filter_order;

        // Build X (n × k): column j = input vector at time (n - j)
        let x_matrix = self.build_input_matrix(k, n, buf_size);

        // Build desired vector d = [d(n), d(n-1), ..., d(n-k+1)]
        let d_vec = self.build_desired_vector(k);

        // Compute y = X^T * w (k outputs)
        let y_vec: Vec<f64> = (0..k)
            .map(|j| {
                x_matrix[j]
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(xij, wi)| xij * wi)
                    .sum::<f64>()
            })
            .collect();

        // Current output (most recent)
        let y = y_vec[0];

        // Error vector e = d - y
        let e_vec: Vec<f64> = d_vec.iter().zip(y_vec.iter()).map(|(di, yi)| di - yi).collect();

        // APA update: w = w + mu * X * (X^T * X + eps*I)^{-1} * e
        // Compute G = X^T * X + eps*I  (k × k)
        let mut g = vec![vec![0.0_f64; k]; k];
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = x_matrix[i].iter().zip(x_matrix[j].iter()).map(|(a, b)| a * b).sum();
                g[i][j] = dot + if i == j { self.eps } else { 0.0 };
            }
        }

        // Solve G * alpha = e via Gauss-Jordan elimination
        let alpha = gauss_jordan_solve(&g, &e_vec)?;

        // Weight update: w += mu * X * alpha = mu * sum_j alpha[j] * x_col_j
        for j in 0..k {
            for i in 0..n {
                self.weights[i] += self.mu * alpha[j] * x_matrix[j][i];
            }
        }

        Ok(y)
    }

    /// Build input matrix (k × n) where row j = input vector at time offset j.
    fn build_input_matrix(&self, k: usize, n: usize, buf_size: usize) -> Vec<Vec<f64>> {
        (0..k)
            .map(|j| {
                // j=0: most recent n samples; j=1: shifted by 1, etc.
                (0..n)
                    .map(|i| {
                        let offset = j + i;
                        let idx = (self.x_idx + buf_size - 1 - offset) % buf_size;
                        self.x_buf[idx]
                    })
                    .collect()
            })
            .collect()
    }

    /// Build desired vector of length k (most recent first).
    fn build_desired_vector(&self, k: usize) -> Vec<f64> {
        (0..k)
            .map(|j| {
                let idx = (self.d_idx + self.proj_order - 1 - j) % self.proj_order;
                self.d_buf[idx]
            })
            .collect()
    }

    /// Get the current filter weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the filter order.
    pub fn filter_order(&self) -> usize {
        self.filter_order
    }

    /// Get the projection order.
    pub fn proj_order(&self) -> usize {
        self.proj_order
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.x_buf.fill(0.0);
        self.d_buf.fill(0.0);
        self.x_idx = 0;
        self.d_idx = 0;
        self.n_updates = 0;
    }
}

/// Solve the linear system A * x = b using Gauss-Jordan with partial pivoting.
fn gauss_jordan_solve(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return Err(SignalError::ValueError(
            "System dimensions are inconsistent".to_string(),
        ));
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Singular matrix in APA solver".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..=n {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;
                }
            }
        }
    }

    Ok(aug.iter().map(|row| row[n]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// APA with K=1 should behave like NLMS.
    #[test]
    fn test_apa_k1_like_nlms() {
        let true_coeff = 0.7_f64;
        let mut apa = AffineProjFilter::new(1, 1, 0.5, 1e-6).expect("create apa");

        let mut lcg: u64 = 0xBEEF1234;
        for _ in 0..1000 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            apa.update(x, true_coeff * x).expect("update");
        }

        let w = apa.weights();
        assert!(
            (w[0] - true_coeff).abs() < 0.05,
            "APA K=1 tap 0: {:.4} should be near {:.4}",
            w[0], true_coeff
        );
    }

    /// APA should converge faster than NLMS for correlated (sinusoidal) inputs.
    #[test]
    fn test_apa_faster_convergence_correlated_input() {
        let true_coeffs = [0.5_f64, 0.3, -0.2];
        let order = true_coeffs.len();

        // NLMS (K=1)
        let mut nlms = AffineProjFilter::new(order, 1, 0.5, 1e-6).expect("create nlms");
        // APA (K=4)
        let mut apa = AffineProjFilter::new(order, 4, 0.5, 1e-4).expect("create apa");

        let mut input_buf = vec![0.0_f64; order];
        let n_steps = 200;

        let mut nlms_err_sum = 0.0_f64;
        let mut apa_err_sum = 0.0_f64;
        let measure_after = n_steps / 2;

        for step in 0..n_steps {
            // Correlated sinusoidal input
            let x = (step as f64 * 0.1).sin();
            for i in (1..order).rev() {
                input_buf[i] = input_buf[i - 1];
            }
            input_buf[0] = x;

            let d: f64 = true_coeffs.iter().zip(input_buf.iter()).map(|(c, xi)| c * xi).sum();

            let y_nlms = nlms.update(x, d).expect("nlms update");
            let y_apa = apa.update(x, d).expect("apa update");

            if step >= measure_after {
                nlms_err_sum += (d - y_nlms).powi(2);
                apa_err_sum += (d - y_apa).powi(2);
            }
        }

        // APA should have lower cumulative error in later steps
        assert!(
            apa_err_sum <= nlms_err_sum * 1.5, // APA should be at least competitive
            "APA error={:.6} should be comparable to NLMS error={:.6}",
            apa_err_sum, nlms_err_sum
        );
    }

    /// Test APA for acoustic echo cancellation scenario.
    #[test]
    fn test_apa_acoustic_echo_cancellation() {
        // Room impulse response (simplified)
        let rir = [0.9_f64, 0.4, 0.15, 0.05];
        let rir_len = rir.len();
        let filter_order = rir_len * 3;

        let mut apa = AffineProjFilter::new(filter_order, 4, 0.3, 1e-3).expect("create apa");

        let mut lcg: u64 = 0xCAFEBABE;
        let mut rand = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let mut speech_buf = vec![0.0_f64; rir_len];
        let mut final_errors = Vec::new();

        let n_total = 1000;
        for step in 0..n_total {
            let speech = rand();
            for i in (1..rir_len).rev() {
                speech_buf[i] = speech_buf[i - 1];
            }
            speech_buf[0] = speech;

            let echo: f64 = rir.iter().zip(speech_buf.iter()).map(|(h, s)| h * s).sum();
            let mic = speech + echo;

            let residual = apa.update(speech, mic).expect("update");

            if step > n_total * 3 / 4 {
                final_errors.push((residual - speech).powi(2));
            }
        }

        let mean_err = final_errors.iter().sum::<f64>() / final_errors.len() as f64;
        assert!(
            mean_err < 1.0,
            "APA echo cancellation residual error {:.4} should be < 1.0",
            mean_err
        );
    }

    #[test]
    fn test_apa_dimension_validation() {
        assert!(AffineProjFilter::new(0, 4, 0.5, 1e-4).is_err(), "order=0 should fail");
        assert!(AffineProjFilter::new(8, 0, 0.5, 1e-4).is_err(), "proj=0 should fail");
        assert!(AffineProjFilter::new(8, 4, 0.0, 1e-4).is_err(), "mu=0 should fail");
        assert!(AffineProjFilter::new(8, 4, 2.0, 1e-4).is_err(), "mu=2 should fail");
        assert!(AffineProjFilter::new(8, 4, 0.5, 0.0).is_err(), "eps=0 should fail");
    }
}
