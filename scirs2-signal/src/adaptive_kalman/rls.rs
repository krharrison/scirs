//! Recursive Least Squares (RLS) adaptive filter.
//!
//! RLS converges much faster than LMS by recursively computing the exact
//! least-squares solution at each time step. It uses a forgetting factor λ
//! to track time-varying systems at the cost of O(N²) per step.
//!
//! # Algorithm
//!
//! Given N tap weights and scalar forgetting factor λ ∈ (0,1]:
//!
//! ```text
//! k(n)   = P(n-1)*x(n) / (λ + x^T(n)*P(n-1)*x(n))   (Kalman gain)
//! e(n)   = d(n) - w^T(n-1)*x(n)                        (a priori error)
//! w(n)   = w(n-1) + k(n)*e(n)
//! P(n)   = (1/λ) * [P(n-1) - k(n)*x^T(n)*P(n-1)]      (Riccati recursion)
//! ```
//!
//! Initialise: `w(0) = 0`, `P(0) = δ⁻¹ * I` (δ small for known initial state,
//! δ large for unknown).
//!
//! # References
//!
//! * Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall.
//! * Sayed, A.H. (2003). *Fundamentals of Adaptive Filtering*. Wiley-Interscience.

use crate::error::{SignalError, SignalResult};

/// Recursive Least Squares adaptive filter.
///
/// # Example
///
/// ```
/// use scirs2_signal::adaptive_kalman::rls::RlsFilter;
///
/// // Track a time-varying system with forgetting factor 0.99
/// let mut rls = RlsFilter::new(4, 0.99, 100.0).expect("create filter");
/// for n in 0..200 {
///     let x = (n as f64 * 0.1).sin();
///     let d = 0.5 * x + 0.3 * (n as f64 * 0.1 - 0.1).sin();
///     let y = rls.update(x, d).expect("update");
/// }
/// ```
pub struct RlsFilter {
    /// Adaptive weights w(n)
    weights: Vec<f64>,
    /// Circular input buffer
    buffer: Vec<f64>,
    /// Write index into buffer
    buf_idx: usize,
    /// Forgetting factor λ ∈ (0,1]
    lambda: f64,
    /// Inverse correlation matrix P(n) (order × order)
    p_matrix: Vec<Vec<f64>>,
    /// Filter order
    order: usize,
}

impl RlsFilter {
    /// Create a new RLS filter.
    ///
    /// # Arguments
    ///
    /// * `order`  - Number of filter taps
    /// * `lambda` - Forgetting factor (0.95–1.0 typical; 1.0 = no forgetting)
    /// * `delta`  - Initial value for diagonal of P matrix (1/δ controls initial weight)
    ///              Use large `delta` (e.g., 100) when initial state is unknown.
    pub fn new(order: usize, lambda: f64, delta: f64) -> SignalResult<Self> {
        if order == 0 {
            return Err(SignalError::ValueError("Order must be >= 1".to_string()));
        }
        if !(0.0 < lambda && lambda <= 1.0) {
            return Err(SignalError::ValueError(
                "lambda must be in (0, 1]".to_string(),
            ));
        }
        if delta <= 0.0 {
            return Err(SignalError::ValueError("delta must be positive".to_string()));
        }

        // P(0) = delta * I (large = uninformative prior)
        let mut p = vec![vec![0.0_f64; order]; order];
        for i in 0..order {
            p[i][i] = delta;
        }

        Ok(RlsFilter {
            weights: vec![0.0_f64; order],
            buffer: vec![0.0_f64; order],
            buf_idx: 0,
            lambda,
            p_matrix: p,
            order,
        })
    }

    /// Process one input-desired pair.
    ///
    /// # Returns
    ///
    /// Filter output `y(n)` (a priori estimate).
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        // Write input into circular buffer
        self.buffer[self.buf_idx] = x;
        self.buf_idx = (self.buf_idx + 1) % self.order;

        // Build input vector (most recent first)
        let x_vec: Vec<f64> = (0..self.order)
            .map(|i| {
                let idx = (self.buf_idx + self.order - 1 - i) % self.order;
                self.buffer[idx]
            })
            .collect();

        // A priori output: y = w^T * x
        let y: f64 = self.weights.iter().zip(x_vec.iter()).map(|(w, xi)| w * xi).sum();

        // A priori error: e = d - y
        let e = d - y;

        // Kalman gain vector k = P * x / (lambda + x^T * P * x)
        let px: Vec<f64> = (0..self.order)
            .map(|i| {
                self.p_matrix[i]
                    .iter()
                    .zip(x_vec.iter())
                    .map(|(pij, xj)| pij * xj)
                    .sum::<f64>()
            })
            .collect();

        let denom: f64 = self.lambda
            + x_vec.iter().zip(px.iter()).map(|(xi, pxi)| xi * pxi).sum::<f64>();

        if denom.abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "RLS denominator near zero".to_string(),
            ));
        }

        let k: Vec<f64> = px.iter().map(|&pxi| pxi / denom).collect();

        // Weight update: w(n) = w(n-1) + k * e
        for (w, &ki) in self.weights.iter_mut().zip(k.iter()) {
            *w += ki * e;
        }

        // Covariance update: P(n) = (I - k*x^T)*P(n-1) / lambda
        // Equivalent: P(n) = [P(n-1) - k * (P(n-1)*x)^T] / lambda
        //                   = [P(n-1) - k * px^T] / lambda
        let lambda_inv = 1.0 / self.lambda;
        let n = self.order;
        for i in 0..n {
            for j in 0..n {
                self.p_matrix[i][j] = lambda_inv * (self.p_matrix[i][j] - k[i] * px[j]);
            }
        }

        Ok(y)
    }

    /// Get the current filter weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the filter order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the forgetting factor.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get a reference to the inverse covariance matrix P.
    pub fn p_matrix(&self) -> &Vec<Vec<f64>> {
        &self.p_matrix
    }

    /// Reset the filter to initial state with a new delta value.
    pub fn reset(&mut self, delta: f64) -> SignalResult<()> {
        if delta <= 0.0 {
            return Err(SignalError::ValueError("delta must be positive".to_string()));
        }
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buf_idx = 0;
        for i in 0..self.order {
            for j in 0..self.order {
                self.p_matrix[i][j] = if i == j { delta } else { 0.0 };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// System identification: identify a 4-tap FIR.
    /// RLS should converge in very few samples compared to LMS.
    #[test]
    fn test_rls_fast_convergence() {
        let true_coeffs = [0.5_f64, 0.3, -0.2, 0.1];
        let order = true_coeffs.len();
        let mut rls = RlsFilter::new(order, 0.99, 100.0).expect("create filter");

        let mut lcg: u64 = 0xABCDEF01;
        let mut rand_sample = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let mut input_buf = vec![0.0_f64; order];
        // RLS needs far fewer iterations than LMS
        for _ in 0..500 {
            let x = rand_sample();
            for i in (1..order).rev() {
                input_buf[i] = input_buf[i - 1];
            }
            input_buf[0] = x;
            let d: f64 = true_coeffs.iter().zip(input_buf.iter()).map(|(c, xi)| c * xi).sum();
            rls.update(x, d).expect("update");
        }

        let w = rls.weights();
        for (i, (&w_est, &w_true)) in w.iter().zip(true_coeffs.iter()).enumerate() {
            assert!(
                (w_est - w_true).abs() < 0.02,
                "Tap {}: estimated={:.5}, true={:.5}",
                i, w_est, w_true
            );
        }
    }

    /// Track a time-varying system (coefficients change mid-stream).
    #[test]
    fn test_rls_tracks_time_varying_system() {
        let order = 2;
        let mut rls = RlsFilter::new(order, 0.95, 100.0).expect("create filter");

        let mut lcg: u64 = 0x12345678;
        let mut rand_sample = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        // Phase 1: true system = [0.9, 0.1]
        let mut input_buf = vec![0.0_f64; order];
        for _ in 0..300 {
            let x = rand_sample();
            input_buf[1] = input_buf[0];
            input_buf[0] = x;
            let d = 0.9 * input_buf[0] + 0.1 * input_buf[1];
            rls.update(x, d).expect("update phase 1");
        }

        let w_phase1 = rls.weights().to_vec();
        assert!(
            (w_phase1[0] - 0.9).abs() < 0.05,
            "Phase 1 tap 0: {:.4}",
            w_phase1[0]
        );

        // Phase 2: system abruptly changes to [-0.5, 0.8]
        for _ in 0..300 {
            let x = rand_sample();
            input_buf[1] = input_buf[0];
            input_buf[0] = x;
            let d = -0.5 * input_buf[0] + 0.8 * input_buf[1];
            rls.update(x, d).expect("update phase 2");
        }

        let w_phase2 = rls.weights();
        assert!(
            (w_phase2[0] - (-0.5)).abs() < 0.1,
            "Phase 2 tap 0: {:.4} should be near -0.5",
            w_phase2[0]
        );
    }

    #[test]
    fn test_rls_reset() {
        let mut rls = RlsFilter::new(3, 0.99, 10.0).expect("create filter");
        for _ in 0..50 {
            rls.update(1.0, 0.5).expect("update");
        }
        rls.reset(10.0).expect("reset");
        let w = rls.weights();
        assert!(w.iter().all(|&v| v == 0.0), "Weights should be zero after reset");
        let p = rls.p_matrix();
        assert!((p[0][0] - 10.0).abs() < 1e-10, "P should be reset to delta*I");
    }
}
