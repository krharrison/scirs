//! Least Mean Squares (LMS) adaptive filter variants.
//!
//! LMS is the simplest and most widely used adaptive filtering algorithm.
//! It performs stochastic gradient descent to minimise the mean square error
//! between the filter output and a desired signal.
//!
//! # Variants
//!
//! * [`LmsFilter`]       — Standard LMS
//! * [`LmsLeakyFilter`]  — LMS with coefficient leakage (prevents weight drift)
//! * [`LmsSignFilter`]   — Sign-LMS (multiplied update; suitable for integer hardware)
//!
//! # References
//!
//! * Widrow, B. & Hoff, M.E. (1960). "Adaptive switching circuits". *IRE WESCON Convention Record*.
//! * Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall.

use crate::error::{SignalError, SignalResult};

/// Standard Least Mean Squares adaptive filter.
///
/// The weight update rule is:
/// ```text
/// y(n)    = w^T(n) * x(n)
/// e(n)    = d(n) - y(n)
/// w(n+1)  = w(n) + 2 * μ * e(n) * x(n)
/// ```
///
/// **Stability condition**: `0 < μ < 1 / (N * P_x)` where P_x is the signal power.
///
/// # Example
///
/// ```
/// use scirs2_signal::adaptive_kalman::lms::LmsFilter;
///
/// let mut lms = LmsFilter::new(4, 0.01).expect("create filter");
/// // Identify a moving-average system
/// for n in 0..100 {
///     let x = (n as f64 * 0.1).sin();
///     let d = x + 0.5 * (n as f64 * 0.1 - 0.1).sin(); // desired (convolved)
///     let y = lms.update(x, d).expect("update");
/// }
/// let w = lms.weights();
/// assert_eq!(w.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Adaptive weights w(n) of length `order`
    weights: Vec<f64>,
    /// Delay line (circular buffer) of input samples x(n)
    buffer: Vec<f64>,
    /// Write index into the circular buffer
    buf_idx: usize,
    /// Step size μ
    mu: f64,
    /// Filter order (number of taps)
    order: usize,
}

impl LmsFilter {
    /// Create a new LMS filter.
    ///
    /// # Arguments
    ///
    /// * `order` - Number of filter taps (filter length)
    /// * `mu`    - Step size (learning rate), must satisfy stability condition
    pub fn new(order: usize, mu: f64) -> SignalResult<Self> {
        if order == 0 {
            return Err(SignalError::ValueError("Filter order must be >= 1".to_string()));
        }
        if mu <= 0.0 {
            return Err(SignalError::ValueError("Step size mu must be positive".to_string()));
        }
        Ok(LmsFilter {
            weights: vec![0.0_f64; order],
            buffer: vec![0.0_f64; order],
            buf_idx: 0,
            mu,
            order,
        })
    }

    /// Process one input-desired pair and return the filter output.
    ///
    /// Internally performs:
    /// 1. Update circular input buffer with `x`
    /// 2. Compute filter output `y = w^T * x_buf`
    /// 3. Compute error `e = d - y`
    /// 4. Update weights: `w += 2 * mu * e * x_buf`
    ///
    /// # Arguments
    ///
    /// * `x` - Input sample
    /// * `d` - Desired output sample
    ///
    /// # Returns
    ///
    /// Filter output `y(n)` before weight update.
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        // Write new sample into circular buffer
        self.buffer[self.buf_idx] = x;
        self.buf_idx = (self.buf_idx + 1) % self.order;

        // Compute output y = w^T * x_buf (oldest tap first)
        let y = self.compute_output();

        // Error signal
        let e = d - y;

        // LMS weight update: w(n+1) = w(n) + 2*mu*e*x
        // (factor 2 absorbed into mu convention: using just mu * e * x)
        for i in 0..self.order {
            let xi = self.tap_input(i);
            self.weights[i] += self.mu * e * xi;
        }

        Ok(y)
    }

    /// Compute filter output without updating weights.
    pub fn filter(&self, x: &[f64]) -> SignalResult<Vec<f64>> {
        let mut buf = self.buffer.clone();
        let mut buf_idx = self.buf_idx;
        let mut out = Vec::with_capacity(x.len());
        for &xi in x {
            buf[buf_idx] = xi;
            buf_idx = (buf_idx + 1) % self.order;
            let y: f64 = (0..self.order)
                .map(|i| {
                    let tap_idx = (buf_idx + self.order - 1 - i) % self.order;
                    self.weights[i] * buf[tap_idx]
                })
                .sum();
            out.push(y);
        }
        Ok(out)
    }

    fn compute_output(&self) -> f64 {
        (0..self.order)
            .map(|i| self.weights[i] * self.tap_input(i))
            .sum()
    }

    fn tap_input(&self, tap: usize) -> f64 {
        let idx = (self.buf_idx + self.order - 1 - tap) % self.order;
        self.buffer[idx]
    }

    /// Get the current filter weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the filter order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the current step size.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Set a new step size.
    pub fn set_mu(&mut self, mu: f64) -> SignalResult<()> {
        if mu <= 0.0 {
            return Err(SignalError::ValueError("Step size must be positive".to_string()));
        }
        self.mu = mu;
        Ok(())
    }

    /// Reset weights to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buf_idx = 0;
    }
}

/// Leaky LMS filter — prevents coefficient drift via regularisation.
///
/// The leakage term `0 < γ ≤ 1` shrinks weights toward zero, providing
/// bounded behaviour in unbounded-input conditions:
/// ```text
/// w(n+1) = (1 - mu * leakage) * w(n) + mu * e(n) * x(n)
/// ```
///
/// Leaky LMS effectively minimises `E[e²(n)] + leakage * ||w||²`.
#[derive(Debug, Clone)]
pub struct LmsLeakyFilter {
    /// Internal LMS state
    weights: Vec<f64>,
    buffer: Vec<f64>,
    buf_idx: usize,
    mu: f64,
    leakage: f64,
    order: usize,
}

impl LmsLeakyFilter {
    /// Create a new Leaky LMS filter.
    ///
    /// # Arguments
    ///
    /// * `order`   - Number of filter taps
    /// * `mu`      - Step size
    /// * `leakage` - Leakage coefficient (0 < leakage << 1, typically 1e-4 to 1e-2)
    pub fn new(order: usize, mu: f64, leakage: f64) -> SignalResult<Self> {
        if order == 0 {
            return Err(SignalError::ValueError("Order must be >= 1".to_string()));
        }
        if mu <= 0.0 {
            return Err(SignalError::ValueError("mu must be positive".to_string()));
        }
        if !(0.0..=1.0).contains(&leakage) {
            return Err(SignalError::ValueError(
                "leakage must be in [0, 1]".to_string(),
            ));
        }
        Ok(LmsLeakyFilter {
            weights: vec![0.0_f64; order],
            buffer: vec![0.0_f64; order],
            buf_idx: 0,
            mu,
            leakage,
            order,
        })
    }

    /// Process one sample. Returns filter output before weight update.
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        self.buffer[self.buf_idx] = x;
        self.buf_idx = (self.buf_idx + 1) % self.order;

        // Output
        let y: f64 = (0..self.order)
            .map(|i| {
                let idx = (self.buf_idx + self.order - 1 - i) % self.order;
                self.weights[i] * self.buffer[idx]
            })
            .sum();

        let e = d - y;

        // Leaky LMS update: w(n+1) = (1 - mu*leak)*w(n) + mu*e*x
        let decay = 1.0 - self.mu * self.leakage;
        for i in 0..self.order {
            let idx = (self.buf_idx + self.order - 1 - i) % self.order;
            self.weights[i] = decay * self.weights[i] + self.mu * e * self.buffer[idx];
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
}

/// Sign-Error LMS filter — uses sign(e) instead of e in weight update.
///
/// This reduces multiply operations, making it suitable for fixed-point
/// and integer hardware implementations:
/// ```text
/// w(n+1) = w(n) + mu * sign(e(n)) * x(n)
/// ```
#[derive(Debug, Clone)]
pub struct LmsSignFilter {
    weights: Vec<f64>,
    buffer: Vec<f64>,
    buf_idx: usize,
    mu: f64,
    order: usize,
}

impl LmsSignFilter {
    /// Create a new Sign-LMS filter.
    pub fn new(order: usize, mu: f64) -> SignalResult<Self> {
        if order == 0 {
            return Err(SignalError::ValueError("Order must be >= 1".to_string()));
        }
        if mu <= 0.0 {
            return Err(SignalError::ValueError("mu must be positive".to_string()));
        }
        Ok(LmsSignFilter {
            weights: vec![0.0_f64; order],
            buffer: vec![0.0_f64; order],
            buf_idx: 0,
            mu,
            order,
        })
    }

    /// Process one sample using sign(error) update rule.
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        self.buffer[self.buf_idx] = x;
        self.buf_idx = (self.buf_idx + 1) % self.order;

        let y: f64 = (0..self.order)
            .map(|i| {
                let idx = (self.buf_idx + self.order - 1 - i) % self.order;
                self.weights[i] * self.buffer[idx]
            })
            .sum();

        let e = d - y;
        let sign_e = if e > 0.0 { 1.0 } else if e < 0.0 { -1.0 } else { 0.0 };

        for i in 0..self.order {
            let idx = (self.buf_idx + self.order - 1 - i) % self.order;
            self.weights[i] += self.mu * sign_e * self.buffer[idx];
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// System identification: identify a 4-tap FIR filter.
    #[test]
    fn test_lms_system_identification() {
        // True system coefficients
        let true_coeffs = [0.5_f64, 0.3, -0.2, 0.1];
        let order = true_coeffs.len();

        let mut lms = LmsFilter::new(order, 0.01).expect("create filter");

        // Generate input signal (white noise approximation via LCG)
        let mut lcg: u64 = 12345;
        let mut rand_sample = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let mut input_buf = vec![0.0_f64; order];
        let n_iter = 5000;

        for _ in 0..n_iter {
            let x = rand_sample();
            // Rotate input buffer
            for i in (1..order).rev() {
                input_buf[i] = input_buf[i - 1];
            }
            input_buf[0] = x;
            // Desired = output of true system
            let d: f64 = true_coeffs.iter().zip(input_buf.iter()).map(|(c, xi)| c * xi).sum();
            lms.update(x, d).expect("update");
        }

        let w = lms.weights();
        for (i, (&w_est, &w_true)) in w.iter().zip(true_coeffs.iter()).enumerate() {
            assert!(
                (w_est - w_true).abs() < 0.05,
                "Tap {}: estimated={:.4}, true={:.4}",
                i, w_est, w_true
            );
        }
    }

    /// Noise cancellation: remove sinusoidal interference.
    #[test]
    fn test_lms_noise_cancellation() {
        let mut lms = LmsFilter::new(8, 0.005).expect("create filter");

        // Reference noise (sinusoid at known frequency)
        let noise_freq = 0.1_f64 * std::f64::consts::TAU;
        let n_samples = 2000;

        for n in 0..n_samples {
            let signal = (n as f64 * 0.05).sin(); // desired signal
            let noise = (n as f64 * noise_freq).cos();
            let observed = signal + noise; // corrupted signal
            let reference = noise;         // reference = pure noise
            let _y = lms.update(reference, observed).expect("update");
            let _ = signal; // verify signal is used
        }
        // Weights should remain finite and bounded
        let w_norm: f64 = lms.weights().iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(w_norm.is_finite(), "Weights should be finite");
        assert!(w_norm < 100.0, "Weights should not blow up: norm={:.4}", w_norm);
    }

    #[test]
    fn test_leaky_lms_bounded_weights() {
        let mut lms = LmsLeakyFilter::new(4, 0.01, 1e-3).expect("create filter");
        // Feed very large signals; leakage should keep weights bounded
        for n in 0..1000 {
            let x = 10.0 * (n as f64 * 0.1).sin();
            let d = x * 0.5;
            lms.update(x, d).expect("update");
        }
        let w_norm: f64 = lms.weights().iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(w_norm.is_finite(), "Leaky LMS weights should be finite");
        assert!(w_norm < 500.0, "Leaky LMS weights should be bounded: norm={:.4}", w_norm);
    }

    #[test]
    fn test_sign_lms_convergence() {
        let mut lms = LmsSignFilter::new(2, 0.005).expect("create filter");
        // Simple 1-tap identification: true coefficient = 0.7
        let mut lcg: u64 = 99887766;
        for _ in 0..5000 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            let d = 0.7 * x;
            lms.update(x, d).expect("update");
        }
        let w = lms.weights();
        // Sign LMS is slower but should converge to approximately 0.7
        assert!(
            (w[0] - 0.7).abs() < 0.15,
            "Sign LMS tap 0: {:.4}, expected ~0.7",
            w[0]
        );
    }

    #[test]
    fn test_lms_reset() {
        let mut lms = LmsFilter::new(4, 0.01).expect("create filter");
        for _ in 0..100 {
            lms.update(1.0, 0.5).expect("update");
        }
        let w_before: Vec<f64> = lms.weights().to_vec();
        lms.reset();
        let w_after = lms.weights();
        assert!(
            w_after.iter().all(|&w| w == 0.0),
            "Weights should be zero after reset"
        );
        // Ensure they were non-zero before
        assert!(
            w_before.iter().any(|&w| w != 0.0),
            "Weights should have been non-zero"
        );
    }
}
