//! Normalized Least Mean Squares (NLMS) adaptive filter.
//!
//! NLMS normalises the LMS step size by the instantaneous input power:
//! ```text
//! y(n)   = w^T(n) * x(n)
//! e(n)   = d(n) - y(n)
//! μ_n    = μ / (ε + ||x(n)||²)
//! w(n+1) = w(n) + μ_n * e(n) * x(n)
//! ```
//!
//! This provides **scale-invariant** convergence: the effective step size
//! automatically adapts to signal power, giving more stable convergence
//! than plain LMS for non-stationary or coloured input signals.
//!
//! # References
//!
//! * Nagumo, J.M. & Noda, A. (1967). "A learning method for system identification".
//!   *IEEE Trans. Automatic Control*, 12(3), 282–287.
//! * Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall.

use crate::error::{SignalError, SignalResult};

/// Normalized Least Mean Squares adaptive filter.
///
/// # Example
///
/// ```
/// use scirs2_signal::adaptive_kalman::nlms::NlmsFilter;
///
/// let mut nlms = NlmsFilter::new(8, 0.5, 1e-6).expect("create filter");
/// for n in 0..200 {
///     let x = (n as f64 * 0.1).sin() * 10.0; // high-amplitude signal
///     let d = 0.7 * x;
///     let y = nlms.update(x, d).expect("update");
/// }
/// let w = nlms.weights();
/// assert!((w[0] - 0.7).abs() < 0.05);
/// ```
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Adaptive weights
    weights: Vec<f64>,
    /// Circular input buffer
    buffer: Vec<f64>,
    /// Buffer write index
    buf_idx: usize,
    /// Normalised step size μ ∈ (0, 2)
    mu: f64,
    /// Regularisation to avoid division by zero
    eps: f64,
    /// Filter order
    order: usize,
    /// Running input power estimate (optional power smoothing)
    power: f64,
    /// Power smoothing coefficient (1.0 = exact, < 1 = exponential moving average)
    power_alpha: f64,
}

impl NlmsFilter {
    /// Create a new NLMS filter.
    ///
    /// # Arguments
    ///
    /// * `order` - Number of filter taps
    /// * `mu`    - Normalised step size, must satisfy `0 < μ < 2` for convergence
    /// * `eps`   - Regularisation constant (prevents division by zero)
    pub fn new(order: usize, mu: f64, eps: f64) -> SignalResult<Self> {
        if order == 0 {
            return Err(SignalError::ValueError("Order must be >= 1".to_string()));
        }
        if !(0.0 < mu && mu < 2.0) {
            return Err(SignalError::ValueError(
                "mu must be in (0, 2) for NLMS convergence".to_string(),
            ));
        }
        if eps <= 0.0 {
            return Err(SignalError::ValueError("eps must be positive".to_string()));
        }
        Ok(NlmsFilter {
            weights: vec![0.0_f64; order],
            buffer: vec![0.0_f64; order],
            buf_idx: 0,
            mu,
            eps,
            order,
            power: 0.0,
            power_alpha: 1.0, // exact power (no smoothing)
        })
    }

    /// Configure exponential moving average for input power estimation.
    ///
    /// Using `power_alpha < 1` gives a smoother power estimate and can
    /// reduce noise sensitivity.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor ∈ (0, 1]; 1.0 = exact instantaneous power
    pub fn set_power_smoothing(&mut self, alpha: f64) -> SignalResult<()> {
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(SignalError::ValueError(
                "power_alpha must be in (0, 1]".to_string(),
            ));
        }
        self.power_alpha = alpha;
        Ok(())
    }

    /// Process one input-desired sample pair.
    ///
    /// # Returns
    ///
    /// Filter output `y(n)` before weight update.
    pub fn update(&mut self, x: f64, d: f64) -> SignalResult<f64> {
        // Write new sample into circular buffer
        self.buffer[self.buf_idx] = x;
        self.buf_idx = (self.buf_idx + 1) % self.order;

        // Build current input vector (most recent first)
        let x_vec: Vec<f64> = (0..self.order)
            .map(|i| {
                let idx = (self.buf_idx + self.order - 1 - i) % self.order;
                self.buffer[idx]
            })
            .collect();

        // Input power ||x||² (possibly smoothed)
        let inst_power: f64 = x_vec.iter().map(|xi| xi * xi).sum();
        if self.power_alpha < 1.0 {
            self.power = self.power_alpha * inst_power + (1.0 - self.power_alpha) * self.power;
        } else {
            self.power = inst_power;
        }

        // Normalised step size
        let mu_n = self.mu / (self.eps + self.power);

        // Filter output y = w^T * x
        let y: f64 = self.weights.iter().zip(x_vec.iter()).map(|(w, xi)| w * xi).sum();

        // Error
        let e = d - y;

        // NLMS weight update: w(n+1) = w(n) + mu_n * e * x
        for (w, xi) in self.weights.iter_mut().zip(x_vec.iter()) {
            *w += mu_n * e * xi;
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

    /// Get the step size.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the regularisation constant.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Reset weights and buffer to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buf_idx = 0;
        self.power = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NLMS should be insensitive to signal amplitude scaling.
    #[test]
    fn test_nlms_scale_invariance() {
        let true_coeff = 0.6_f64;

        // Test with small amplitude
        let mut nlms_small = NlmsFilter::new(1, 0.5, 1e-6).expect("create small");
        let mut lcg: u64 = 0xFEDCBA98;
        let mut rand = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        for _ in 0..500 {
            let x = rand() * 0.01; // small amplitude
            nlms_small.update(x, true_coeff * x).expect("update");
        }

        // Test with large amplitude
        let mut nlms_large = NlmsFilter::new(1, 0.5, 1e-6).expect("create large");
        lcg = 0xFEDCBA98; // same seed
        let mut rand2 = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };
        for _ in 0..500 {
            let x = rand2() * 100.0; // large amplitude
            nlms_large.update(x, true_coeff * x).expect("update");
        }

        // Both should converge to approximately the same weight
        let w_small = nlms_small.weights()[0];
        let w_large = nlms_large.weights()[0];
        assert!(
            (w_small - true_coeff).abs() < 0.05,
            "NLMS (small): weight={:.4}, expected {:.4}",
            w_small, true_coeff
        );
        assert!(
            (w_large - true_coeff).abs() < 0.05,
            "NLMS (large): weight={:.4}, expected {:.4}",
            w_large, true_coeff
        );
    }

    /// Echo cancellation simulation.
    #[test]
    fn test_nlms_echo_cancellation() {
        let room_impulse = [0.8_f64, 0.4, 0.2, 0.1, 0.05]; // simulated room IR
        let ir_len = room_impulse.len();
        let filter_order = ir_len * 2; // over-model

        let mut nlms = NlmsFilter::new(filter_order, 0.5, 1e-4).expect("create nlms");

        let mut lcg: u64 = 0x9ABCDEF0;
        let mut rand = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let mut speech_buf = vec![0.0_f64; ir_len];
        let mut echo_errors = Vec::new();

        for step in 0..1000 {
            let speech = rand();
            // Shift buffer
            for i in (1..ir_len).rev() {
                speech_buf[i] = speech_buf[i - 1];
            }
            speech_buf[0] = speech;

            // Microphone signal = speech + echo
            let echo: f64 = room_impulse
                .iter()
                .zip(speech_buf.iter())
                .map(|(h, s)| h * s)
                .sum();
            let mic_signal = speech + echo;

            // NLMS cancels echo using near-end speech as reference
            let _residual = nlms.update(speech, mic_signal).expect("update nlms");

            if step > 800 {
                // Measure residual error (should be close to speech)
                let error_sq = (_residual - speech).powi(2);
                echo_errors.push(error_sq);
            }
        }

        let mean_error: f64 = echo_errors.iter().sum::<f64>() / echo_errors.len() as f64;
        // After convergence, residual error should be small
        assert!(
            mean_error < 0.5,
            "Echo cancellation residual error {:.4} should be small",
            mean_error
        );
    }

    #[test]
    fn test_nlms_multi_tap_system_id() {
        let true_coeffs = [0.5_f64, -0.3, 0.2, -0.1];
        let order = true_coeffs.len();
        let mut nlms = NlmsFilter::new(order, 0.5, 1e-6).expect("create nlms");

        let mut lcg: u64 = 0x11223344;
        let mut input_buf = vec![0.0_f64; order];

        for _ in 0..2000 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((lcg >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            for i in (1..order).rev() {
                input_buf[i] = input_buf[i - 1];
            }
            input_buf[0] = x;
            let d: f64 = true_coeffs.iter().zip(input_buf.iter()).map(|(c, xi)| c * xi).sum();
            nlms.update(x, d).expect("update");
        }

        let w = nlms.weights();
        for (i, (&w_est, &w_true)) in w.iter().zip(true_coeffs.iter()).enumerate() {
            assert!(
                (w_est - w_true).abs() < 0.05,
                "NLMS tap {}: est={:.4}, true={:.4}",
                i, w_est, w_true
            );
        }
    }

    #[test]
    fn test_nlms_mu_bounds() {
        assert!(NlmsFilter::new(4, 0.0, 1e-6).is_err(), "mu=0 should fail");
        assert!(NlmsFilter::new(4, 2.0, 1e-6).is_err(), "mu=2.0 should fail");
        assert!(NlmsFilter::new(4, -1.0, 1e-6).is_err(), "mu<0 should fail");
        assert!(NlmsFilter::new(4, 1.0, 1e-6).is_ok(), "mu=1.0 should succeed");
    }
}
