//! NLMS and RLS adaptive filters for acoustic echo cancellation.
//!
//! This module provides two complementary adaptive filter algorithms:
//!
//! - **NLMS** (Normalized Least Mean Squares): O(N) per sample, robust and
//!   widely deployed in real-time AEC applications.
//! - **RLS** (Recursive Least Squares): O(N²) per sample, faster convergence
//!   but higher computational cost.  Intended as a research reference.
//!
//! # Usage
//!
//! ```
//! use scirs2_signal::echo_cancellation::nlms::{NlmsConfig, NlmsFilter};
//!
//! let mut filter = NlmsFilter::new(NlmsConfig::default());
//! // reference: loudspeaker signal, microphone: mic (echo + near-end)
//! let cleaned = filter.update(0.5, 0.48); // echo-cancelled sample
//! ```

use std::collections::VecDeque;

// ── NLMS ─────────────────────────────────────────────────────────────────────

/// Configuration for the NLMS adaptive filter.
#[derive(Debug, Clone)]
pub struct NlmsConfig {
    /// Length of the adaptive FIR filter (number of taps).
    pub filter_length: usize,
    /// Normalized step size μ ∈ (0, 2).  Typical values: 0.3–0.8.
    pub step_size: f64,
    /// Regularization constant ε > 0 to prevent division by zero.
    pub regularization: f64,
    /// Leakage factor λ ∈ (0, 1].  1.0 disables leakage.
    pub leakage: f64,
}

impl Default for NlmsConfig {
    fn default() -> Self {
        Self {
            filter_length: 512,
            step_size: 0.5,
            regularization: 1e-6,
            leakage: 1.0,
        }
    }
}

/// Normalized Least Mean Squares (NLMS) adaptive filter for echo cancellation.
///
/// Given a reference signal x (far-end/loudspeaker) and a microphone signal d,
/// the filter estimates the echo path and returns the residual `d - ŷ`.
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Adaptive weights (echo-path estimate).
    pub weights: Vec<f64>,
    /// Circular input buffer for the reference signal.
    pub buffer: VecDeque<f64>,
    /// Filter configuration.
    pub config: NlmsConfig,
}

impl NlmsFilter {
    /// Create a new NLMS filter with the given configuration.
    pub fn new(config: NlmsConfig) -> Self {
        let n = config.filter_length;
        Self {
            weights: vec![0.0; n],
            buffer: VecDeque::from(vec![0.0; n]),
            config,
        }
    }

    /// Process one sample pair and return the echo-cancelled output.
    ///
    /// # Arguments
    ///
    /// * `reference`  – Far-end (loudspeaker) sample x(n).
    /// * `microphone` – Microphone (near-end + echo) sample d(n).
    ///
    /// # Returns
    ///
    /// Echo-cancelled error signal e(n) = d(n) − ŷ(n).
    pub fn update(&mut self, reference: f64, microphone: f64) -> f64 {
        // Shift reference sample into the buffer
        self.buffer.pop_back();
        self.buffer.push_front(reference);

        let n = self.config.filter_length;
        // Compute echo estimate: ŷ = w^T · x_buf
        let echo_estimate: f64 = self
            .weights
            .iter()
            .zip(self.buffer.iter())
            .map(|(w, x)| w * x)
            .sum();

        let error = microphone - echo_estimate;

        // NLMS update:  normalization = x_buf^T · x_buf + ε
        let power: f64 = self.buffer.iter().map(|x| x * x).sum::<f64>();
        let normalization = power + self.config.regularization;
        let mu_norm = self.config.step_size / normalization;

        // w_new = leakage * w + (μ/norm) * e * x_buf
        for (i, w) in self.weights.iter_mut().enumerate() {
            let xi = if i < n {
                *self.buffer.get(i).unwrap_or(&0.0)
            } else {
                0.0
            };
            *w = self.config.leakage * *w + mu_norm * error * xi;
        }

        error
    }

    /// Process an entire block of samples.
    ///
    /// # Arguments
    ///
    /// * `reference`  – Slice of far-end samples.
    /// * `microphone` – Slice of microphone samples (same length).
    ///
    /// # Returns
    ///
    /// Vector of echo-cancelled samples.
    pub fn process_block(&mut self, reference: &[f64], microphone: &[f64]) -> Vec<f64> {
        let len = reference.len().min(microphone.len());
        (0..len)
            .map(|i| self.update(reference[i], microphone[i]))
            .collect()
    }

    /// Reset the filter to its initial zero state.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ── RLS ──────────────────────────────────────────────────────────────────────

/// Configuration for the RLS adaptive filter.
#[derive(Debug, Clone)]
pub struct RlsConfig {
    /// Length of the adaptive FIR filter (number of taps).
    ///
    /// **Note**: RLS complexity is O(N²) per sample.  For production AEC use
    /// `filter_length` ≤ 256 or switch to the NLMS filter for longer paths.
    pub filter_length: usize,
    /// Forgetting factor λ ∈ (0, 1].  Values near 1.0 track slowly changing
    /// echo paths; lower values adapt faster but are less stable.
    pub forgetting_factor: f64,
    /// Initial value of the diagonal of P (= δ⁻¹ · I).
    ///
    /// Larger values correspond to less prior knowledge about the echo path.
    pub delta: f64,
}

impl Default for RlsConfig {
    fn default() -> Self {
        Self {
            filter_length: 256,
            forgetting_factor: 0.99,
            delta: 1.0,
        }
    }
}

/// Recursive Least Squares (RLS) adaptive filter for echo cancellation.
///
/// RLS achieves faster convergence than NLMS at the cost of O(N²) per-sample
/// complexity.  This is a research-reference implementation; for long echo
/// paths consider a partitioned-block or QR-based variant.
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Adaptive weights (echo-path estimate).
    pub weights: Vec<f64>,
    /// Inverse correlation matrix P, shape filter_length × filter_length.
    pub p_matrix: Vec<Vec<f64>>,
    /// Circular input buffer for the reference signal.
    pub buffer: VecDeque<f64>,
    /// RLS configuration.
    pub config: RlsConfig,
}

impl RlsFilter {
    /// Create a new RLS filter with the given configuration.
    ///
    /// P is initialised as (1/δ)·I.
    pub fn new(config: RlsConfig) -> Self {
        let n = config.filter_length;
        let init_p = 1.0 / config.delta.max(1e-12);
        let mut p_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            p_matrix[i][i] = init_p;
        }
        Self {
            weights: vec![0.0; n],
            p_matrix,
            buffer: VecDeque::from(vec![0.0; n]),
            config,
        }
    }

    /// Process one sample pair and return the echo-cancelled output.
    ///
    /// Implements the standard RLS update:
    ///
    /// ```text
    /// k   = P·x / (λ + x^T·P·x)
    /// e   = d − w^T·x
    /// w  += k · e
    /// P   = (P − k·(P·x)^T) / λ
    /// ```
    ///
    /// # Returns
    ///
    /// Echo-cancelled error signal e(n) = d(n) − ŷ(n).
    pub fn update(&mut self, reference: f64, microphone: f64) -> f64 {
        // Shift reference into buffer
        self.buffer.pop_back();
        self.buffer.push_front(reference);

        let n = self.config.filter_length;
        let lam = self.config.forgetting_factor;

        // Collect buffer as a flat vector
        let x: Vec<f64> = self.buffer.iter().copied().collect();

        // Compute P·x  (n-vector)
        let px: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| self.p_matrix[i][j] * x[j]).sum::<f64>())
            .collect();

        // Scalar denominator: λ + x^T·P·x
        let xt_px: f64 = x.iter().zip(px.iter()).map(|(xi, pxi)| xi * pxi).sum();
        let denom = lam + xt_px;

        // Kalman gain: k = P·x / denom
        let k: Vec<f64> = px.iter().map(|pxi| pxi / denom).collect();

        // Echo estimate and error
        let echo: f64 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        let error = microphone - echo;

        // Weight update: w += k · e
        for (w, ki) in self.weights.iter_mut().zip(k.iter()) {
            *w += ki * error;
        }

        // P update: P = (P − k·(P·x)^T) / λ
        // P·x == px  (already computed)
        for i in 0..n {
            for j in 0..n {
                self.p_matrix[i][j] = (self.p_matrix[i][j] - k[i] * px[j]) / lam;
            }
        }

        error
    }

    /// Process an entire block of samples.
    ///
    /// # Arguments
    ///
    /// * `reference`  – Slice of far-end samples.
    /// * `microphone` – Slice of microphone samples (same length).
    ///
    /// # Returns
    ///
    /// Vector of echo-cancelled samples.
    pub fn process_block(&mut self, reference: &[f64], microphone: &[f64]) -> Vec<f64> {
        let len = reference.len().min(microphone.len());
        (0..len)
            .map(|i| self.update(reference[i], microphone[i]))
            .collect()
    }

    /// Reset filter to initial zero state.
    pub fn reset(&mut self) {
        let n = self.config.filter_length;
        let init_p = 1.0 / self.config.delta.max(1e-12);
        self.weights.fill(0.0);
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n {
            for j in 0..n {
                self.p_matrix[i][j] = if i == j { init_p } else { 0.0 };
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlms_pure_echo_converges_to_zero() {
        // When microphone == reference (pure echo, no near-end), the NLMS
        // filter should drive the error toward zero.
        let cfg = NlmsConfig {
            filter_length: 32,
            step_size: 0.5,
            regularization: 1e-6,
            leakage: 1.0,
        };
        let mut f = NlmsFilter::new(cfg);

        // Simulate a single-tap echo path: mic[n] = 0.8 * ref[n]
        let n_samples = 2000;
        let mut last_errors: Vec<f64> = Vec::new();
        for i in 0..n_samples {
            let ref_sample = (i as f64 * 0.07).sin();
            let mic_sample = 0.8 * ref_sample; // pure echo
            let err = f.update(ref_sample, mic_sample);
            if i >= n_samples - 100 {
                last_errors.push(err.abs());
            }
        }
        let avg_err: f64 = last_errors.iter().sum::<f64>() / last_errors.len() as f64;
        assert!(
            avg_err < 0.05,
            "NLMS should cancel pure echo: avg_err={avg_err:.4}"
        );
    }

    #[test]
    fn nlms_process_block_length() {
        let mut f = NlmsFilter::new(NlmsConfig::default());
        let ref_sig: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let mic_sig: Vec<f64> = ref_sig.clone();
        let out = f.process_block(&ref_sig, &mic_sig);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn rls_pure_echo_converges_faster_than_nlms() {
        // RLS should converge to low error faster than NLMS.
        let rls_cfg = RlsConfig {
            filter_length: 16,
            forgetting_factor: 0.995,
            delta: 0.01,
        };
        let mut rls = RlsFilter::new(rls_cfg);

        let n_samples = 500;
        let mut rls_last: Vec<f64> = Vec::new();
        for i in 0..n_samples {
            let r = (i as f64 * 0.07).sin();
            let m = 0.7 * r;
            let e = rls.update(r, m);
            if i >= n_samples - 50 {
                rls_last.push(e.abs());
            }
        }
        let rls_err: f64 = rls_last.iter().sum::<f64>() / rls_last.len() as f64;
        assert!(
            rls_err < 0.1,
            "RLS should cancel pure echo: avg_err={rls_err:.4}"
        );
    }

    #[test]
    fn rls_process_block_length() {
        let mut f = RlsFilter::new(RlsConfig {
            filter_length: 8,
            ..Default::default()
        });
        let r: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
        let m = r.clone();
        let out = f.process_block(&r, &m);
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn nlms_reset_zeroes_weights() {
        let mut f = NlmsFilter::new(NlmsConfig {
            filter_length: 16,
            ..Default::default()
        });
        for i in 0..50 {
            let r = (i as f64 * 0.1).sin();
            f.update(r, r * 0.9);
        }
        f.reset();
        assert!(f.weights.iter().all(|&w| w == 0.0));
        assert!(f.buffer.iter().all(|&v| v == 0.0));
    }
}
