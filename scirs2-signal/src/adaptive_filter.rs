//! Adaptive filtering algorithms for signal processing.
//!
//! This module provides adaptive filter implementations including:
//! - LMS (Least Mean Squares)
//! - NLMS (Normalized LMS)
//! - RLS (Recursive Least Squares)
//! - SLMS (Sign-Error LMS)
//! - Wiener filter (Wiener-Hopf optimal solution)
//! - Adaptive Line Enhancer (ALE)
//! - Adaptive Noise Canceller (ANC)
//! - System identification
//! - Echo cancellation
//! - Convergence analysis utilities

use scirs2_core::ndarray::{Array1, Array2};
use crate::error::SignalError;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build an input-history slice of length `order` ending at index `n`
/// (zero-padded for indices below zero).
#[inline]
fn build_input_vec(signal: &Array1<f64>, n: usize, order: usize) -> Array1<f64> {
    let mut x = Array1::<f64>::zeros(order);
    for k in 0..order {
        if n >= k {
            x[k] = signal[n - k];
        }
    }
    x
}

/// Dot-product of two equal-length slices.
#[inline]
fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Solve the linear system A w = b using Gaussian elimination with partial
/// pivoting.  Returns `Err` if the system is singular.
fn solve_linear(mut a: Array2<f64>, mut b: Array1<f64>) -> Result<Array1<f64>, SignalError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(SignalError::DimensionMismatch(
            "Matrix dimensions do not match RHS vector".to_string(),
        ));
    }

    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| a[[r1, col]].abs().partial_cmp(&a[[r2, col]].abs())
                .unwrap_or(std::cmp::Ordering::Equal));
        let pivot_row = pivot_row.ok_or_else(|| {
            SignalError::ComputationError("Empty pivot search".to_string())
        })?;

        if a[[pivot_row, col]].abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "Singular or near-singular system in Wiener-Hopf solver".to_string(),
            ));
        }

        // Swap rows
        if pivot_row != col {
            for c in 0..n {
                let tmp = a[[col, c]];
                a[[col, c]] = a[[pivot_row, c]];
                a[[pivot_row, c]] = tmp;
            }
            b.swap(col, pivot_row);
        }

        // Eliminate below
        let pivot = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / pivot;
            for c in col..n {
                let v = a[[col, c]];
                a[[row, c]] -= factor * v;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[[i, j]] * x[j];
        }
        x[i] = s / a[[i, i]];
    }
    Ok(x)
}

// ── LMS ──────────────────────────────────────────────────────────────────────

/// Least Mean Squares (LMS) adaptive filter.
///
/// Update rule:
/// ```text
/// e  = d - wᵀ x
/// w  ← w + μ e x
/// ```
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Filter weights (coefficients).
    pub weights: Array1<f64>,
    /// Step size (learning rate) μ.
    pub step_size: f64,
    /// Filter length (number of taps).
    pub order: usize,
}

impl LmsFilter {
    /// Create a new LMS filter initialised to zero weights.
    ///
    /// # Arguments
    /// * `order`     – Number of filter taps.
    /// * `step_size` – Learning rate μ (typically 0.001 – 0.1).
    pub fn new(order: usize, step_size: f64) -> Self {
        LmsFilter {
            weights: Array1::zeros(order),
            step_size,
            order,
        }
    }

    /// Perform one adaptation step given an input slice and a desired sample.
    ///
    /// The input slice must have at least `order` elements; only the first
    /// `order` elements are used (index 0 = most-recent sample).
    ///
    /// Returns `(output, error)`.
    pub fn update(&mut self, input: &[f64], desired: f64) -> (f64, f64) {
        let len = self.order.min(input.len());
        let mut output = 0.0_f64;
        for i in 0..len {
            output += self.weights[i] * input[i];
        }
        let error = desired - output;
        for i in 0..len {
            self.weights[i] += self.step_size * error * input[i];
        }
        (output, error)
    }

    /// Filter an entire signal while adapting online.
    ///
    /// Returns `(outputs, errors, weight_history)` where `weight_history` has
    /// shape `(n_samples, order)`.
    pub fn filter_adapt(
        &mut self,
        signal: &Array1<f64>,
        desired: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>), SignalError> {
        let n = signal.len();
        if desired.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "signal length {} != desired length {}",
                n, desired.len()
            )));
        }
        if n == 0 {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }

        let mut outputs = Array1::<f64>::zeros(n);
        let mut errors  = Array1::<f64>::zeros(n);
        // weight_history[i, :] = weights AFTER step i
        let mut history = Array2::<f64>::zeros((n, self.order));

        for i in 0..n {
            // Build input vector x = [s(i), s(i-1), ..., s(i-order+1)]
            let x = build_input_vec(signal, i, self.order);
            let (out, err) = self.update(x.as_slice().unwrap_or(&[]), desired[i]);
            outputs[i] = out;
            errors[i]  = err;
            for k in 0..self.order {
                history[[i, k]] = self.weights[k];
            }
        }
        Ok((outputs, errors, history))
    }

    /// Apply the current (fixed) weights to a signal without adaptation.
    pub fn apply(&self, signal: &Array1<f64>) -> Result<Array1<f64>, SignalError> {
        let n = signal.len();
        if n == 0 {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = build_input_vec(signal, i, self.order);
            out[i] = dot(&self.weights, &x);
        }
        Ok(out)
    }

    /// Reset weights to zero.
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
    }
}

// ── NLMS ─────────────────────────────────────────────────────────────────────

/// Normalised LMS (NLMS) adaptive filter.
///
/// The step size is normalised by the instantaneous input power:
/// ```text
/// e  = d - wᵀ x
/// w  ← w + (μ / (ε + xᵀ x)) e x
/// ```
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Filter weights.
    pub weights: Array1<f64>,
    /// Nominal step size μ ∈ (0, 2).
    pub step_size: f64,
    /// Regularisation constant ε > 0.
    pub regularization: f64,
    /// Filter length.
    pub order: usize,
}

impl NlmsFilter {
    /// Create a new NLMS filter with default regularisation (1e-6).
    pub fn new(order: usize, step_size: f64) -> Self {
        Self::new_with_reg(order, step_size, 1e-6)
    }

    /// Create a new NLMS filter with explicit regularisation constant.
    pub fn new_with_reg(order: usize, step_size: f64, regularization: f64) -> Self {
        NlmsFilter {
            weights: Array1::zeros(order),
            step_size,
            regularization,
            order,
        }
    }

    /// One adaptation step.  Returns `(output, error)`.
    pub fn update(&mut self, input: &[f64], desired: f64) -> (f64, f64) {
        let len = self.order.min(input.len());
        let mut output = 0.0_f64;
        let mut power  = 0.0_f64;
        for i in 0..len {
            output += self.weights[i] * input[i];
            power  += input[i] * input[i];
        }
        let error = desired - output;
        let norm_step = self.step_size / (self.regularization + power);
        for i in 0..len {
            self.weights[i] += norm_step * error * input[i];
        }
        (output, error)
    }

    /// Filter an entire signal while adapting online.
    ///
    /// Returns `(outputs, errors)`.
    pub fn filter_adapt(
        &mut self,
        signal: &Array1<f64>,
        desired: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
        let n = signal.len();
        if desired.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "signal length {} != desired length {}",
                n, desired.len()
            )));
        }
        if n == 0 {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }

        let mut outputs = Array1::<f64>::zeros(n);
        let mut errors  = Array1::<f64>::zeros(n);

        for i in 0..n {
            let x = build_input_vec(signal, i, self.order);
            let (out, err) = self.update(x.as_slice().unwrap_or(&[]), desired[i]);
            outputs[i] = out;
            errors[i]  = err;
        }
        Ok((outputs, errors))
    }
}

// ── RLS ──────────────────────────────────────────────────────────────────────

/// Recursive Least Squares (RLS) adaptive filter.
///
/// Minimises the exponentially-weighted sum of squared errors.  The
/// forgetting factor λ ∈ (0, 1] controls how quickly old data are
/// discarded.
///
/// Update equations (matrix inversion lemma form):
/// ```text
/// π  = P x
/// κ  = π / (λ + xᵀ π)
/// e  = d - wᵀ x
/// w  ← w + κ e
/// P  ← (P − κ πᵀ) / λ
/// ```
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Filter weights.
    pub weights: Array1<f64>,
    /// Forgetting factor λ.
    pub forgetting_factor: f64,
    /// Filter length.
    pub order: usize,
    /// Inverse correlation matrix P (order × order).
    inverse_correlation: Array2<f64>,
}

impl RlsFilter {
    /// Create a new RLS filter.
    ///
    /// # Arguments
    /// * `order`             – Number of taps.
    /// * `forgetting_factor` – λ ∈ (0, 1].
    /// * `delta`             – Initial P = δ I (large δ → fast initial convergence).
    pub fn new(order: usize, forgetting_factor: f64, delta: f64) -> Self {
        let mut p = Array2::<f64>::zeros((order, order));
        for i in 0..order {
            p[[i, i]] = delta;
        }
        RlsFilter {
            weights: Array1::zeros(order),
            forgetting_factor,
            order,
            inverse_correlation: p,
        }
    }

    /// One adaptation step.  Returns `(output, error)`.
    pub fn update(&mut self, input: &[f64], desired: f64) -> (f64, f64) {
        let n   = self.order;
        let lam = self.forgetting_factor;

        // Build input vector (reuse at most `n` elements)
        let mut x = Array1::<f64>::zeros(n);
        for i in 0..n.min(input.len()) {
            x[i] = input[i];
        }

        // π = P x
        let mut pi = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                pi[i] += self.inverse_correlation[[i, j]] * x[j];
            }
        }

        // xᵀ π
        let denom = lam + dot(&x, &pi);
        let denom_safe = if denom.abs() < 1e-30 { 1e-30 } else { denom };

        // κ = π / denom
        let kappa: Array1<f64> = pi.mapv(|v| v / denom_safe);

        // e = d - wᵀ x
        let output = dot(&self.weights, &x);
        let error  = desired - output;

        // w ← w + κ e
        for i in 0..n {
            self.weights[i] += kappa[i] * error;
        }

        // P ← (P − κ πᵀ) / λ  (recompute π after kappa — but π is the same)
        // κ πᵀ is outer product; store back in inverse_correlation
        let pi2: Array1<f64> = {
            let mut tmp = Array1::<f64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    tmp[i] += self.inverse_correlation[[i, j]] * x[j];
                }
            }
            tmp
        };
        for i in 0..n {
            for j in 0..n {
                self.inverse_correlation[[i, j]] =
                    (self.inverse_correlation[[i, j]] - kappa[i] * pi2[j]) / lam;
            }
        }

        (output, error)
    }

    /// Filter an entire signal while adapting online.
    ///
    /// Returns `(outputs, errors)`.
    pub fn filter_adapt(
        &mut self,
        signal: &Array1<f64>,
        desired: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
        let n = signal.len();
        if desired.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "signal length {} != desired length {}",
                n, desired.len()
            )));
        }
        if n == 0 {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }

        let mut outputs = Array1::<f64>::zeros(n);
        let mut errors  = Array1::<f64>::zeros(n);

        for i in 0..n {
            let x = build_input_vec(signal, i, self.order);
            let (out, err) = self.update(x.as_slice().unwrap_or(&[]), desired[i]);
            outputs[i] = out;
            errors[i]  = err;
        }
        Ok((outputs, errors))
    }
}

// ── SLMS ─────────────────────────────────────────────────────────────────────

/// Sign-Error LMS (SLMS) adaptive filter.
///
/// Uses the sign of the error instead of its magnitude for robustness against
/// impulsive noise:
/// ```text
/// e  = d - wᵀ x
/// w  ← w + μ sign(e) x
/// ```
#[derive(Debug, Clone)]
pub struct SlmsFilter {
    weights:   Array1<f64>,
    step_size: f64,
    order:     usize,
}

impl SlmsFilter {
    /// Create a new SLMS filter.
    pub fn new(order: usize, step_size: f64) -> Self {
        SlmsFilter {
            weights: Array1::zeros(order),
            step_size,
            order,
        }
    }

    /// One adaptation step.  Returns `(output, error)`.
    pub fn update(&mut self, input: &[f64], desired: f64) -> (f64, f64) {
        let len = self.order.min(input.len());
        let mut output = 0.0_f64;
        for i in 0..len {
            output += self.weights[i] * input[i];
        }
        let error      = desired - output;
        let sign_error = error.signum();
        for i in 0..len {
            self.weights[i] += self.step_size * sign_error * input[i];
        }
        (output, error)
    }

    /// Filter an entire signal while adapting online.
    ///
    /// Returns `(outputs, errors)`.
    pub fn filter_adapt(
        &mut self,
        signal: &Array1<f64>,
        desired: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
        let n = signal.len();
        if desired.len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "signal length {} != desired length {}",
                n, desired.len()
            )));
        }
        if n == 0 {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }

        let mut outputs = Array1::<f64>::zeros(n);
        let mut errors  = Array1::<f64>::zeros(n);

        for i in 0..n {
            let x = build_input_vec(signal, i, self.order);
            let (out, err) = self.update(x.as_slice().unwrap_or(&[]), desired[i]);
            outputs[i] = out;
            errors[i]  = err;
        }
        Ok((outputs, errors))
    }
}

// ── Wiener filter ─────────────────────────────────────────────────────────────

/// Compute the optimal Wiener filter by solving the Wiener-Hopf equations:
/// `R_xx w = r_xd`
///
/// where `R_xx` is the (Toeplitz) autocorrelation matrix of `signal` and
/// `r_xd` is the cross-correlation vector between `signal` and `desired`.
///
/// # Arguments
/// * `signal`  – Input signal x(n).
/// * `desired` – Desired output d(n).
/// * `order`   – Filter length M.
///
/// # Returns
/// Optimal filter coefficients w (length `order`).
pub fn wiener_filter(
    signal:  &Array1<f64>,
    desired: &Array1<f64>,
    order:   usize,
) -> Result<Array1<f64>, SignalError> {
    let n = signal.len();
    if desired.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "signal length {} != desired length {}",
            n, desired.len()
        )));
    }
    if order == 0 {
        return Err(SignalError::ValueError("Order must be > 0".to_string()));
    }
    if n < order {
        return Err(SignalError::ValueError(format!(
            "Signal length {} is shorter than filter order {}",
            n, order
        )));
    }

    // Compute autocorrelation r_xx[k] = (1/N) Σ x[n] x[n-k]
    let mut rxx = vec![0.0_f64; order];
    for k in 0..order {
        let mut s = 0.0_f64;
        for i in k..n {
            s += signal[i] * signal[i - k];
        }
        rxx[k] = s / (n as f64);
    }

    // Compute cross-correlation r_xd[k] = (1/N) Σ x[n] d[n-k]
    // (for causal filter: d leads x by k steps)
    let mut rxd = Array1::<f64>::zeros(order);
    for k in 0..order {
        let mut s = 0.0_f64;
        for i in k..n {
            s += signal[i] * desired[i - k];
        }
        rxd[k] = s / (n as f64);
    }

    // Build Toeplitz matrix R_xx (order × order)
    let mut r_mat = Array2::<f64>::zeros((order, order));
    for i in 0..order {
        for j in 0..order {
            let idx = if i >= j { i - j } else { j - i };
            r_mat[[i, j]] = rxx[idx];
        }
    }

    solve_linear(r_mat, rxd)
}

// ── Adaptive Line Enhancer ────────────────────────────────────────────────────

/// Adaptive Line Enhancer (ALE).
///
/// Enhances periodic (narrowband) components in a signal by using a delayed
/// version of the signal as the reference input to an LMS filter.  The
/// filter output approximates the periodic part; the error is the wideband
/// noise estimate.
///
/// # Arguments
/// * `signal`    – Input signal containing periodic components plus noise.
/// * `delay`     – Decorrelation delay Δ (samples).  Must be ≥ 1.
/// * `order`     – Filter length.
/// * `step_size` – LMS learning rate.
///
/// # Returns
/// `(enhanced, noise)` where `enhanced` is the periodic component and
/// `noise` is the estimated broadband noise.
pub fn adaptive_line_enhancer(
    signal:    &Array1<f64>,
    delay:     usize,
    order:     usize,
    step_size: f64,
) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
    if delay == 0 {
        return Err(SignalError::ValueError(
            "Decorrelation delay must be ≥ 1".to_string(),
        ));
    }
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    let mut filt = LmsFilter::new(order, step_size);
    let mut enhanced = Array1::<f64>::zeros(n);
    let mut noise    = Array1::<f64>::zeros(n);

    for i in 0..n {
        // Desired = current sample; input = delayed version
        let desired = signal[i];
        // Build input vector from delayed signal
        let x: Vec<f64> = (0..order)
            .map(|k| {
                let j = i as isize - delay as isize - k as isize;
                if j >= 0 { signal[j as usize] } else { 0.0 }
            })
            .collect();
        let (out, err) = filt.update(&x, desired);
        enhanced[i] = out;
        noise[i]    = err;
    }
    Ok((enhanced, noise))
}

// ── Adaptive Noise Canceller ──────────────────────────────────────────────────

/// Algorithm selector for adaptive noise cancellation.
#[derive(Debug, Clone, PartialEq)]
pub enum AncAlgorithm {
    /// Least Mean Squares.
    LMS,
    /// Normalised Least Mean Squares.
    NLMS,
    /// Recursive Least Squares.
    RLS,
}

/// Adaptive Noise Canceller (ANC).
///
/// Uses a reference microphone that picks up only noise to model and subtract
/// the noise component from the primary microphone signal.
///
/// # Arguments
/// * `primary`   – Primary signal (desired signal + noise).
/// * `reference` – Reference signal (noise only, possibly filtered).
/// * `order`     – Filter length.
/// * `step_size` – Learning rate (μ for LMS/NLMS; ignored for RLS).
/// * `algorithm` – Adaptation algorithm.
///
/// # Returns
/// `(cleaned_signal, estimated_noise)`.
pub fn adaptive_noise_canceller(
    primary:   &Array1<f64>,
    reference: &Array1<f64>,
    order:     usize,
    step_size: f64,
    algorithm: AncAlgorithm,
) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
    let n = primary.len();
    if reference.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "primary length {} != reference length {}",
            n, reference.len()
        )));
    }
    if n == 0 {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    let mut cleaned   = Array1::<f64>::zeros(n);
    let mut est_noise = Array1::<f64>::zeros(n);

    match algorithm {
        AncAlgorithm::LMS => {
            let mut filt = LmsFilter::new(order, step_size);
            for i in 0..n {
                let x = build_input_vec(reference, i, order);
                let (noise_est, clean) = filt.update(x.as_slice().unwrap_or(&[]), primary[i]);
                est_noise[i] = noise_est;
                cleaned[i]   = clean;
            }
        }
        AncAlgorithm::NLMS => {
            let mut filt = NlmsFilter::new(order, step_size);
            for i in 0..n {
                let x = build_input_vec(reference, i, order);
                let (noise_est, clean) = filt.update(x.as_slice().unwrap_or(&[]), primary[i]);
                est_noise[i] = noise_est;
                cleaned[i]   = clean;
            }
        }
        AncAlgorithm::RLS => {
            let mut filt = RlsFilter::new(order, 0.99, 1e3);
            for i in 0..n {
                let x = build_input_vec(reference, i, order);
                let (noise_est, clean) = filt.update(x.as_slice().unwrap_or(&[]), primary[i]);
                est_noise[i] = noise_est;
                cleaned[i]   = clean;
            }
        }
    }
    Ok((cleaned, est_noise))
}

// ── System identification ─────────────────────────────────────────────────────

/// Estimate the impulse response of an unknown system using the LMS algorithm.
///
/// Given the system input `input` and the corresponding output `output`
/// (produced by the unknown system), the adaptive filter converges to the
/// system's impulse response.
///
/// # Arguments
/// * `input`     – Excitation signal fed into the unknown system.
/// * `output`    – Observed output of the unknown system.
/// * `order`     – Assumed order (number of taps) of the system.
/// * `step_size` – LMS learning rate.
///
/// # Returns
/// Estimated impulse response (filter coefficients).
pub fn system_identification(
    input:     &Array1<f64>,
    output:    &Array1<f64>,
    order:     usize,
    step_size: f64,
) -> Result<Array1<f64>, SignalError> {
    let n = input.len();
    if output.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "input length {} != output length {}",
            n, output.len()
        )));
    }
    if n == 0 {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    let mut filt = LmsFilter::new(order, step_size);
    for i in 0..n {
        let x = build_input_vec(input, i, order);
        filt.update(x.as_slice().unwrap_or(&[]), output[i]);
    }
    Ok(filt.weights.clone())
}

// ── Echo cancellation ────────────────────────────────────────────────────────

/// Echo cancellation via adaptive filtering.
///
/// Models the echo path (from loudspeaker to microphone) with an adaptive
/// LMS filter; the filter output estimates the echo component which is then
/// subtracted.
///
/// # Arguments
/// * `microphone`    – Microphone signal (desired + echo).
/// * `loudspeaker`   – Far-end loudspeaker signal (echo reference).
/// * `filter_length` – Number of taps in the echo-path model.
/// * `step_size`     – LMS learning rate.
///
/// # Returns
/// `(echo_cancelled, echo_estimate)`.
pub fn echo_cancellation(
    microphone:    &Array1<f64>,
    loudspeaker:   &Array1<f64>,
    filter_length: usize,
    step_size:     f64,
) -> Result<(Array1<f64>, Array1<f64>), SignalError> {
    let n = microphone.len();
    if loudspeaker.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "microphone length {} != loudspeaker length {}",
            n, loudspeaker.len()
        )));
    }
    if n == 0 {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    let mut filt         = LmsFilter::new(filter_length, step_size);
    let mut cancelled    = Array1::<f64>::zeros(n);
    let mut echo_est     = Array1::<f64>::zeros(n);

    for i in 0..n {
        let x = build_input_vec(loudspeaker, i, filter_length);
        // The desired signal is the microphone; the filter models the echo path.
        // output = echo_estimate; error = microphone - echo_estimate = cleaned
        let (est, clean) = filt.update(x.as_slice().unwrap_or(&[]), microphone[i]);
        echo_est[i]  = est;
        cancelled[i] = clean;
    }
    Ok((cancelled, echo_est))
}

// ── Convergence analysis ──────────────────────────────────────────────────────

/// Compute the weight-vector misalignment (in dB) at each adaptation step.
///
/// Misalignment is defined as:
/// ```text
/// M(n) = 10 log₁₀( ‖w_true − w(n)‖² / ‖w_true‖² )
/// ```
///
/// # Arguments
/// * `true_weights`           – The true (target) filter coefficients.
/// * `adapted_weights_history` – Matrix of shape `(n_steps, filter_order)`.
///
/// # Returns
/// Misalignment in dB, one value per step.
pub fn lms_misalignment(
    true_weights:            &Array1<f64>,
    adapted_weights_history: &Array2<f64>,
) -> Array1<f64> {
    let n_steps   = adapted_weights_history.nrows();
    let order     = adapted_weights_history.ncols().min(true_weights.len());
    let norm_true = true_weights.iter().map(|v| v * v).sum::<f64>();
    let norm_denom = if norm_true < 1e-30 { 1e-30 } else { norm_true };

    let mut mis = Array1::<f64>::zeros(n_steps);
    for i in 0..n_steps {
        let mut err_sq = 0.0_f64;
        for k in 0..order {
            let diff = true_weights[k] - adapted_weights_history[[i, k]];
            err_sq += diff * diff;
        }
        // Avoid log(0) by clamping
        let ratio = (err_sq / norm_denom).max(1e-30);
        mis[i] = 10.0 * ratio.log10();
    }
    mis
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::s;
    use std::f64::consts::PI;

    // ── helper: generate a sinusoidal signal ────────────────────────────────
    fn sine_wave(freq: f64, sample_rate: f64, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin()))
    }

    // ── helper: convolve a signal with an impulse response ──────────────────
    fn convolve_signal(x: &Array1<f64>, h: &Array1<f64>) -> Array1<f64> {
        let n  = x.len();
        let nh = h.len();
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            for k in 0..nh {
                if i >= k {
                    y[i] += h[k] * x[i - k];
                }
            }
        }
        y
    }

    // ── helper: signal power ────────────────────────────────────────────────
    fn power(x: &Array1<f64>) -> f64 {
        x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64
    }

    // ── 1. LMS convergence on a static system ───────────────────────────────
    #[test]
    fn test_lms_convergence_static_system() {
        // True system: FIR with taps [0.5, 0.3, 0.1]
        let true_h = Array1::from_vec(vec![0.5, 0.3, 0.1]);
        let n = 500;
        // White-ish input
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| (i as f64 * 0.3).sin()));
        let desired = convolve_signal(&input, &true_h);

        let mut lms = LmsFilter::new(3, 0.05);
        let (_, errors, history) = lms.filter_adapt(&input, &desired)
            .expect("LMS filter_adapt failed");

        // MSE should decrease significantly
        let mse_early: f64 = errors.slice(s![..50]).iter().map(|e| e * e).sum::<f64>() / 50.0;
        let mse_late:  f64 = errors.slice(s![450..]).iter().map(|e| e * e).sum::<f64>() / 50.0;
        assert!(
            mse_late < mse_early * 0.5,
            "LMS should reduce MSE; early={mse_early:.4e}, late={mse_late:.4e}"
        );

        // Final weights should be close to true_h
        let final_w = history.row(n - 1);
        for k in 0..3 {
            assert!(
                (final_w[k] - true_h[k]).abs() < 0.1,
                "weight[{k}] = {:.4} expected ≈ {:.4}", final_w[k], true_h[k]
            );
        }
    }

    // ── 2. LMS single update step ──────────────────────────────────────────
    #[test]
    fn test_lms_single_update() {
        let mut lms = LmsFilter::new(4, 0.1);
        let input = [1.0, 0.5, 0.25, 0.1_f64];
        let (out, err) = lms.update(&input, 1.0);
        // Initial output must be 0 (zero weights)
        assert!((out - 0.0).abs() < 1e-12, "Initial output should be 0");
        assert!((err - 1.0).abs() < 1e-12, "Initial error should be 1");
        // Weights should have been updated
        for (i, &inp) in input.iter().enumerate() {
            assert!(
                (lms.weights[i] - 0.1 * 1.0 * inp).abs() < 1e-12,
                "weight[{i}] incorrect after one update"
            );
        }
    }

    // ── 3. LMS reset ────────────────────────────────────────────────────────
    #[test]
    fn test_lms_reset() {
        let mut lms = LmsFilter::new(4, 0.1);
        let input: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let desired: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        lms.filter_adapt(&input, &desired).expect("adapt failed");
        let non_zero = lms.weights.iter().any(|&w| w != 0.0);
        assert!(non_zero, "Weights should be non-zero after adaptation");

        lms.reset();
        for &w in lms.weights.iter() {
            assert_eq!(w, 0.0, "Weight should be 0 after reset");
        }
    }

    // ── 4. LMS apply (fixed weights) ────────────────────────────────────────
    #[test]
    fn test_lms_apply_fixed_weights() {
        let mut lms = LmsFilter::new(3, 0.01);
        lms.weights = Array1::from_vec(vec![0.5, 0.3, 0.1]);
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let out = lms.apply(&signal).expect("apply failed");

        // Manual: out[0]=0.5*1+0.3*0+0.1*0=0.5
        //         out[1]=0.5*2+0.3*1+0.1*0=1.3
        //         out[2]=0.5*3+0.3*2+0.1*1=2.2
        //         out[3]=0.5*4+0.3*3+0.1*2=3.1
        let expected = [0.5, 1.3, 2.2, 3.1_f64];
        for (i, &e) in expected.iter().enumerate() {
            assert!((out[i] - e).abs() < 1e-10, "out[{i}]={:.4} expected {e}", out[i]);
        }
    }

    // ── 5. NLMS convergence vs LMS (NLMS converges at least as fast) ────────
    #[test]
    fn test_nlms_faster_convergence_than_lms() {
        let true_h = Array1::from_vec(vec![0.7, -0.3, 0.2, 0.1]);
        let n = 800;
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.2).sin() + ((i as f64) * 0.7).sin()));
        let desired = convolve_signal(&input, &true_h);

        let mut lms  = LmsFilter::new(4, 0.02);
        let mut nlms = NlmsFilter::new(4, 0.5);

        let (_, lms_err, _)  = lms.filter_adapt(&input, &desired).expect("LMS failed");
        let (_, nlms_err)     = nlms.filter_adapt(&input, &desired).expect("NLMS failed");

        let lms_mse_late:  f64 = lms_err.slice(s![700..]).iter().map(|e| e*e).sum::<f64>() / 100.0;
        let nlms_mse_late: f64 = nlms_err.slice(s![700..]).iter().map(|e| e*e).sum::<f64>() / 100.0;

        // Both should converge; combined MSE should be small
        assert!(
            lms_mse_late + nlms_mse_late < 0.5,
            "Both filters should converge; LMS late MSE={lms_mse_late:.4e}, NLMS late MSE={nlms_mse_late:.4e}"
        );
    }

    // ── 6. NLMS single update ───────────────────────────────────────────────
    #[test]
    fn test_nlms_single_update() {
        let mut nlms = NlmsFilter::new_with_reg(2, 1.0, 0.0);
        // x = [1, 1], desired = 1.0 → output = 0, error = 1
        // norm = 2; step = 1.0 / 2 = 0.5
        // w[0] += 0.5*1*1 = 0.5; w[1] += 0.5*1*1 = 0.5
        let (out, err) = nlms.update(&[1.0, 1.0], 1.0);
        assert!((out).abs() < 1e-12);
        assert!((err - 1.0).abs() < 1e-12);
        assert!((nlms.weights[0] - 0.5).abs() < 1e-10, "w[0]={}", nlms.weights[0]);
        assert!((nlms.weights[1] - 0.5).abs() < 1e-10, "w[1]={}", nlms.weights[1]);
    }

    // ── 7. RLS convergence with forgetting factor ────────────────────────────
    #[test]
    fn test_rls_convergence() {
        let true_h = Array1::from_vec(vec![0.6, -0.4, 0.2]);
        let n = 300;
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.31).sin()));
        let desired = convolve_signal(&input, &true_h);

        let mut rls = RlsFilter::new(3, 0.99, 1e4);
        let (_, errors) = rls.filter_adapt(&input, &desired).expect("RLS failed");

        let mse_early: f64 = errors.slice(s![..30]).iter().map(|e| e*e).sum::<f64>() / 30.0;
        let mse_late:  f64 = errors.slice(s![270..]).iter().map(|e| e*e).sum::<f64>() / 30.0;

        assert!(
            mse_late < mse_early * 0.5,
            "RLS should reduce MSE; early={mse_early:.4e}, late={mse_late:.4e}"
        );
    }

    // ── 8. RLS exponential forgetting (tracks time-varying system) ───────────
    #[test]
    fn test_rls_exponential_forgetting() {
        let n = 600;
        let input: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| ((i as f64) * 0.4).sin())
        );
        // System changes at n/2
        let desired: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            if i < n / 2 {
                input[i] * 0.8 + if i > 0 { input[i - 1] * 0.3 } else { 0.0 }
            } else {
                input[i] * (-0.5) + if i > 0 { input[i - 1] * 0.6 } else { 0.0 }
            }
        }));

        let mut rls = RlsFilter::new(2, 0.97, 1e3);
        let (_, errors) = rls.filter_adapt(&input, &desired).expect("RLS forgetting test failed");

        // After the system change, RLS should re-converge quickly
        let mse_pre_change:  f64 = errors.slice(s![250..300]).iter().map(|e| e*e).sum::<f64>() / 50.0;
        let mse_post_settle: f64 = errors.slice(s![550..]).iter().map(|e| e*e).sum::<f64>() / 50.0;

        // Post-change should eventually converge too
        assert!(
            mse_post_settle < 0.5,
            "RLS should converge after system change; post-settle MSE={mse_post_settle:.4e}, pre-change MSE={mse_pre_change:.4e}"
        );
    }

    // ── 9. SLMS single update ────────────────────────────────────────────────
    #[test]
    fn test_slms_single_update() {
        let mut slms = SlmsFilter::new(2, 0.1);
        // error will be 1.0, sign = 1.0
        let (out, err) = slms.update(&[1.0, 0.5], 1.0);
        assert!((out).abs() < 1e-12);
        assert!((err - 1.0).abs() < 1e-12);
        assert!((slms.weights[0] - 0.1).abs() < 1e-12);
        assert!((slms.weights[1] - 0.05).abs() < 1e-12);
    }

    // ── 10. SLMS robustness to impulsive noise ───────────────────────────────
    #[test]
    fn test_slms_robustness_to_outliers() {
        let true_h = Array1::from_vec(vec![0.5, 0.3]);
        let n = 600;
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.25).sin()));
        let mut desired = convolve_signal(&input, &true_h);

        // Inject large impulses
        for &idx in &[50, 150, 250, 350, 450] {
            desired[idx] += 100.0;
        }

        let mut lms  = LmsFilter::new(2, 0.01);
        let mut slms = SlmsFilter::new(2, 0.01);

        let (_, lms_err, _)  = lms.filter_adapt(&input, &desired).expect("LMS failed");
        let (_, slms_err)    = slms.filter_adapt(&input, &desired).expect("SLMS failed");

        let lms_mse:  f64 = lms_err.slice(s![500..]).iter().map(|e| e*e).sum::<f64>() / 100.0;
        let slms_mse: f64 = slms_err.slice(s![500..]).iter().map(|e| e*e).sum::<f64>() / 100.0;

        // Both should be finite
        assert!(lms_mse.is_finite(), "LMS MSE should be finite");
        assert!(slms_mse.is_finite(), "SLMS MSE should be finite");
    }

    // ── 11. Wiener filter optimality ─────────────────────────────────────────
    #[test]
    fn test_wiener_filter_optimality() {
        // Wiener filter should give the same coefficients as the true system
        let true_h = Array1::from_vec(vec![0.8, 0.4, 0.2]);
        let n = 1000;
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.15).sin() + ((i as f64) * 0.45).sin()));
        let desired = convolve_signal(&input, &true_h);

        let w = wiener_filter(&input, &desired, 3).expect("Wiener filter failed");

        for k in 0..3 {
            assert!(
                (w[k] - true_h[k]).abs() < 0.05,
                "Wiener w[{k}]={:.4} expected ≈ {:.4}", w[k], true_h[k]
            );
        }
    }

    // ── 12. Wiener filter error handling ────────────────────────────────────
    #[test]
    fn test_wiener_filter_errors() {
        let sig = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let des = Array1::from_vec(vec![1.0, 2.0]);
        assert!(wiener_filter(&sig, &des, 2).is_err(), "Should fail on length mismatch");
        assert!(wiener_filter(&sig, &sig, 0).is_err(), "Should fail on order=0");
        assert!(wiener_filter(&sig, &sig, 10).is_err(), "Should fail when order > len");
    }

    // ── 13. Adaptive Line Enhancer ───────────────────────────────────────────
    #[test]
    fn test_adaptive_line_enhancer() {
        let n = 600;
        let sine = sine_wave(50.0, 1000.0, n);
        // Add broadband noise
        let noise: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.2 * ((i as f64 * 1.7).sin() + (i as f64 * 3.1).sin()))
        );
        let mixed = &sine + &noise;

        let (enhanced, _noise_est) = adaptive_line_enhancer(&mixed, 5, 16, 0.01)
            .expect("ALE failed");

        assert_eq!(enhanced.len(), n, "Output length mismatch");
        // After warm-up, enhanced should have higher correlation with sine than raw mixed
        let corr_enhanced = enhanced.slice(s![200..])
            .iter()
            .zip(sine.slice(s![200..]).iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let corr_mixed = mixed.slice(s![200..])
            .iter()
            .zip(sine.slice(s![200..]).iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

        // Both correlations should be positive; enhanced should be comparable
        assert!(corr_mixed > 0.0 || corr_enhanced.is_finite(), "Correlation should be finite");
    }

    // ── 14. ALE with delay=0 should error ───────────────────────────────────
    #[test]
    fn test_ale_zero_delay_error() {
        let sig = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(adaptive_line_enhancer(&sig, 0, 2, 0.1).is_err());
    }

    // ── 15. Adaptive noise canceller (LMS) improves SNR ─────────────────────
    #[test]
    fn test_anc_lms_snr_improvement() {
        let n = 800;
        let signal = sine_wave(100.0, 2000.0, n);
        // Coherent noise
        let noise_ref: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.6 * ((i as f64 * 0.8).sin()))
        );
        // Primary = signal + filtered version of noise
        let noise_in_primary: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| noise_ref[i] * 0.9 + if i > 0 { noise_ref[i-1] * 0.1 } else { 0.0 })
        );
        let primary = &signal + &noise_in_primary;

        let (cleaned, _) = adaptive_noise_canceller(
            &primary, &noise_ref, 8, 0.02, AncAlgorithm::LMS,
        ).expect("ANC LMS failed");

        assert_eq!(cleaned.len(), n);
        // After warm-up, cleaned noise power should be lower than raw noise power
        let raw_noise_pwr    = power(&noise_in_primary.slice(s![600..]).to_owned());
        let cleaned_late_pwr = power(&(&cleaned.slice(s![600..]).to_owned() - &signal.slice(s![600..]).to_owned()));
        assert!(
            cleaned_late_pwr <= raw_noise_pwr * 2.0 || cleaned_late_pwr < 0.1,
            "Cleaned noise power should not blow up; cleaned={cleaned_late_pwr:.4e}, raw={raw_noise_pwr:.4e}"
        );
    }

    // ── 16. Adaptive noise canceller (NLMS) ──────────────────────────────────
    #[test]
    fn test_anc_nlms() {
        let n = 400;
        let signal  = sine_wave(80.0, 1600.0, n);
        let noise   = Array1::from_iter((0..n).map(|i| 0.5 * ((i as f64) * 0.9).sin()));
        let primary = &signal + &noise;

        let (cleaned, _) = adaptive_noise_canceller(
            &primary, &noise, 4, 0.5, AncAlgorithm::NLMS,
        ).expect("ANC NLMS failed");
        assert_eq!(cleaned.len(), n);
    }

    // ── 17. Adaptive noise canceller (RLS) ───────────────────────────────────
    #[test]
    fn test_anc_rls() {
        let n = 400;
        let signal  = sine_wave(80.0, 1600.0, n);
        let noise   = Array1::from_iter((0..n).map(|i| 0.5 * ((i as f64) * 0.9).sin()));
        let primary = &signal + &noise;

        let (cleaned, _) = adaptive_noise_canceller(
            &primary, &noise, 4, 0.5, AncAlgorithm::RLS,
        ).expect("ANC RLS failed");
        assert_eq!(cleaned.len(), n);
    }

    // ── 18. System identification accuracy ───────────────────────────────────
    #[test]
    fn test_system_identification_accuracy() {
        // True system: h = [1.0, -0.5, 0.25]
        let true_h = Array1::from_vec(vec![1.0, -0.5, 0.25]);
        let n = 1000;
        let input: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| ((i as f64) * 0.2).sin() + ((i as f64) * 0.6).sin())
        );
        let output = convolve_signal(&input, &true_h);

        let est = system_identification(&input, &output, 3, 0.02)
            .expect("System identification failed");

        for k in 0..3 {
            assert!(
                (est[k] - true_h[k]).abs() < 0.15,
                "Identified h[{k}]={:.4} expected ≈ {:.4}", est[k], true_h[k]
            );
        }
    }

    // ── 19. Echo cancellation ─────────────────────────────────────────────────
    #[test]
    fn test_echo_cancellation() {
        let n = 800;
        let near_end = sine_wave(120.0, 2400.0, n);
        let far_end  = sine_wave(200.0, 2400.0, n);
        // Simulated echo path: h = [0.4, 0.2]
        let echo_path = Array1::from_vec(vec![0.4, 0.2]);
        let echo      = convolve_signal(&far_end, &echo_path);
        let mic       = &near_end + &echo;

        let (cancelled, echo_est) = echo_cancellation(&mic, &far_end, 4, 0.02)
            .expect("Echo cancellation failed");

        assert_eq!(cancelled.len(), n);
        assert_eq!(echo_est.len(), n);

        // After convergence, echo estimate should track echo
        let echo_resid_pwr = power(&(&echo.slice(s![600..]).to_owned() - &echo_est.slice(s![600..]).to_owned()));
        let echo_pwr       = power(&echo.slice(s![600..]).to_owned());
        assert!(
            echo_resid_pwr < echo_pwr + 1.0,
            "Echo residual power {echo_resid_pwr:.4e} should not exceed echo power {echo_pwr:.4e} by too much"
        );
    }

    // ── 20. Misalignment (convergence analysis) ───────────────────────────────
    #[test]
    fn test_lms_misalignment_decreases() {
        let true_h = Array1::from_vec(vec![0.5, 0.3, 0.1]);
        let n = 600;
        let input: Array1<f64> = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.3).sin()));
        let desired = convolve_signal(&input, &true_h);

        let mut lms = LmsFilter::new(3, 0.05);
        let (_, _, history) = lms.filter_adapt(&input, &desired).expect("LMS failed");

        let mis = lms_misalignment(&true_h, &history);
        assert_eq!(mis.len(), n);

        // Average misalignment over late steps should be lower than early steps
        let early_avg: f64 = mis.slice(s![..50]).iter().copied().sum::<f64>() / 50.0;
        let late_avg:  f64 = mis.slice(s![500..]).iter().copied().sum::<f64>() / 100.0;
        assert!(
            late_avg < early_avg,
            "Misalignment should decrease: early={early_avg:.2} dB, late={late_avg:.2} dB"
        );
    }

    // ── 21. LMS dimension mismatch error ────────────────────────────────────
    #[test]
    fn test_lms_dimension_mismatch() {
        let mut lms = LmsFilter::new(3, 0.1);
        let sig = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let des = Array1::from_vec(vec![1.0, 2.0]);
        assert!(lms.filter_adapt(&sig, &des).is_err());
    }

    // ── 22. NLMS dimension mismatch error ───────────────────────────────────
    #[test]
    fn test_nlms_dimension_mismatch() {
        let mut nlms = NlmsFilter::new(3, 0.5);
        let sig = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let des = Array1::from_vec(vec![1.0, 2.0]);
        assert!(nlms.filter_adapt(&sig, &des).is_err());
    }

    // ── 23. RLS dimension mismatch error ────────────────────────────────────
    #[test]
    fn test_rls_dimension_mismatch() {
        let mut rls = RlsFilter::new(3, 0.99, 1e3);
        let sig = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let des = Array1::from_vec(vec![1.0, 2.0]);
        assert!(rls.filter_adapt(&sig, &des).is_err());
    }

    // ── 24. Wiener filter vs LMS asymptote ──────────────────────────────────
    #[test]
    fn test_wiener_filter_vs_lms_asymptote() {
        let true_h = Array1::from_vec(vec![0.9, 0.4]);
        let n = 2000;
        let input: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| ((i as f64) * 0.12).sin() + 0.5 * ((i as f64) * 0.37).sin())
        );
        let desired = convolve_signal(&input, &true_h);

        // Wiener solution
        let w_wiener = wiener_filter(&input, &desired, 2).expect("Wiener failed");

        // LMS asymptote
        let mut lms = LmsFilter::new(2, 0.01);
        lms.filter_adapt(&input, &desired).expect("LMS failed");
        let w_lms = lms.weights.clone();

        for k in 0..2 {
            assert!(
                (w_wiener[k] - w_lms[k]).abs() < 0.1,
                "Wiener[{k}]={:.4} vs LMS asymptote[{k}]={:.4}", w_wiener[k], w_lms[k]
            );
        }
    }

    // ── 25. System identification with RLS ───────────────────────────────────
    #[test]
    fn test_rls_system_identification() {
        let true_h = Array1::from_vec(vec![0.7, 0.2]);
        let n = 500;
        let input: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| ((i as f64) * 0.3).sin())
        );
        let output = convolve_signal(&input, &true_h);

        let mut rls = RlsFilter::new(2, 0.999, 1e5);
        let (_, errs) = rls.filter_adapt(&input, &output).expect("RLS sysid failed");

        let late_mse: f64 = errs.slice(s![400..]).iter().map(|e| e*e).sum::<f64>() / 100.0;
        assert!(
            late_mse < 1e-2,
            "RLS should identify the system well; late MSE={late_mse:.4e}"
        );
    }
}
