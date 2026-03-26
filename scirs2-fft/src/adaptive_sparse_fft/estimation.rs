//! Sparsity estimation methods for the Adaptive Sparse FFT.
//!
//! Provides parameter-free estimation of signal sparsity based on energy
//! distribution in the frequency domain using the Elbow / knee-point method.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;

/// Estimate the sparsity of a signal (number of dominant frequency components).
///
/// The estimation works by:
/// 1. Computing the power spectrum `|FFT(x)|^2`.
/// 2. Sorting the power values in descending order.
/// 3. Computing the cumulative energy as a fraction of total energy.
/// 4. Finding the "elbow" / knee-point where additional components contribute
///    negligibly — specifically, the index where the second difference of the
///    sorted power sequence is maximised (steepest change in slope).
///
/// # Arguments
///
/// * `signal` - Real-valued input signal.
///
/// # Returns
///
/// Estimated sparsity `k >= 1`.
///
/// # Errors
///
/// Returns an error if the FFT fails or the signal is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::adaptive_sparse_fft::estimation::estimate_sparsity;
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 10.0 * i as f64 / n as f64).sin())
///     .collect();
/// let k = estimate_sparsity(&signal).expect("estimation should succeed");
/// assert!(k >= 1 && k <= 4);
/// ```
pub fn estimate_sparsity(signal: &[f64]) -> FFTResult<usize> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("Signal must not be empty".to_string()));
    }

    let spectrum = fft(signal, None)?;
    let n = spectrum.len();

    // Compute power spectrum
    let mut power: Vec<f64> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();

    let total_energy: f64 = power.iter().sum();
    if total_energy < f64::EPSILON {
        return Ok(1);
    }

    // Sort descending
    power.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Normalise by total energy
    let normalised: Vec<f64> = power.iter().map(|&p| p / total_energy).collect();

    // Find knee point using second differences of normalised sorted power.
    // The second difference is large where the sorted curve changes slope most.
    let limit = n.min(256); // Consider at most 256 bins for efficiency
    if limit < 3 {
        return Ok(1);
    }

    let mut max_second_diff = 0.0_f64;
    let mut knee_idx = 1_usize;

    for i in 1..(limit - 1) {
        let second_diff = (normalised[i - 1] - 2.0 * normalised[i] + normalised[i + 1]).abs();
        if second_diff > max_second_diff {
            max_second_diff = second_diff;
            knee_idx = i;
        }
    }

    // Knee index + 1 gives the sparsity (minimum 1)
    Ok((knee_idx + 1).max(1))
}

/// Windowed sparsity estimator for long non-stationary signals.
///
/// Splits the signal into overlapping windows and estimates the sparsity
/// within each window, returning the median estimate across all windows.
pub struct SparsityEstimator {
    /// Window length in samples.
    pub window_size: usize,
    /// Hop size between consecutive windows (default: `window_size / 2`).
    pub hop_size: usize,
}

impl SparsityEstimator {
    /// Create a new `SparsityEstimator` with the given window size.
    ///
    /// The hop size defaults to half the window size (50% overlap).
    pub fn new(window_size: usize) -> Self {
        let hop_size = (window_size / 2).max(1);
        Self {
            window_size,
            hop_size,
        }
    }

    /// Create a `SparsityEstimator` with explicit hop size.
    pub fn with_hop(window_size: usize, hop_size: usize) -> Self {
        Self {
            window_size,
            hop_size: hop_size.max(1),
        }
    }

    /// Estimate sparsity across all windows of `signal`.
    ///
    /// Returns the median sparsity estimate. If `signal` is shorter than
    /// `window_size`, the whole signal is analysed as one window.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_fft::adaptive_sparse_fft::estimation::SparsityEstimator;
    /// use std::f64::consts::PI;
    ///
    /// let n = 1024;
    /// let signal: Vec<f64> = (0..n)
    ///     .map(|i| {
    ///         (2.0 * PI * 5.0 * i as f64 / n as f64).sin()
    ///         + 0.5 * (2.0 * PI * 13.0 * i as f64 / n as f64).sin()
    ///     })
    ///     .collect();
    ///
    /// let estimator = SparsityEstimator::new(256);
    /// let k = estimator.estimate(&signal).expect("estimate should succeed");
    /// assert!(k >= 1);
    /// ```
    pub fn estimate(&self, signal: &[f64]) -> FFTResult<usize> {
        if signal.is_empty() {
            return Err(FFTError::ValueError("Signal must not be empty".to_string()));
        }

        let n = signal.len();
        if n <= self.window_size {
            return estimate_sparsity(signal);
        }

        let mut estimates = Vec::new();
        let mut start = 0;
        while start + self.window_size <= n {
            let window = &signal[start..start + self.window_size];
            if let Ok(k) = estimate_sparsity(window) {
                estimates.push(k)
            }
            start += self.hop_size;
        }

        if estimates.is_empty() {
            return Ok(1);
        }

        // Return median
        estimates.sort_unstable();
        let mid = estimates.len() / 2;
        Ok(estimates[mid])
    }

    /// Estimate sparsity and return all per-window estimates for analysis.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty.
    pub fn estimate_all_windows(&self, signal: &[f64]) -> FFTResult<Vec<usize>> {
        if signal.is_empty() {
            return Err(FFTError::ValueError("Signal must not be empty".to_string()));
        }

        let n = signal.len();
        if n <= self.window_size {
            return Ok(vec![estimate_sparsity(signal)?]);
        }

        let mut estimates = Vec::new();
        let mut start = 0;
        while start + self.window_size <= n {
            let window = &signal[start..start + self.window_size];
            if let Ok(k) = estimate_sparsity(window) {
                estimates.push(k);
            }
            start += self.hop_size;
        }

        if estimates.is_empty() {
            estimates.push(1);
        }
        Ok(estimates)
    }
}

/// Estimate the noise floor of the power spectrum.
///
/// Uses the median of the lower 50% of power values as a robust noise estimate.
/// Returns the noise floor as an absolute power value.
pub fn estimate_noise_floor(signal: &[f64]) -> FFTResult<f64> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("Signal must not be empty".to_string()));
    }

    let spectrum = fft(signal, None)?;
    let mut power: Vec<f64> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();

    if power.is_empty() {
        return Ok(0.0);
    }

    power.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Median of lower half
    let half = power.len() / 2;
    let noise = if half == 0 { power[0] } else { power[half / 2] };

    Ok(noise)
}
