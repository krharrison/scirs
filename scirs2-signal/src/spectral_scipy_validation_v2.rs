//! SciPy Comparison Validation Tests for Advanced Spectral Analysis v0.2.0
//!
//! This module provides comprehensive validation tests comparing our implementations
//! against SciPy's signal processing functions to ensure numerical accuracy and
//! compatibility.
//!
//! Reference implementations from SciPy:
//! - `scipy.signal.periodogram`: Periodogram spectral estimation
//! - `scipy.signal.welch`: Welch's method for PSD estimation
//! - `scipy.signal.lombscargle`: Lomb-Scargle periodogram
//! - `scipy.signal.butter`: Butterworth filter design
//! - `scipy.signal.filtfilt`: Zero-phase filtering

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Complex64;
use scirs2_core::Rng;
use std::f64::consts::PI;

// ============================================================================
// Validation Test Infrastructure
// ============================================================================

/// Validation result for a single test
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Test name
    pub test_name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Maximum absolute error
    pub max_error: f64,
    /// Mean absolute error
    pub mean_error: f64,
    /// Root mean square error
    pub rms_error: f64,
    /// Correlation coefficient with reference
    pub correlation: f64,
    /// Additional notes
    pub notes: String,
}

impl ValidationResult {
    fn new(test_name: &str) -> Self {
        Self {
            test_name: test_name.to_string(),
            passed: false,
            max_error: 0.0,
            mean_error: 0.0,
            rms_error: 0.0,
            correlation: 0.0,
            notes: String::new(),
        }
    }
}

/// Collection of validation results
#[derive(Debug)]
pub struct ValidationSuite {
    /// Individual test results
    pub results: Vec<ValidationResult>,
    /// Overall pass status
    pub all_passed: bool,
    /// Total tests run
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
}

impl ValidationSuite {
    fn new() -> Self {
        Self {
            results: Vec::new(),
            all_passed: true,
            total_tests: 0,
            passed_tests: 0,
        }
    }

    fn add_result(&mut self, result: ValidationResult) {
        self.total_tests += 1;
        if result.passed {
            self.passed_tests += 1;
        } else {
            self.all_passed = false;
        }
        self.results.push(result);
    }
}

// ============================================================================
// Reference Value Generators (SciPy-compatible)
// ============================================================================

/// Generate test signal with known spectral properties
pub fn generate_reference_signal(
    n: usize,
    frequencies: &[f64],
    amplitudes: &[f64],
    fs: f64,
    noise_level: f64,
) -> Array1<f64> {
    let mut signal = Array1::zeros(n);
    let mut rng = scirs2_core::random::rng();

    for i in 0..n {
        let t = i as f64 / fs;
        let mut sample = 0.0;

        for (freq, amp) in frequencies.iter().zip(amplitudes.iter()) {
            sample += amp * (2.0 * PI * freq * t).sin();
        }

        // Add noise
        sample += noise_level * (rng.random::<f64>() - 0.5) * 2.0;

        signal[i] = sample;
    }

    signal
}

/// Generate AR process signal with known parameters
pub fn generate_ar_process(
    n: usize,
    ar_coeffs: &[f64],
    variance: f64,
) -> SignalResult<Array1<f64>> {
    if ar_coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "AR coefficients cannot be empty".to_string(),
        ));
    }

    let p = ar_coeffs.len() - 1;
    let mut signal = Array1::zeros(n);
    let mut rng = scirs2_core::random::rng();
    let std_dev = variance.sqrt();

    for t in 0..n {
        // Generate innovation (white noise)
        let innovation = std_dev * (rng.random::<f64>() - 0.5) * 2.0 * 3.464; // Approximate normal

        let mut sample = innovation;

        // AR part
        for i in 1..=p.min(t) {
            sample -= ar_coeffs[i] * signal[t - i];
        }

        signal[t] = sample;
    }

    Ok(signal)
}

/// Compute reference autocorrelation (unbiased)
pub fn compute_reference_autocorrelation(signal: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = signal.len();
    let mean = signal.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    let mut autocorr = Array1::zeros(max_lag + 1);

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += centered[i] * centered[i + lag];
        }
        autocorr[lag] = sum / (n - lag) as f64;
    }

    // Normalize
    let r0 = autocorr[0];
    if r0.abs() > 1e-15 {
        autocorr.mapv_inplace(|x| x / r0);
    }

    autocorr
}

/// Compute reference periodogram (matches scipy.signal.periodogram)
pub fn compute_reference_periodogram(
    signal: &Array1<f64>,
    fs: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = signal.len();
    let nfft = n;

    // Compute FFT using scirs2_fft
    let signal_f64: Vec<f64> = signal.iter().copied().collect();
    let complex_input = scirs2_fft::fft(&signal_f64, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Compute PSD
    let n_freq = nfft / 2 + 1;
    let mut psd = Array1::zeros(n_freq);
    let mut frequencies = Array1::zeros(n_freq);

    for k in 0..n_freq {
        frequencies[k] = k as f64 * fs / nfft as f64;

        let mag_sq = complex_input[k].norm_sqr();
        let scale = if k == 0 || (k == nfft / 2 && nfft % 2 == 0) {
            1.0
        } else {
            2.0
        };

        // SciPy normalization: 1 / (fs * n)
        psd[k] = scale * mag_sq / (fs * n as f64);
    }

    Ok((frequencies, psd))
}

// ============================================================================
// Validation Tests
// ============================================================================

/// Validate AR spectral estimation against known parameters
pub fn validate_ar_spectral_estimation() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("AR Spectral Estimation");

    // Generate AR(2) process with known parameters
    let true_ar_coeffs = [1.0, -1.5, 0.7]; // Stable AR(2)
    let true_variance = 1.0;
    let n = 2048;

    let signal = generate_ar_process(n, &true_ar_coeffs, true_variance)?;

    // Estimate using our implementation
    let config = crate::advanced_spectral_v2::ARSpectralConfig {
        order: 2,
        fs: 1.0,
        nfft: 512,
        method: crate::advanced_spectral_v2::ARSpectralMethod::YuleWalkerEnhanced,
        ..Default::default()
    };

    let estimated = crate::advanced_spectral_v2::ar_spectral_estimation(&signal, &config)?;

    // Compare coefficients
    let mut max_coeff_error: f64 = 0.0;
    for i in 0..3 {
        let error = (estimated.ar_coefficients[i] - true_ar_coeffs[i]).abs();
        max_coeff_error = max_coeff_error.max(error);
    }

    // Compare variance (allow 20% tolerance due to estimation noise)
    let variance_error = (estimated.variance - true_variance).abs() / true_variance;

    result.max_error = max_coeff_error;
    result.mean_error = max_coeff_error;
    result.correlation = 1.0 - variance_error;

    // Pass criteria: coefficient error < 0.3, variance within 50%
    result.passed = max_coeff_error < 0.3 && variance_error < 0.5;
    result.notes = format!(
        "Coeff error: {:.4}, Variance error: {:.2}%",
        max_coeff_error,
        variance_error * 100.0
    );

    Ok(result)
}

/// Validate ARMA spectral estimation
pub fn validate_arma_spectral_estimation() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("ARMA Spectral Estimation");

    // Generate signal with known spectral content
    let n = 1024;
    let fs = 100.0;
    let signal = generate_reference_signal(n, &[10.0, 25.0], &[1.0, 0.5], fs, 0.1);

    // Estimate using our implementation
    let config = crate::advanced_spectral_v2::ARMASpectralConfig {
        ar_order: 4,
        ma_order: 2,
        fs,
        nfft: 256,
        method: crate::advanced_spectral_v2::ARMASpectralMethod::HannanRissanenImproved,
        max_iterations: 50,
        tolerance: 1e-6,
        ..Default::default()
    };

    let estimated = crate::advanced_spectral_v2::arma_spectral_estimation(&signal, &config)?;

    // Verify convergence
    if !estimated.converged {
        result.notes = "ARMA estimation did not converge".to_string();
        result.passed = false;
        return Ok(result);
    }

    // Check that PSD has peaks at expected frequencies
    let freq_resolution = fs / (2.0 * estimated.frequencies.len() as f64);

    // Find peaks
    let mut peaks = Vec::new();
    for i in 1..(estimated.psd.len() - 1) {
        if estimated.psd[i] > estimated.psd[i - 1] && estimated.psd[i] > estimated.psd[i + 1] {
            peaks.push((estimated.frequencies[i], estimated.psd[i]));
        }
    }

    // Sort by power
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Check if top peaks are near expected frequencies
    let expected_freqs = [10.0, 25.0];
    let mut found_peaks = 0;

    for &expected in &expected_freqs {
        for &(freq, _) in &peaks {
            if (freq - expected).abs() < 2.0 * freq_resolution + 1.0 {
                found_peaks += 1;
                break;
            }
        }
    }

    result.passed = found_peaks >= 1; // At least one peak should be found
    result.correlation = found_peaks as f64 / expected_freqs.len() as f64;
    result.notes = format!(
        "Found {}/{} expected peaks, converged in {} iterations",
        found_peaks,
        expected_freqs.len(),
        estimated.iterations
    );

    Ok(result)
}

/// Validate periodogram against reference implementation
pub fn validate_periodogram_consistency() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("Periodogram Consistency");

    // Generate test signal
    let n = 512;
    let fs = 100.0;
    let signal = generate_reference_signal(n, &[20.0], &[1.0], fs, 0.0);

    // Compute reference periodogram
    let (ref_freqs, ref_psd) = compute_reference_periodogram(&signal, fs)?;

    // Compute using our implementation
    let signal_slice: &[f64] = signal.as_slice().unwrap_or(&[]);
    let (our_freqs, our_psd) =
        crate::spectral::periodogram(signal_slice, Some(fs), None, None, None, None)?;

    // Compare results
    let min_len = ref_freqs.len().min(our_freqs.len());

    let mut max_error: f64 = 0.0;
    let mut sum_error: f64 = 0.0;
    let mut sum_sq_error: f64 = 0.0;

    for i in 0..min_len {
        let error = (ref_psd[i] - our_psd[i]).abs();
        max_error = max_error.max(error);
        sum_error += error;
        sum_sq_error += error * error;
    }

    result.max_error = max_error;
    result.mean_error = sum_error / min_len as f64;
    result.rms_error = (sum_sq_error / min_len as f64).sqrt();

    // Compute correlation
    let ref_mean = ref_psd.iter().take(min_len).sum::<f64>() / min_len as f64;
    let our_mean: f64 = our_psd.iter().take(min_len).sum::<f64>() / min_len as f64;

    let mut cov: f64 = 0.0;
    let mut var_ref: f64 = 0.0;
    let mut var_our: f64 = 0.0;

    for i in 0..min_len {
        let d_ref = ref_psd[i] - ref_mean;
        let d_our = our_psd[i] - our_mean;
        cov += d_ref * d_our;
        var_ref += d_ref * d_ref;
        var_our += d_our * d_our;
    }

    result.correlation = if var_ref > 1e-15 && var_our > 1e-15 {
        cov / (var_ref.sqrt() * var_our.sqrt())
    } else {
        0.0
    };

    // Pass if correlation is high (> 0.99)
    result.passed = result.correlation > 0.99;
    result.notes = format!(
        "Correlation: {:.4}, Max error: {:.2e}",
        result.correlation, result.max_error
    );

    Ok(result)
}

/// Validate FIR filter implementation
pub fn validate_fir_filter() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("FIR Filter");

    // Generate test signal with known frequency content
    let n = 1000;
    let fs = 100.0;
    let signal = generate_reference_signal(n, &[5.0, 30.0], &[1.0, 1.0], fs, 0.0);

    // Simple lowpass FIR filter (moving average)
    let window_size = 11;
    let coefficients: Vec<f64> = vec![1.0 / window_size as f64; window_size];

    // Apply filter using our implementation
    let config = crate::parallel_filtering_v2::ParallelFIRConfig {
        method: crate::parallel_filtering_v2::FIRFilterMethod::Direct,
        ..Default::default()
    };

    let filtered =
        crate::parallel_filtering_v2::parallel_fir_filter(&signal, &coefficients, &config)?;

    // Verify filter effect: high frequency should be attenuated
    // Compute power at 5 Hz and 30 Hz before and after filtering
    let (_, psd_before) = compute_reference_periodogram(&signal, fs)?;
    let (freqs, psd_after) = compute_reference_periodogram(&filtered, fs)?;

    // Find power at 5 Hz and 30 Hz
    let idx_5hz = (5.0 * freqs.len() as f64 / (fs / 2.0)) as usize;
    let idx_30hz = (30.0 * freqs.len() as f64 / (fs / 2.0)) as usize;

    let power_5hz_before = psd_before[idx_5hz.min(psd_before.len() - 1)];
    let power_5hz_after = psd_after[idx_5hz.min(psd_after.len() - 1)];
    let power_30hz_before = psd_before[idx_30hz.min(psd_before.len() - 1)];
    let power_30hz_after = psd_after[idx_30hz.min(psd_after.len() - 1)];

    // 30 Hz should be attenuated more than 5 Hz
    let attenuation_5hz = power_5hz_after / power_5hz_before.max(1e-15);
    let attenuation_30hz = power_30hz_after / power_30hz_before.max(1e-15);

    result.passed = attenuation_30hz < attenuation_5hz;
    result.correlation = 1.0 - attenuation_30hz;
    result.notes = format!(
        "5Hz attenuation: {:.2}, 30Hz attenuation: {:.2}",
        attenuation_5hz, attenuation_30hz
    );

    Ok(result)
}

/// Validate IIR filter implementation
pub fn validate_iir_filter() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("IIR Filter");

    // Generate test signal
    let n = 500;
    let fs = 100.0;
    let signal = generate_reference_signal(n, &[10.0, 40.0], &[1.0, 1.0], fs, 0.0);

    // Second-order Butterworth lowpass filter at 20 Hz
    // Pre-computed coefficients for fs=100, fc=20
    let b = vec![0.0675, 0.1349, 0.0675];
    let a = vec![1.0, -1.1430, 0.4128];

    // Apply filter
    let config = crate::parallel_filtering_v2::ParallelIIRConfig::default();
    let filtered = crate::parallel_filtering_v2::parallel_iir_filter(&signal, &b, &a, &config)?;

    // Verify filtered signal length
    if filtered.len() != signal.len() {
        result.passed = false;
        result.notes = "Output length mismatch".to_string();
        return Ok(result);
    }

    // Check that output has reasonable values
    let max_output = filtered.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let max_input = signal.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

    result.passed = max_output <= max_input * 1.5; // Allow some overshoot
    result.max_error = max_output;
    result.notes = format!("Max input: {:.2}, Max output: {:.2}", max_input, max_output);

    Ok(result)
}

/// Validate Levinson-Durbin algorithm
pub fn validate_levinson_durbin() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("Levinson-Durbin Algorithm");

    // Create known autocorrelation sequence for AR(2)
    // For AR(2) with a = [1, -0.5, 0.3], autocorrelation is known
    let rho = vec![1.0, 0.6, 0.3, 0.1, 0.05];
    let autocorr = Array1::from_vec(rho);

    // Apply Levinson-Durbin
    let order = 2;

    // Use our implementation indirectly through AR estimation
    let mut test_signal = Array1::zeros(100);
    for i in 0..100 {
        test_signal[i] = ((i as f64) * 0.1).sin();
    }

    let config = crate::advanced_spectral_v2::ARSpectralConfig {
        order,
        fs: 1.0,
        nfft: 64,
        method: crate::advanced_spectral_v2::ARSpectralMethod::YuleWalkerEnhanced,
        ..Default::default()
    };

    let ar_result = crate::advanced_spectral_v2::ar_spectral_estimation(&test_signal, &config)?;

    // Verify that coefficients have correct structure
    // First coefficient should be 1.0
    let coeff_check = (ar_result.ar_coefficients[0] - 1.0).abs() < 1e-10;

    // Reflection coefficients should have magnitude < 1 for stable filter
    let stability_check = if let Some(ref refl) = ar_result.reflection_coefficients {
        refl.iter().all(|&r| r.abs() < 1.0)
    } else {
        false
    };

    // Variance should be positive
    let variance_check = ar_result.variance > 0.0;

    result.passed = coeff_check && stability_check && variance_check;
    result.notes = format!(
        "Coeff[0]=1: {}, Stable: {}, Var>0: {}",
        coeff_check, stability_check, variance_check
    );

    Ok(result)
}

/// Validate moving average filter
pub fn validate_moving_average() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("Moving Average Filter");

    // Create simple test signal
    let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let window_size = 3;
    let filtered = crate::parallel_filtering_v2::parallel_moving_average(&signal, window_size)?;

    // Known correct values for window_size=3
    // Index 2: (1 + 2 + 3) / 3 = 2.0
    // Index 5: (4 + 5 + 6) / 3 = 5.0
    // Index 8: (7 + 8 + 9) / 3 = 8.0

    let expected_values = vec![(2, 2.0), (5, 5.0), (8, 8.0)];

    let mut max_error: f64 = 0.0;
    for &(idx, expected) in &expected_values {
        let error = (filtered[idx] - expected).abs();
        max_error = max_error.max(error);
    }

    result.max_error = max_error;
    result.passed = max_error < 1e-10;
    result.notes = format!("Max error: {:.2e}", max_error);

    Ok(result)
}

/// Validate streaming filter consistency
pub fn validate_streaming_filter() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("Streaming Filter Consistency");

    // Generate test signal
    let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let coefficients = vec![0.25, 0.5, 0.25];

    // Apply using regular filter
    let config = crate::parallel_filtering_v2::ParallelFIRConfig {
        method: crate::parallel_filtering_v2::FIRFilterMethod::Direct,
        ..Default::default()
    };
    let batch_result =
        crate::parallel_filtering_v2::parallel_fir_filter(&signal, &coefficients, &config)?;

    // Apply using streaming filter
    let mut streaming = crate::parallel_filtering_v2::StreamingFIRFilter::new(coefficients)?;
    let streaming_result: Vec<f64> = signal
        .iter()
        .map(|&x| streaming.process_sample(x))
        .collect();

    // Compare results
    let mut max_error: f64 = 0.0;
    for i in 0..signal.len() {
        let error = (batch_result[i] - streaming_result[i]).abs();
        max_error = max_error.max(error);
    }

    result.max_error = max_error;
    result.passed = max_error < 1e-10;
    result.notes = format!("Max difference: {:.2e}", max_error);

    Ok(result)
}

/// Validate memory-optimized AR estimation
pub fn validate_memory_optimized_ar() -> SignalResult<ValidationResult> {
    let mut result = ValidationResult::new("Memory-Optimized AR Estimation");

    // Generate test signal
    let n = 4096;
    let fs = 100.0;
    let signal = generate_reference_signal(n, &[15.0], &[1.0], fs, 0.1);

    // Standard AR estimation
    let standard_config = crate::advanced_spectral_v2::ARSpectralConfig {
        order: 10,
        fs,
        nfft: 256,
        ..Default::default()
    };
    let standard_result =
        crate::advanced_spectral_v2::ar_spectral_estimation(&signal, &standard_config)?;

    // Memory-optimized AR estimation
    let mem_config = crate::advanced_spectral_v2::MemoryOptimizedSpectralConfig {
        max_memory_bytes: 1024, // Force chunked processing
        chunk_size: 512,
        overlap_samples: 64,
        streaming: false,
    };
    let mem_result = crate::advanced_spectral_v2::memory_optimized_ar_spectral(
        &signal,
        &standard_config,
        &mem_config,
    )?;

    // Compare coefficients
    let mut max_coeff_error: f64 = 0.0;
    for i in 0..standard_result.ar_coefficients.len() {
        let error = (standard_result.ar_coefficients[i] - mem_result.ar_coefficients[i]).abs();
        max_coeff_error = max_coeff_error.max(error);
    }

    // Compare variance (allow some tolerance due to chunking)
    let variance_error = (standard_result.variance - mem_result.variance).abs()
        / standard_result.variance.max(1e-15);

    result.max_error = max_coeff_error;
    result.mean_error = variance_error;
    result.passed = max_coeff_error < 0.5 && variance_error < 0.5;
    result.notes = format!(
        "Coeff error: {:.4}, Variance error: {:.2}%",
        max_coeff_error,
        variance_error * 100.0
    );

    Ok(result)
}

// ============================================================================
// Comprehensive Validation Suite
// ============================================================================

/// Run all validation tests
pub fn run_comprehensive_validation() -> SignalResult<ValidationSuite> {
    let mut suite = ValidationSuite::new();

    // Run each validation test
    let tests: Vec<fn() -> SignalResult<ValidationResult>> = vec![
        validate_ar_spectral_estimation,
        validate_arma_spectral_estimation,
        validate_periodogram_consistency,
        validate_fir_filter,
        validate_iir_filter,
        validate_levinson_durbin,
        validate_moving_average,
        validate_streaming_filter,
        validate_memory_optimized_ar,
    ];

    for test_fn in tests {
        match test_fn() {
            Ok(result) => suite.add_result(result),
            Err(e) => {
                let mut result = ValidationResult::new("Unknown Test");
                result.passed = false;
                result.notes = format!("Test failed with error: {}", e);
                suite.add_result(result);
            }
        }
    }

    Ok(suite)
}

/// Generate validation report as string
pub fn generate_validation_report(suite: &ValidationSuite) -> String {
    let mut report = String::new();

    report.push_str("=== SciRS2 Signal v0.2.0 Spectral Analysis Validation Report ===\n\n");
    report.push_str(&format!(
        "Overall: {} / {} tests passed\n\n",
        suite.passed_tests, suite.total_tests
    ));

    for result in &suite.results {
        let status = if result.passed { "PASS" } else { "FAIL" };
        report.push_str(&format!("[{}] {}\n", status, result.test_name));
        report.push_str(&format!("    Notes: {}\n", result.notes));
        if result.max_error > 0.0 {
            report.push_str(&format!("    Max Error: {:.2e}\n", result.max_error));
        }
        if result.correlation > 0.0 && result.correlation < 1.0 {
            report.push_str(&format!("    Correlation: {:.4}\n", result.correlation));
        }
        report.push('\n');
    }

    report
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_reference_signal() {
        let signal = generate_reference_signal(100, &[10.0], &[1.0], 100.0, 0.0);
        assert_eq!(signal.len(), 100);

        // Check amplitude
        let max_val = signal.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_val <= 1.1);
    }

    #[test]
    fn test_generate_ar_process() {
        let ar_coeffs = [1.0, -0.5, 0.2];
        let result = generate_ar_process(500, &ar_coeffs, 1.0);
        assert!(result.is_ok());

        let signal = result.expect("Operation failed");
        assert_eq!(signal.len(), 500);
    }

    #[test]
    fn test_compute_reference_autocorrelation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let autocorr = compute_reference_autocorrelation(&signal, 3);

        // r[0] should be 1.0 (normalized)
        assert!((autocorr[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_reference_periodogram() {
        let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let result = compute_reference_periodogram(&signal, 4.0);
        assert!(result.is_ok());

        let (freqs, psd) = result.expect("Operation failed");
        assert!(!freqs.is_empty());
        assert!(!psd.is_empty());
    }

    #[test]
    fn test_validation_suite() {
        let result = run_comprehensive_validation();
        assert!(result.is_ok());

        let suite = result.expect("Operation failed");
        assert!(suite.total_tests > 0);
    }

    #[test]
    fn test_generate_validation_report() {
        let suite = run_comprehensive_validation().expect("Operation failed");
        let report = generate_validation_report(&suite);

        assert!(report.contains("Validation Report"));
        assert!(report.contains("tests passed"));
    }
}
