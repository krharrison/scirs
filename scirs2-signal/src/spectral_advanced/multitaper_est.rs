//! Enhanced multitaper spectral estimation (Thomson's method)
//!
//! Implements Thomson's multitaper method with:
//! - DPSS (discrete prolate spheroidal sequences) windows via dpss_enhanced
//! - Adaptive weighting algorithm (Thomson 1982)
//! - F-test for line component detection
//!
//! References:
//! - Thomson, D.J. (1982). "Spectrum estimation and harmonic analysis."
//!   Proceedings of the IEEE, 70(9), 1055-1096.
//! - Percival, D.B. and Walden, A.T. (1993). "Spectral Analysis for
//!   Physical Applications." Cambridge University Press.

use super::types::{FTestResult, MultitaperConfig, MultitaperResult};
use crate::error::{SignalError, SignalResult};
use crate::multitaper::dpss_enhanced::dpss_enhanced;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Compute multitaper power spectral density estimate using Thomson's method.
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - Multitaper configuration parameters
///
/// # Returns
///
/// A `MultitaperResult` containing frequencies, PSD estimate, optional weights
/// and eigenspectra.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral_advanced::{multitaper_psd, MultitaperConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 256;
/// let t = Array1::linspace(0.0, 1.0, n);
/// let signal: Vec<f64> = t.iter().map(|&ti| {
///     (2.0 * std::f64::consts::PI * 10.0 * ti).sin()
/// }).collect();
///
/// let config = MultitaperConfig { nw: 4.0, fs: 256.0, ..Default::default() };
/// let result = multitaper_psd(&signal, &config).expect("multitaper_psd failed");
/// assert!(!result.psd.is_empty());
/// ```
pub fn multitaper_psd(signal: &[f64], config: &MultitaperConfig) -> SignalResult<MultitaperResult> {
    // Validate input
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    if config.nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product NW must be positive".to_string(),
        ));
    }
    if config.fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let k = config.k.unwrap_or_else(|| {
        let max_k = (2.0 * config.nw).floor() as usize;
        if max_k > 0 {
            max_k - 1
        } else {
            1
        }
    });

    if k == 0 {
        return Err(SignalError::ValueError(
            "Number of tapers K must be at least 1".to_string(),
        ));
    }

    // Compute DPSS tapers
    let (tapers, ratios) = dpss_enhanced(n, config.nw, k, true)?;
    let concentration_ratios = ratios.ok_or_else(|| {
        SignalError::ComputationError("Failed to compute concentration ratios".to_string())
    })?;

    // Determine NFFT
    let nfft = config.nfft.unwrap_or_else(|| next_power_of_2(n));
    let nfft = nfft.max(n);

    // Compute eigenspectra for each taper
    let n_positive_freqs = nfft / 2 + 1;
    let mut eigenspectra = Array2::zeros((k, n_positive_freqs));

    for taper_idx in 0..k {
        // Apply taper to signal
        let mut tapered = vec![0.0; nfft];
        for i in 0..n {
            tapered[i] = signal[i] * tapers[[taper_idx, i]];
        }

        // FFT using scirs2-fft compatible approach (manual DFT for correctness)
        let spectrum = compute_rfft(&tapered, nfft)?;

        // Compute power spectrum: |X(f)|^2
        for freq_idx in 0..n_positive_freqs {
            eigenspectra[[taper_idx, freq_idx]] = spectrum[freq_idx].norm_sqr();
        }
    }

    // Compute PSD estimate
    let psd = if config.adaptive {
        adaptive_weighting(&eigenspectra, &concentration_ratios, config)?
    } else {
        // Simple average of eigenspectra weighted by concentration ratios
        simple_weighted_average(&eigenspectra, &concentration_ratios)?
    };

    // Scale PSD: divide by fs to get proper units (power/Hz)
    let scale = 1.0 / config.fs;
    let psd = psd.mapv(|v| v * scale);

    // Build frequency vector
    let freq_step = config.fs / nfft as f64;
    let frequencies = Array1::from_vec(
        (0..n_positive_freqs)
            .map(|i| i as f64 * freq_step)
            .collect(),
    );

    Ok(MultitaperResult {
        frequencies,
        psd,
        weights: None, // Weights are internal to adaptive_weighting
        eigenspectra: Some(eigenspectra),
        concentration_ratios,
    })
}

/// Compute multitaper F-test for line component detection.
///
/// The F-test identifies deterministic sinusoidal components in a signal
/// by testing whether the coherent power (captured by the mean of tapered
/// Fourier coefficients) is significantly larger than the residual.
///
/// # Arguments
///
/// * `signal` - Input time series data
/// * `config` - Multitaper configuration parameters
/// * `significance_level` - p-value threshold for significance (e.g., 0.01)
///
/// # Returns
///
/// An `FTestResult` with F-statistics, p-values, and detected line components.
pub fn multitaper_ftest_line_detection(
    signal: &[f64],
    config: &MultitaperConfig,
    significance_level: f64,
) -> SignalResult<FTestResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal must not be empty".to_string(),
        ));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&significance_level) {
        return Err(SignalError::ValueError(
            "Significance level must be in (0, 1)".to_string(),
        ));
    }

    let n = signal.len();
    let k = config.k.unwrap_or_else(|| {
        let max_k = (2.0 * config.nw).floor() as usize;
        if max_k > 0 {
            max_k - 1
        } else {
            1
        }
    });

    if k < 2 {
        return Err(SignalError::ValueError(
            "F-test requires at least 2 tapers".to_string(),
        ));
    }

    // Compute DPSS tapers
    let (tapers, ratios) = dpss_enhanced(n, config.nw, k, true)?;
    let concentration_ratios = ratios.ok_or_else(|| {
        SignalError::ComputationError("Failed to compute concentration ratios".to_string())
    })?;

    let nfft = config.nfft.unwrap_or_else(|| next_power_of_2(n));
    let nfft = nfft.max(n);
    let n_positive_freqs = nfft / 2 + 1;

    // Compute complex Fourier coefficients for each taper
    let mut coefficients = Array2::from_elem((k, n_positive_freqs), Complex64::new(0.0, 0.0));
    // Taper sums (H_k(0) = sum of taper k values) for the F-test
    let mut taper_sums = Array1::zeros(k);

    for taper_idx in 0..k {
        let mut tapered = vec![0.0; nfft];
        for i in 0..n {
            tapered[i] = signal[i] * tapers[[taper_idx, i]];
            taper_sums[taper_idx] += tapers[[taper_idx, i]];
        }

        let spectrum = compute_rfft(&tapered, nfft)?;
        for freq_idx in 0..n_positive_freqs {
            coefficients[[taper_idx, freq_idx]] = spectrum[freq_idx];
        }
    }

    // Compute F-test at each frequency
    // F = (K-1) * |sum_k H_k(0) * Y_k(f)|^2 / (sum_k |H_k(0)|^2 * sum_k |Y_k(f) - mu(f)*H_k(f)|^2)
    //
    // Simplified approach: use the mean of the tapered transforms as the line estimate
    let mut f_statistic = Array1::zeros(n_positive_freqs);
    let mut p_values = Array1::ones(n_positive_freqs);
    let mut line_amplitudes = Array1::zeros(n_positive_freqs);

    // Compute taper spectrum at DC (zero frequency): H_k(0) = sum_i v_k[i]
    let taper_norm_sq: f64 = taper_sums.iter().map(|&h| h * h).sum();

    for freq_idx in 0..n_positive_freqs {
        // Weighted mean (line component estimate)
        let mut numerator = Complex64::new(0.0, 0.0);
        for taper_idx in 0..k {
            numerator +=
                Complex64::new(taper_sums[taper_idx], 0.0) * coefficients[[taper_idx, freq_idx]];
        }
        let mu_hat = if taper_norm_sq > 1e-15 {
            numerator / taper_norm_sq
        } else {
            Complex64::new(0.0, 0.0)
        };

        line_amplitudes[freq_idx] = mu_hat.norm();

        // Compute coherent power (numerator of F-test)
        let coherent_power = mu_hat.norm_sqr() * taper_norm_sq;

        // Compute residual power (denominator of F-test)
        let mut residual_power = 0.0;
        for taper_idx in 0..k {
            let residual = coefficients[[taper_idx, freq_idx]] - mu_hat * taper_sums[taper_idx];
            residual_power += residual.norm_sqr();
        }

        // F-statistic with (2, 2*(K-1)) degrees of freedom
        // For real data: F = (K-1) * coherent / residual
        if residual_power > 1e-30 {
            f_statistic[freq_idx] = (k as f64 - 1.0) * coherent_power / residual_power;
        }

        // Approximate p-value using F-distribution CDF
        // F(1, K-1) distribution -> use beta incomplete function approximation
        let f_val = f_statistic[freq_idx];
        let df1 = 1.0;
        let df2 = k as f64 - 1.0;
        p_values[freq_idx] = f_distribution_survival(f_val, df1, df2);
    }

    // Find significant line components
    let significant_indices: Vec<usize> = p_values
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < significance_level)
        .map(|(i, _)| i)
        .collect();

    // Build frequency vector
    let freq_step = config.fs / nfft as f64;
    let frequencies = Array1::from_vec(
        (0..n_positive_freqs)
            .map(|i| i as f64 * freq_step)
            .collect(),
    );

    Ok(FTestResult {
        frequencies,
        f_statistic,
        p_values,
        line_amplitudes,
        significant_indices,
    })
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Thomson's adaptive weighting algorithm
///
/// Iteratively computes weights d_k(f) that minimize broadband bias
/// while preserving resolution. Based on equation (5.4) in Thomson (1982).
fn adaptive_weighting(
    eigenspectra: &Array2<f64>,
    concentration_ratios: &Array1<f64>,
    config: &MultitaperConfig,
) -> SignalResult<Array1<f64>> {
    let k = eigenspectra.nrows();
    let n_freq = eigenspectra.ncols();

    // Initial PSD estimate: simple average
    let mut psd = simple_weighted_average(eigenspectra, concentration_ratios)?;

    // Estimate broadband variance (noise level)
    let broadband_var = psd.iter().sum::<f64>() / n_freq as f64;

    // Iterative adaptive weighting
    for _iteration in 0..config.max_adaptive_iter {
        let mut new_psd = Array1::zeros(n_freq);
        let mut max_change = 0.0_f64;

        for freq_idx in 0..n_freq {
            let s_hat = psd[freq_idx];
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for taper_idx in 0..k {
                let lambda_k = concentration_ratios[taper_idx];

                // Adaptive weight: d_k(f) = sqrt(lambda_k) * S(f) / (lambda_k * S(f) + (1-lambda_k) * sigma^2)
                let denominator = lambda_k * s_hat + (1.0 - lambda_k) * broadband_var;
                let weight = if denominator > 1e-30 {
                    let d_k = (lambda_k.sqrt() * s_hat) / denominator;
                    d_k * d_k // Use d_k^2 as the actual weight
                } else {
                    0.0
                };

                weighted_sum += weight * eigenspectra[[taper_idx, freq_idx]];
                weight_sum += weight;
            }

            new_psd[freq_idx] = if weight_sum > 1e-30 {
                weighted_sum / weight_sum
            } else {
                s_hat
            };

            let change = if s_hat > 1e-30 {
                ((new_psd[freq_idx] - s_hat) / s_hat).abs()
            } else {
                0.0
            };
            max_change = max_change.max(change);
        }

        psd = new_psd;

        if max_change < config.adaptive_tol {
            break;
        }
    }

    Ok(psd)
}

/// Simple weighted average of eigenspectra
fn simple_weighted_average(
    eigenspectra: &Array2<f64>,
    concentration_ratios: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let k = eigenspectra.nrows();
    let n_freq = eigenspectra.ncols();

    let total_weight: f64 = concentration_ratios.iter().sum();
    if total_weight < 1e-30 {
        return Err(SignalError::ComputationError(
            "Sum of concentration ratios is too small".to_string(),
        ));
    }

    let mut psd = Array1::zeros(n_freq);
    for freq_idx in 0..n_freq {
        let mut weighted = 0.0;
        for taper_idx in 0..k {
            weighted += concentration_ratios[taper_idx] * eigenspectra[[taper_idx, freq_idx]];
        }
        psd[freq_idx] = weighted / total_weight;
    }

    Ok(psd)
}

/// Compute real FFT returning complex spectrum for positive frequencies
fn compute_rfft(signal: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    let n_positive = nfft / 2 + 1;
    let mut result = vec![Complex64::new(0.0, 0.0); n_positive];

    // DFT for positive frequencies only
    for freq_idx in 0..n_positive {
        let omega = 2.0 * PI * freq_idx as f64 / nfft as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &val) in signal.iter().enumerate() {
            let phase = omega * t as f64;
            re += val * phase.cos();
            im -= val * phase.sin();
        }
        result[freq_idx] = Complex64::new(re, im);
    }

    Ok(result)
}

/// Next power of 2 >= n
fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Survival function (1 - CDF) of the F-distribution
/// Approximation using the regularized incomplete beta function
fn f_distribution_survival(f_val: f64, df1: f64, df2: f64) -> f64 {
    if f_val <= 0.0 {
        return 1.0;
    }
    // For F(1, v) distribution: P(F > x) = (1 + x/v)^(-(v+1)/2) * correction
    // Use the relation between F and Beta distributions:
    // If X ~ F(d1, d2), then d1*X/(d1*X + d2) ~ Beta(d1/2, d2/2)
    let x = df1 * f_val / (df1 * f_val + df2);
    // P(F > f_val) = 1 - I_x(df1/2, df2/2)
    // where I_x is the regularized incomplete beta function
    let a = df1 / 2.0;
    let b = df2 / 2.0;
    regularized_incomplete_beta_complement(x, a, b)
}

/// Complement of the regularized incomplete beta function: 1 - I_x(a, b)
/// Uses a continued fraction expansion for numerical stability.
fn regularized_incomplete_beta_complement(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }

    // For the F-test case with df1=1, use a simpler approximation
    // P(F(1,v) > x) approximately equals (1 + x/v)^(-(v+1)/2) for moderate v
    // More accurate: use the relation P(F(1,v) > x) = 2 * P(t(v) > sqrt(x))
    // where t(v) is the Student's t-distribution with v dof
    if (a - 0.5).abs() < 1e-10 {
        // df1 = 1 case: use Student-t survival function
        let t_val = x.sqrt();
        return student_t_survival(t_val, 2.0 * b);
    }

    // General case: Newton's approximation for the beta distribution
    // Use a series expansion
    beta_cf_approximation(x, a, b)
}

/// Student's t survival function: P(T > t) where T ~ t(nu)
fn student_t_survival(t_val: f64, nu: f64) -> f64 {
    if !t_val.is_finite() {
        return 0.0;
    }
    // P(T > t) = 0.5 * I_{v/(v+t^2)}(v/2, 1/2)
    // For large nu, approximation: P(T > t) ≈ 0.5 * erfc(t / sqrt(2))
    if nu > 100.0 {
        return 0.5 * erfc_approx(t_val / std::f64::consts::SQRT_2);
    }

    // For general nu, use the continued fraction representation
    let x = nu / (nu + t_val * t_val);
    0.5 * beta_cf_approximation(x, nu / 2.0, 0.5)
}

/// Complementary error function approximation (Abramowitz & Stegun)
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    // Horner form of rational approximation (Abramowitz & Stegun 7.1.26)
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    poly * (-x * x).exp()
}

/// Regularized incomplete beta function via continued fraction (Lentz's method)
fn beta_cf_approximation(x: f64, a: f64, b: f64) -> f64 {
    // If x > (a+1)/(a+b+2), use the symmetry relation
    let threshold = (a + 1.0) / (a + b + 2.0);
    if x > threshold {
        return 1.0 - beta_cf_approximation(1.0 - x, b, a);
    }

    // Compute I_x(a, b) using continued fraction, return 1 - I_x(a, b)
    let prefix = x.powf(a) * (1.0 - x).powf(b) / (a * beta_function(a, b));

    // Continued fraction (Lentz's algorithm)
    let max_iter = 200;
    let tiny = 1e-30;
    let eps = 1e-12;

    let mut f = tiny;
    let mut c = tiny;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    f = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step
        let a_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + a_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let a_odd = -((a + m_f) * (a + b + m_f)) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + a_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    // I_x(a, b) = prefix * f
    let i_x = prefix * f;
    1.0 - i_x.clamp(0.0, 1.0)
}

/// Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
/// Uses Stirling's approximation for the log-gamma function
fn beta_function(a: f64, b: f64) -> f64 {
    (log_gamma(a) + log_gamma(b) - log_gamma(a + b)).exp()
}

/// Stirling's log-gamma approximation (Lanczos approximation)
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation (g=7, n=9)
    let coeff = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let g = 7.0;
    let x_shifted = x - 1.0;

    let mut sum = coeff[0];
    for (i, &c) in coeff.iter().enumerate().skip(1) {
        sum += c / (x_shifted + i as f64);
    }

    let t = x_shifted + g + 0.5;
    0.5 * (2.0 * PI).ln() + (x_shifted + 0.5) * t.ln() - t + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sinusoidal(n: usize, fs: f64, freqs: &[f64], amps: &[f64]) -> Vec<f64> {
        let mut signal = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 / fs;
            for (f, a) in freqs.iter().zip(amps.iter()) {
                signal[i] += a * (2.0 * PI * f * t).sin();
            }
        }
        signal
    }

    #[test]
    fn test_multitaper_psd_basic() {
        let n = 256;
        let fs = 256.0;
        let signal = generate_sinusoidal(n, fs, &[30.0], &[1.0]);
        let config = MultitaperConfig {
            nw: 4.0,
            fs,
            adaptive: false,
            ..Default::default()
        };

        let result = multitaper_psd(&signal, &config);
        assert!(result.is_ok(), "multitaper_psd failed: {:?}", result.err());
        let result = result.expect("already checked");
        assert!(!result.psd.is_empty());
        assert_eq!(result.frequencies.len(), result.psd.len());

        // PSD should have a peak near 30 Hz
        let peak_idx = result
            .psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should have peak");
        let peak_freq = result.frequencies[peak_idx];
        assert!(
            (peak_freq - 30.0).abs() < 5.0,
            "Peak at {peak_freq} Hz, expected ~30 Hz"
        );
    }

    #[test]
    fn test_multitaper_psd_adaptive() {
        let n = 256;
        let fs = 256.0;
        let signal = generate_sinusoidal(n, fs, &[20.0, 60.0], &[1.0, 0.5]);
        let config = MultitaperConfig {
            nw: 4.0,
            fs,
            adaptive: true,
            ..Default::default()
        };

        let result = multitaper_psd(&signal, &config);
        assert!(
            result.is_ok(),
            "adaptive multitaper failed: {:?}",
            result.err()
        );
        let result = result.expect("already checked");
        assert!(
            result.psd.iter().all(|&v| v >= 0.0),
            "PSD must be non-negative"
        );
    }

    #[test]
    fn test_multitaper_eigenspectra_shape() {
        let n = 128;
        let fs = 128.0;
        let signal = generate_sinusoidal(n, fs, &[10.0], &[1.0]);
        let config = MultitaperConfig {
            nw: 3.0,
            k: Some(5),
            fs,
            adaptive: false,
            ..Default::default()
        };

        let result = multitaper_psd(&signal, &config).expect("should succeed");
        let eigenspectra = result.eigenspectra.expect("eigenspectra should be present");
        assert_eq!(eigenspectra.nrows(), 5, "Should have 5 eigenspectra");
        assert_eq!(
            eigenspectra.ncols(),
            result.frequencies.len(),
            "Eigenspectra cols should match frequency count"
        );
    }

    #[test]
    fn test_multitaper_concentration_ratios() {
        let n = 128;
        let signal = vec![1.0; n]; // constant signal
        let config = MultitaperConfig {
            nw: 4.0,
            fs: 128.0,
            ..Default::default()
        };

        let result = multitaper_psd(&signal, &config).expect("should succeed");
        // First few concentration ratios should be close to 1
        assert!(
            result.concentration_ratios[0] > 0.99,
            "First ratio should be ~1, got {}",
            result.concentration_ratios[0]
        );
        // Ratios should be decreasing
        for i in 1..result.concentration_ratios.len() {
            assert!(
                result.concentration_ratios[i] <= result.concentration_ratios[i - 1] + 1e-10,
                "Ratios should be non-increasing"
            );
        }
    }

    #[test]
    fn test_multitaper_psd_validation_errors() {
        let config = MultitaperConfig::default();

        // Empty signal
        let result = multitaper_psd(&[], &config);
        assert!(result.is_err());

        // NaN in signal
        let result = multitaper_psd(&[1.0, f64::NAN, 3.0], &config);
        assert!(result.is_err());

        // Invalid NW
        let bad_config = MultitaperConfig {
            nw: -1.0,
            ..Default::default()
        };
        let result = multitaper_psd(&[1.0, 2.0, 3.0], &bad_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ftest_line_detection() {
        let n = 128;
        let fs = 128.0;
        // Signal with strong sinusoid at 20 Hz + small pseudo-noise
        let mut signal = generate_sinusoidal(n, fs, &[20.0], &[10.0]);
        for (i, val) in signal.iter_mut().enumerate() {
            *val += 0.01 * ((i as f64 * 7.3).sin() + (i as f64 * 13.7).cos());
        }

        let config = MultitaperConfig {
            nw: 4.0,
            k: Some(5),
            fs,
            nfft: Some(128),
            ..Default::default()
        };

        let result = multitaper_ftest_line_detection(&signal, &config, 0.05);
        assert!(result.is_ok(), "F-test failed: {:?}", result.err());
        let result = result.expect("already checked");

        // F-statistics should be finite and non-negative
        assert!(result
            .f_statistic
            .iter()
            .all(|&f| f.is_finite() && f >= 0.0));
        // p-values should be in [0,1]
        assert!(result.p_values.iter().all(|&p| (0.0..=1.0).contains(&p)));

        // The F-statistic near the signal frequency should be among the highest
        let nfft_actual = 128;
        let freq_step = fs / nfft_actual as f64;
        let idx_20 = (20.0 / freq_step).round() as usize;
        if idx_20 < result.f_statistic.len() {
            // The F-stat at the signal frequency should be substantial
            let mean_f: f64 =
                result.f_statistic.iter().sum::<f64>() / result.f_statistic.len() as f64;
            assert!(
                result.f_statistic[idx_20] > mean_f,
                "F-stat at 20Hz ({}) should exceed mean ({})",
                result.f_statistic[idx_20],
                mean_f
            );
        }
    }

    #[test]
    fn test_ftest_no_line_in_noise() {
        let n = 256;
        let fs = 256.0;
        // Pure "noise" (deterministic pseudo-random)
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x * 2.3).sin() * 0.1
                    + (x * 5.7).cos() * 0.08
                    + (x * 11.1).sin() * 0.06
                    + (x * 17.3).cos() * 0.04
                    + (x * 23.9).sin() * 0.03
            })
            .collect();

        let config = MultitaperConfig {
            nw: 4.0,
            fs,
            ..Default::default()
        };

        let result = multitaper_ftest_line_detection(&signal, &config, 0.001);
        assert!(result.is_ok());
        // In broadband noise, few or no lines should be detected at very low significance
        // (This is a weak test since pseudo-random may have some structure)
    }

    #[test]
    fn test_ftest_validation_errors() {
        let config = MultitaperConfig::default();

        // Empty signal
        let result = multitaper_ftest_line_detection(&[], &config, 0.05);
        assert!(result.is_err());

        // Invalid significance level
        let result = multitaper_ftest_line_detection(&[1.0, 2.0, 3.0, 4.0], &config, -0.1);
        assert!(result.is_err());

        // Too few tapers
        let bad_config = MultitaperConfig {
            nw: 0.5,
            k: Some(1),
            ..Default::default()
        };
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = multitaper_ftest_line_detection(&signal, &bad_config, 0.05);
        assert!(result.is_err());
    }
}
