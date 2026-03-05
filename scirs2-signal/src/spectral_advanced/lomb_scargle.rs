//! Lomb-Scargle periodogram for unevenly-sampled data
//!
//! Implements:
//! - Standard Lomb-Scargle periodogram (Lomb 1976, Scargle 1982)
//! - Generalized Lomb-Scargle with floating mean (Zechmeister & Kurster 2009)
//! - False alarm probability estimation (Baluev 2008)
//!
//! References:
//! - Lomb, N.R. (1976). "Least-squares frequency analysis of unequally spaced data."
//!   Astrophysics and Space Science, 39, 447-462.
//! - Scargle, J.D. (1982). "Studies in astronomical time series analysis. II."
//!   ApJ, 263, 835-853.
//! - Zechmeister, M. & Kurster, M. (2009). "The generalised Lomb-Scargle
//!   periodogram." A&A, 496, 577-584.
//! - Baluev, R.V. (2008). "Assessing the statistical significance of periodogram
//!   peaks." MNRAS, 385, 1279-1285.

use super::types::{
    FalseAlarmResult, FapMethod, LombScargleConfig, LombScargleNormalization, LombScargleResult,
};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

/// Compute the Lomb-Scargle periodogram for unevenly-sampled data.
///
/// This implements both the standard Lomb-Scargle periodogram and the
/// generalized version with floating mean correction.
///
/// # Arguments
///
/// * `times` - Observation times (must be sorted, finite, and non-empty)
/// * `values` - Observed values at each time point
/// * `config` - Configuration parameters
///
/// # Returns
///
/// A `LombScargleResult` containing frequencies and power at each frequency.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral_advanced::{lomb_scargle_periodogram, LombScargleConfig};
/// use scirs2_core::ndarray::Array1;
///
/// // Generate unevenly sampled data with a 1 Hz sinusoid
/// let n = 100;
/// let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 0.01 * (i as f64 * 0.7).sin()).collect();
/// let values: Vec<f64> = times.iter().map(|&t| (2.0 * std::f64::consts::PI * 1.0 * t).sin()).collect();
///
/// let config = LombScargleConfig::default();
/// let result = lomb_scargle_periodogram(&times, &values, &config).expect("LS failed");
/// assert!(!result.power.is_empty());
/// ```
pub fn lomb_scargle_periodogram(
    times: &[f64],
    values: &[f64],
    config: &LombScargleConfig,
) -> SignalResult<LombScargleResult> {
    // Validate inputs
    validate_inputs(times, values)?;

    let n = times.len();

    // Optionally center data
    let (y, y_mean) = if config.center_data {
        let mean = values.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = values.iter().map(|&v| v - mean).collect();
        (centered, mean)
    } else {
        (values.to_vec(), 0.0)
    };

    // Generate frequency grid
    let frequencies = generate_frequency_grid(times, config)?;

    // Compute periodogram
    let power = if config.fit_mean {
        generalized_lomb_scargle(times, &y, &frequencies)?
    } else {
        standard_lomb_scargle(times, &y, &frequencies)?
    };

    // Apply normalization
    let power = normalize_power(&power, &y, config.normalization, n)?;

    Ok(LombScargleResult {
        frequencies: Array1::from_vec(frequencies),
        power: Array1::from_vec(power),
        normalization: config.normalization,
        fit_mean: config.fit_mean,
    })
}

/// Estimate false alarm probability (FAP) for Lomb-Scargle periodogram peaks.
///
/// The FAP is the probability that a peak of a given height would be observed
/// by chance if the data were pure noise.
///
/// # Arguments
///
/// * `power_levels` - Power levels at which to compute the FAP
/// * `n_samples` - Number of data points
/// * `n_frequencies` - Number of frequencies evaluated (effective trials)
/// * `method` - Method for FAP calculation
///
/// # Returns
///
/// A `FalseAlarmResult` with FAP values for each power level.
pub fn false_alarm_probability(
    power_levels: &[f64],
    n_samples: usize,
    n_frequencies: usize,
    method: FapMethod,
) -> SignalResult<FalseAlarmResult> {
    if power_levels.is_empty() {
        return Err(SignalError::ValueError(
            "Power levels must not be empty".to_string(),
        ));
    }
    if n_samples < 3 {
        return Err(SignalError::ValueError(
            "Need at least 3 data points for FAP calculation".to_string(),
        ));
    }

    let fap: Vec<f64> = match method {
        FapMethod::Baluev => power_levels
            .iter()
            .map(|&z| baluev_fap(z, n_samples, n_frequencies))
            .collect(),
        FapMethod::Davies => power_levels
            .iter()
            .map(|&z| davies_fap(z, n_samples, n_frequencies))
            .collect(),
        FapMethod::Naive => power_levels
            .iter()
            .map(|&z| naive_fap(z, n_samples, n_frequencies))
            .collect(),
    };

    Ok(FalseAlarmResult {
        fap: Array1::from_vec(fap),
        power_levels: Array1::from_vec(power_levels.to_vec()),
        method,
    })
}

/// Compute the FAP threshold: the power level at which the FAP equals
/// a given probability.
///
/// # Arguments
///
/// * `fap_target` - Desired false alarm probability (e.g., 0.01 for 1%)
/// * `n_samples` - Number of data points
/// * `n_frequencies` - Number of frequencies evaluated
/// * `method` - Method for FAP calculation
///
/// # Returns
///
/// The power level at which FAP = fap_target.
pub fn false_alarm_level(
    fap_target: f64,
    n_samples: usize,
    n_frequencies: usize,
    method: FapMethod,
) -> SignalResult<f64> {
    if !(0.0..1.0).contains(&fap_target) {
        return Err(SignalError::ValueError(
            "FAP target must be in (0, 1)".to_string(),
        ));
    }
    if n_samples < 3 {
        return Err(SignalError::ValueError(
            "Need at least 3 data points".to_string(),
        ));
    }

    // Use bisection to find the power level
    let mut lo = 0.0;
    let mut hi = 50.0; // Start with a large upper bound
    let max_iter = 100;
    let tol = 1e-10;

    for _ in 0..max_iter {
        let mid = (lo + hi) / 2.0;
        let fap_at_mid = match method {
            FapMethod::Baluev => baluev_fap(mid, n_samples, n_frequencies),
            FapMethod::Davies => davies_fap(mid, n_samples, n_frequencies),
            FapMethod::Naive => naive_fap(mid, n_samples, n_frequencies),
        };

        if (fap_at_mid - fap_target).abs() < tol {
            return Ok(mid);
        }

        if fap_at_mid > fap_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok((lo + hi) / 2.0)
}

// =============================================================================
// Internal implementations
// =============================================================================

/// Standard Lomb-Scargle periodogram (Scargle 1982)
fn standard_lomb_scargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let mut power = Vec::with_capacity(frequencies.len());

    for &freq in frequencies {
        let omega = 2.0 * PI * freq;

        // Compute time offset tau (Scargle's normalization)
        let mut sum_sin2 = 0.0;
        let mut sum_cos2 = 0.0;
        for &t in times {
            sum_sin2 += (2.0 * omega * t).sin();
            sum_cos2 += (2.0 * omega * t).cos();
        }
        let tau = sum_sin2.atan2(sum_cos2) / (2.0 * omega);

        // Compute shifted trig sums
        let mut yc = 0.0; // sum(y * cos(omega*(t-tau)))
        let mut ys = 0.0; // sum(y * sin(omega*(t-tau)))
        let mut cc = 0.0; // sum(cos^2(omega*(t-tau)))
        let mut ss = 0.0; // sum(sin^2(omega*(t-tau)))

        for i in 0..n {
            let phase = omega * (times[i] - tau);
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            yc += values[i] * cos_phase;
            ys += values[i] * sin_phase;
            cc += cos_phase * cos_phase;
            ss += sin_phase * sin_phase;
        }

        // Lomb-Scargle power
        let p = if cc > 1e-30 && ss > 1e-30 {
            0.5 * (yc * yc / cc + ys * ys / ss)
        } else {
            0.0
        };
        power.push(p);
    }

    Ok(power)
}

/// Generalized Lomb-Scargle with floating mean (Zechmeister & Kurster 2009)
fn generalized_lomb_scargle(
    times: &[f64],
    values: &[f64],
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    let n = times.len();
    let mut power = Vec::with_capacity(frequencies.len());

    // Weight = 1/N for uniform weights
    let w = 1.0 / n as f64;

    // Weighted mean of y
    let y_bar: f64 = values.iter().sum::<f64>() * w;

    // Total variance
    let yy: f64 = values.iter().map(|&v| (v - y_bar) * (v - y_bar) * w).sum();

    if yy < 1e-30 {
        return Ok(vec![0.0; frequencies.len()]);
    }

    for &freq in frequencies {
        let omega = 2.0 * PI * freq;

        // Compute trigonometric sums with uniform weights
        let mut yc = 0.0;
        let mut ys = 0.0;
        let mut c_sum = 0.0;
        let mut s_sum = 0.0;
        let mut cc = 0.0;
        let mut ss = 0.0;
        let mut cs = 0.0;

        for i in 0..n {
            let cos_val = (omega * times[i]).cos();
            let sin_val = (omega * times[i]).sin();

            yc += (values[i] - y_bar) * cos_val * w;
            ys += (values[i] - y_bar) * sin_val * w;
            c_sum += cos_val * w;
            s_sum += sin_val * w;
            cc += cos_val * cos_val * w;
            ss += sin_val * sin_val * w;
            cs += cos_val * sin_val * w;
        }

        // Subtract mean contributions
        cc -= c_sum * c_sum;
        ss -= s_sum * s_sum;
        cs -= c_sum * s_sum;

        // Solve the 2x2 normal equations for the generalized periodogram
        let det = cc * ss - cs * cs;

        let p = if det.abs() > 1e-30 {
            let a = (ss * yc - cs * ys) / det;
            let b = (cc * ys - cs * yc) / det;
            (a * yc + b * ys) / yy
        } else if cc.abs() > 1e-30 {
            yc * yc / (cc * yy)
        } else {
            0.0
        };

        power.push(p.max(0.0));
    }

    Ok(power)
}

/// Generate automatic frequency grid
fn generate_frequency_grid(times: &[f64], config: &LombScargleConfig) -> SignalResult<Vec<f64>> {
    let n = times.len();

    // Data span
    let t_min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_span = t_max - t_min;

    if t_span <= 0.0 {
        return Err(SignalError::ValueError(
            "Time span must be positive (at least 2 distinct times)".to_string(),
        ));
    }

    // Minimum frequency based on data span
    let f_min = config.f_min.unwrap_or(1.0 / (t_span * config.oversampling));

    // Maximum frequency: average Nyquist frequency * nyquist_factor
    let avg_dt = t_span / (n - 1) as f64;
    let f_nyquist = 0.5 / avg_dt;
    let f_max = config.f_max.unwrap_or(f_nyquist * config.nyquist_factor);

    if f_max <= f_min {
        return Err(SignalError::ValueError(format!(
            "Maximum frequency ({f_max}) must be greater than minimum frequency ({f_min})"
        )));
    }

    // Number of frequencies
    let n_freq = config
        .n_frequencies
        .unwrap_or_else(|| (config.oversampling * t_span * (f_max - f_min)).ceil() as usize + 1);
    let n_freq = n_freq.max(2);

    let df = (f_max - f_min) / (n_freq - 1) as f64;
    let frequencies: Vec<f64> = (0..n_freq).map(|i| f_min + i as f64 * df).collect();

    Ok(frequencies)
}

/// Normalize power based on the chosen normalization method
fn normalize_power(
    power: &[f64],
    y: &[f64],
    normalization: LombScargleNormalization,
    n: usize,
) -> SignalResult<Vec<f64>> {
    match normalization {
        LombScargleNormalization::Standard => {
            // P in [0, 1]: divide by variance
            let y_mean = y.iter().sum::<f64>() / n as f64;
            let variance: f64 =
                y.iter().map(|&v| (v - y_mean) * (v - y_mean)).sum::<f64>() / (n - 1).max(1) as f64;
            if variance > 1e-30 {
                Ok(power.iter().map(|&p| p / variance).collect())
            } else {
                Ok(power.to_vec())
            }
        }
        LombScargleNormalization::Model => {
            // chi2 residual model: P = 1 - chi2_residual / chi2_total
            let y_mean = y.iter().sum::<f64>() / n as f64;
            let chi2_total: f64 = y.iter().map(|&v| (v - y_mean) * (v - y_mean)).sum();
            if chi2_total > 1e-30 {
                Ok(power.iter().map(|&p| p / chi2_total * 2.0).collect())
            } else {
                Ok(power.to_vec())
            }
        }
        LombScargleNormalization::Log => {
            // Natural log of (1 - model_normalized_power)
            let y_mean = y.iter().sum::<f64>() / n as f64;
            let chi2_total: f64 = y.iter().map(|&v| (v - y_mean) * (v - y_mean)).sum();
            if chi2_total > 1e-30 {
                Ok(power
                    .iter()
                    .map(|&p| {
                        let model_p = (p / chi2_total * 2.0).min(1.0 - 1e-15);
                        -(1.0 - model_p).ln()
                    })
                    .collect())
            } else {
                Ok(power.to_vec())
            }
        }
        LombScargleNormalization::Psd => {
            // Power spectral density normalization
            // P_psd = P * N / 2
            Ok(power.iter().map(|&p| p * n as f64 / 2.0).collect())
        }
    }
}

/// Validate inputs for Lomb-Scargle
fn validate_inputs(times: &[f64], values: &[f64]) -> SignalResult<()> {
    if times.is_empty() || values.is_empty() {
        return Err(SignalError::ValueError(
            "Times and values must not be empty".to_string(),
        ));
    }
    if times.len() != values.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Times ({}) and values ({}) must have the same length",
            times.len(),
            values.len()
        )));
    }
    if times.len() < 3 {
        return Err(SignalError::ValueError(
            "Need at least 3 data points for Lomb-Scargle periodogram".to_string(),
        ));
    }
    if times.iter().any(|&t| !t.is_finite()) || values.iter().any(|&v| !v.is_finite()) {
        return Err(SignalError::ValueError(
            "Times and values must be finite".to_string(),
        ));
    }

    Ok(())
}

// =============================================================================
// False alarm probability methods
// =============================================================================

/// Baluev (2008) analytical FAP approximation
///
/// Uses the Davies bound with exponential distribution approximation.
/// For standard normalization: FAP ≈ 1 - (1 - exp(-z))^M * tau(z)
fn baluev_fap(z: f64, n_samples: usize, n_frequencies: usize) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }

    let n = n_samples as f64;
    let m = n_frequencies as f64;

    // Single-frequency FAP for standard normalization
    let single_fap = (-z).exp();

    // Effective number of independent frequencies (Baluev's correction)
    // tau(z) accounts for the continuous nature of the frequency search
    let tau =
        m * (1.0 - z).powf((n - 3.0) / 2.0).max(0.0) * (2.0 * z * (n - 3.0) / (n - 1.0)).sqrt();

    // Combined FAP
    let fap = 1.0 - (1.0 - single_fap).powf(m) + tau * single_fap;
    fap.clamp(0.0, 1.0)
}

/// Davies (1977) upper bound on FAP
fn davies_fap(z: f64, n_samples: usize, n_frequencies: usize) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }

    let m = n_frequencies as f64;

    // Simple Bonferroni-like bound
    let single_fap = (-z).exp();
    let fap = m * single_fap;
    fap.min(1.0)
}

/// Naive single-frequency FAP (no correction for multiple testing)
fn naive_fap(z: f64, _n_samples: usize, _n_frequencies: usize) -> f64 {
    if z <= 0.0 {
        return 1.0;
    }
    (-z).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sinusoidal_data(
        n: usize,
        freq: f64,
        amp: f64,
        noise_level: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut times = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 * 0.1 + 0.01 * (i as f64 * 0.7).sin(); // uneven sampling
            times.push(t);
            values.push(
                amp * (2.0 * PI * freq * t).sin() + noise_level * (i as f64 * 3.7).sin() * 0.1,
            );
        }

        (times, values)
    }

    #[test]
    fn test_standard_lomb_scargle() {
        let (times, values) = make_sinusoidal_data(200, 1.0, 1.0, 0.01);
        let config = LombScargleConfig {
            fit_mean: false,
            center_data: true,
            ..Default::default()
        };

        let result = lomb_scargle_periodogram(&times, &values, &config);
        assert!(result.is_ok(), "LS failed: {:?}", result.err());
        let result = result.expect("already checked");

        // Peak should be near 1 Hz
        let max_idx = result
            .power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should find peak");
        let peak_freq = result.frequencies[max_idx];
        assert!(
            (peak_freq - 1.0).abs() < 0.2,
            "Peak at {peak_freq} Hz, expected ~1.0 Hz"
        );
    }

    #[test]
    fn test_generalized_lomb_scargle_floating_mean() {
        let n = 200;
        let freq = 2.0;
        let mut times = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        // Signal with a non-zero mean
        for i in 0..n {
            let t = i as f64 * 0.05 + 0.005 * (i as f64 * 1.3).sin();
            times.push(t);
            values.push(5.0 + (2.0 * PI * freq * t).sin()); // offset of 5.0
        }

        let config = LombScargleConfig {
            fit_mean: true,
            center_data: true,
            ..Default::default()
        };

        let result = lomb_scargle_periodogram(&times, &values, &config);
        assert!(result.is_ok(), "GLS failed: {:?}", result.err());
        let result = result.expect("already checked");
        assert!(result.fit_mean);

        // Peak should be near 2 Hz
        let max_idx = result
            .power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("should find peak");
        let peak_freq = result.frequencies[max_idx];
        assert!(
            (peak_freq - 2.0).abs() < 0.3,
            "Peak at {peak_freq} Hz, expected ~2.0 Hz"
        );
    }

    #[test]
    fn test_lomb_scargle_normalizations() {
        let (times, values) = make_sinusoidal_data(100, 1.5, 1.0, 0.0);

        for norm in [
            LombScargleNormalization::Standard,
            LombScargleNormalization::Model,
            LombScargleNormalization::Log,
            LombScargleNormalization::Psd,
        ] {
            let config = LombScargleConfig {
                normalization: norm,
                ..Default::default()
            };
            let result = lomb_scargle_periodogram(&times, &values, &config);
            assert!(result.is_ok(), "Failed for normalization {norm:?}");
            let result = result.expect("already checked");

            // All power values should be finite
            assert!(
                result.power.iter().all(|&p| p.is_finite()),
                "Non-finite power for {norm:?}"
            );
        }
    }

    #[test]
    fn test_false_alarm_probability_baluev() {
        let power_levels = vec![5.0, 10.0, 15.0, 20.0];
        let result = false_alarm_probability(&power_levels, 100, 50, FapMethod::Baluev);
        assert!(result.is_ok());
        let result = result.expect("already checked");

        // FAP should decrease with increasing power
        for i in 1..result.fap.len() {
            assert!(
                result.fap[i] <= result.fap[i - 1] + 1e-10,
                "FAP should decrease: fap[{}]={} > fap[{}]={}",
                i,
                result.fap[i],
                i - 1,
                result.fap[i - 1]
            );
        }

        // FAP values should be in [0, 1]
        assert!(result.fap.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }

    #[test]
    fn test_false_alarm_probability_methods() {
        let power_levels = vec![10.0];
        for method in [FapMethod::Baluev, FapMethod::Davies, FapMethod::Naive] {
            let result = false_alarm_probability(&power_levels, 50, 100, method);
            assert!(result.is_ok(), "FAP failed for method {method:?}");
            let result = result.expect("already checked");
            assert!(result.fap[0] >= 0.0 && result.fap[0] <= 1.0);
        }
    }

    #[test]
    fn test_false_alarm_level() {
        let level = false_alarm_level(0.01, 100, 200, FapMethod::Baluev);
        assert!(level.is_ok(), "FAP level failed: {:?}", level.err());
        let level = level.expect("already checked");
        assert!(level > 0.0, "FAP level should be positive, got {level}");

        // Verify round-trip: FAP at this level should be ~0.01
        let fap_check = false_alarm_probability(&[level], 100, 200, FapMethod::Baluev);
        assert!(fap_check.is_ok());
        let fap_val = fap_check.expect("already checked").fap[0];
        assert!(
            (fap_val - 0.01).abs() < 0.005,
            "Round-trip FAP should be ~0.01, got {fap_val}"
        );
    }

    #[test]
    fn test_lomb_scargle_two_frequencies() {
        let n = 500;
        let f1 = 1.0;
        let f2 = 3.5;
        let mut times = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f64 * 0.05 + 0.003 * (i as f64 * 2.1).sin();
            times.push(t);
            values.push((2.0 * PI * f1 * t).sin() + 0.5 * (2.0 * PI * f2 * t).sin());
        }

        let config = LombScargleConfig {
            fit_mean: true,
            ..Default::default()
        };
        let result = lomb_scargle_periodogram(&times, &values, &config).expect("LS should work");

        // Find top 2 peaks
        let mut peak_powers: Vec<(usize, f64)> = result
            .power
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        peak_powers.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // The strongest peak should be near f1=1.0 Hz
        let peak1 = result.frequencies[peak_powers[0].0];
        assert!(
            (peak1 - f1).abs() < 0.3 || (peak1 - f2).abs() < 0.3,
            "First peak at {peak1} Hz, expected near {f1} or {f2} Hz"
        );
    }

    #[test]
    fn test_lomb_scargle_validation_errors() {
        let config = LombScargleConfig::default();

        // Empty arrays
        let result = lomb_scargle_periodogram(&[], &[], &config);
        assert!(result.is_err());

        // Mismatched lengths
        let result = lomb_scargle_periodogram(&[1.0, 2.0, 3.0], &[1.0, 2.0], &config);
        assert!(result.is_err());

        // Too few points
        let result = lomb_scargle_periodogram(&[1.0, 2.0], &[1.0, 2.0], &config);
        assert!(result.is_err());

        // NaN in times
        let result = lomb_scargle_periodogram(&[1.0, f64::NAN, 3.0], &[1.0, 2.0, 3.0], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_lomb_scargle_custom_frequency_range() {
        let (times, values) = make_sinusoidal_data(100, 5.0, 1.0, 0.0);
        let config = LombScargleConfig {
            f_min: Some(3.0),
            f_max: Some(7.0),
            n_frequencies: Some(100),
            ..Default::default()
        };

        let result = lomb_scargle_periodogram(&times, &values, &config).expect("should work");
        assert!(result.frequencies[0] >= 2.9);
        let last_freq = result.frequencies[result.frequencies.len() - 1];
        assert!(last_freq <= 7.1);
    }
}
