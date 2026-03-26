//! Delay-and-Sum (conventional) beamformer
//!
//! The conventional beamformer applies uniform weights toward the look direction:
//!
//! `w = a(theta_0) / N`
//!
//! The beamformed output is:
//!
//! `y(t) = w^H * x(t)`
//!
//! The spatial spectrum is:
//!
//! `P(theta) = w(theta)^H * R * w(theta)  =  a^H(theta) * R * a(theta) / N^2`
//!
//! This module provides:
//! - [`delay_and_sum_power`]: compute output power for a given direction
//! - [`delay_and_sum_weights`]: compute weight vector
//! - [`delay_and_sum_filter`]: apply beamformer to time-domain signals
//! - [`delay_and_sum_spectrum`]: compute spatial power spectrum
//! - [`delay_and_sum_beam_pattern`]: compute beam pattern (array factor)
//! - [`delay_and_sum_frequency_domain`]: frequency-domain beamforming using FFT
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{
    estimate_covariance_real, inner_product_conj, mat_vec_mul, steering_vector_ula,
};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of delay-and-sum beamforming
#[derive(Debug, Clone)]
pub struct DelayAndSumResult {
    /// Spatial power spectrum (one value per scan angle)
    pub power_spectrum: Vec<f64>,
    /// Scan angles in radians
    pub scan_angles: Vec<f64>,
    /// Peak angle (highest power) in radians
    pub peak_angle: f64,
    /// Peak power value
    pub peak_power: f64,
}

// ---------------------------------------------------------------------------
// Weight computation
// ---------------------------------------------------------------------------

/// Compute delay-and-sum weight vector for a given look direction
///
/// `w = a(theta_0) / N` (conjugate applied during filtering)
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `look_angle_rad` - Look direction in radians
/// * `element_spacing` - Element spacing in wavelengths
pub fn delay_and_sum_weights(
    n_elements: usize,
    look_angle_rad: f64,
    element_spacing: f64,
) -> SignalResult<Vec<Complex64>> {
    let sv = steering_vector_ula(n_elements, look_angle_rad, element_spacing)?;
    let scale = 1.0 / n_elements as f64;
    Ok(sv.iter().map(|&s| s * scale).collect())
}

// ---------------------------------------------------------------------------
// Power computation
// ---------------------------------------------------------------------------

/// Compute delay-and-sum beamformer output power
///
/// `P_DAS(theta) = a^H(theta) * R * a(theta) / M^2`
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector for the desired look direction
pub fn delay_and_sum_power(
    covariance: &[Vec<Complex64>],
    steering_vec: &[Complex64],
) -> SignalResult<f64> {
    let m = covariance.len();
    if m == 0 {
        return Err(SignalError::ValueError(
            "Covariance matrix must not be empty".to_string(),
        ));
    }
    if steering_vec.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "Steering vector length {} does not match covariance size {}",
            steering_vec.len(),
            m
        )));
    }
    for i in 0..m {
        if covariance[i].len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance row {} has length {}, expected {}",
                i,
                covariance[i].len(),
                m
            )));
        }
    }

    let r_a = mat_vec_mul(covariance, steering_vec);
    let power = inner_product_conj(steering_vec, &r_a);
    Ok(power.re / (m * m) as f64)
}

// ---------------------------------------------------------------------------
// Time-domain beamforming
// ---------------------------------------------------------------------------

/// Apply delay-and-sum beamformer to time-domain signals (narrowband approximation)
///
/// Steers the array by applying conjugate phase shifts and summing.
///
/// `y[n] = (1/M) * sum_m conj(a_m) * x_m[n]`
///
/// # Arguments
///
/// * `signals` - Multi-channel input (n_elements x n_samples), real-valued
/// * `steering_vec` - Steering vector for the desired look direction
pub fn delay_and_sum_filter(
    signals: &[Vec<f64>],
    steering_vec: &[Complex64],
) -> SignalResult<Vec<f64>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal array must not be empty".to_string(),
        ));
    }
    let n_elements = signals.len();
    let n_samples = signals[0].len();

    if steering_vec.len() != n_elements {
        return Err(SignalError::DimensionMismatch(format!(
            "Steering vector length {} does not match {} elements",
            steering_vec.len(),
            n_elements
        )));
    }
    for (idx, sig) in signals.iter().enumerate() {
        if sig.len() != n_samples {
            return Err(SignalError::DimensionMismatch(format!(
                "Channel {} has {} samples, expected {}",
                idx,
                sig.len(),
                n_samples
            )));
        }
    }

    let weights: Vec<Complex64> = steering_vec.iter().map(|s| s.conj()).collect();
    let scale = 1.0 / n_elements as f64;

    let mut output = vec![0.0; n_samples];
    for (ch_idx, channel) in signals.iter().enumerate() {
        let w_re = weights[ch_idx].re;
        for (i, &sample) in channel.iter().enumerate() {
            output[i] += w_re * sample * scale;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Spatial spectrum scanning
// ---------------------------------------------------------------------------

/// Compute delay-and-sum spatial power spectrum across a range of angles
///
/// # Arguments
///
/// * `signals` - Multi-channel time-domain signals (n_elements x n_samples)
/// * `scan_angles_rad` - Angles to evaluate
/// * `element_spacing` - Element spacing in wavelengths
pub fn delay_and_sum_spectrum(
    signals: &[Vec<f64>],
    scan_angles_rad: &[f64],
    element_spacing: f64,
) -> SignalResult<DelayAndSumResult> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal array must not be empty".to_string(),
        ));
    }
    if scan_angles_rad.is_empty() {
        return Err(SignalError::ValueError(
            "Scan angles must not be empty".to_string(),
        ));
    }

    let n_elements = signals.len();
    let cov = estimate_covariance_real(signals)?;

    let mut power_spectrum = Vec::with_capacity(scan_angles_rad.len());
    for &angle in scan_angles_rad {
        let sv = steering_vector_ula(n_elements, angle, element_spacing)?;
        let power = delay_and_sum_power(&cov, &sv)?;
        power_spectrum.push(power);
    }

    // Find peak
    let (peak_idx, &peak_power) = power_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let peak_angle = scan_angles_rad.get(peak_idx).copied().unwrap_or(0.0);

    Ok(DelayAndSumResult {
        power_spectrum,
        scan_angles: scan_angles_rad.to_vec(),
        peak_angle,
        peak_power,
    })
}

// ---------------------------------------------------------------------------
// Beam pattern
// ---------------------------------------------------------------------------

/// Compute the beam pattern (array factor) for a DAS beamformer
///
/// The beam pattern shows the array response vs angle for a given look direction.
///
/// `B(theta) = |w^H a(theta)|^2 = |sum_m conj(w_m) a_m(theta)|^2`
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `look_angle_rad` - Look direction in radians
/// * `element_spacing` - Element spacing in wavelengths
/// * `scan_angles_rad` - Angles at which to evaluate the beam pattern
pub fn delay_and_sum_beam_pattern(
    n_elements: usize,
    look_angle_rad: f64,
    element_spacing: f64,
    scan_angles_rad: &[f64],
) -> SignalResult<Vec<f64>> {
    let weights = delay_and_sum_weights(n_elements, look_angle_rad, element_spacing)?;

    let mut pattern = Vec::with_capacity(scan_angles_rad.len());
    for &angle in scan_angles_rad {
        let sv = steering_vector_ula(n_elements, angle, element_spacing)?;
        let response = inner_product_conj(&weights, &sv);
        pattern.push(response.norm_sqr());
    }
    Ok(pattern)
}

// ---------------------------------------------------------------------------
// Frequency-domain beamforming
// ---------------------------------------------------------------------------

/// Frequency-domain delay-and-sum beamforming using FFT
///
/// For wideband signals, applies narrowband beamforming per frequency bin.
/// Each frequency bin gets its own steering vector phase shift.
///
/// # Arguments
///
/// * `signals` - Multi-channel input (n_elements x n_samples), real-valued
/// * `look_angle_rad` - Look direction in radians
/// * `element_spacing` - Element spacing in wavelengths (at the reference frequency)
/// * `sample_rate` - Sampling frequency in Hz
/// * `center_frequency` - Center frequency of the signal in Hz
pub fn delay_and_sum_frequency_domain(
    signals: &[Vec<f64>],
    look_angle_rad: f64,
    element_spacing: f64,
    sample_rate: f64,
    center_frequency: f64,
) -> SignalResult<Vec<f64>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal array must not be empty".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }
    if center_frequency <= 0.0 {
        return Err(SignalError::ValueError(
            "Center frequency must be positive".to_string(),
        ));
    }

    let n_elements = signals.len();
    let n_samples = signals[0].len();
    let wavelength_ref = sample_rate / center_frequency;
    let d_meters = element_spacing * wavelength_ref;

    // FFT each channel
    let n_fft = n_samples;
    let mut spectra: Vec<Vec<Complex64>> = Vec::with_capacity(n_elements);
    for channel in signals {
        spectra.push(dft_real(channel));
    }

    // Beamform per frequency bin
    let mut output_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];
    for bin in 0..n_fft {
        let freq = bin as f64 * sample_rate / n_fft as f64;
        // Wavelength at this bin frequency
        let wavelength = if freq > 1e-10 {
            sample_rate / freq
        } else {
            // DC bin: use very large wavelength (no spatial phase shift)
            1e10
        };

        // Steering vector at this frequency
        let phase_increment = -2.0 * PI * d_meters * look_angle_rad.sin() / wavelength;
        for ch in 0..n_elements {
            let phase = phase_increment * ch as f64;
            let weight = Complex64::new(phase.cos(), -phase.sin()); // conjugate
            output_spectrum[bin] += weight * spectra[ch][bin];
        }
        output_spectrum[bin] /= n_elements as f64;
    }

    // IDFT to get time-domain output
    let output = idft_real(&output_spectrum);
    Ok(output)
}

/// Simple DFT of a real slice
fn dft_real(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut s = Complex64::new(0.0, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
                s += xj * Complex64::new(angle.cos(), angle.sin());
            }
            s
        })
        .collect()
}

/// Simple IDFT returning real part
fn idft_real(x: &[Complex64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let scale = 1.0 / n as f64;
    (0..n)
        .map(|k| {
            let mut s = Complex64::new(0.0, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * j as f64 / n as f64;
                s += xj * Complex64::new(angle.cos(), angle.sin());
            }
            s.re * scale
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beamforming::array::scan_angles_degrees;
    use approx::assert_relative_eq;

    #[test]
    fn test_das_weights_sum_to_steering_vector_normalized() {
        let n = 4;
        let angle = 0.3;
        let d = 0.5;
        let weights = delay_and_sum_weights(n, angle, d).expect("should compute weights");
        assert_eq!(weights.len(), n);
        // Weights should be a(theta)/N, so sum of norms = 1
        let norm_sum: f64 = weights.iter().map(|w| w.norm_sqr()).sum();
        assert_relative_eq!(norm_sum, 1.0 / n as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_das_power_identity_covariance() {
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        let sv = steering_vector_ula(m, 0.0, 0.5).expect("should compute SV");
        let power = delay_and_sum_power(&cov, &sv).expect("should compute DAS power");
        // a^H * I * a = M, then / M^2 = 1/M
        assert_relative_eq!(power, 1.0 / m as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_das_power_non_negative() {
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(2.0, 0.0);
        }
        cov[0][1] = Complex64::new(0.5, 0.1);
        cov[1][0] = Complex64::new(0.5, -0.1);
        let sv = steering_vector_ula(m, 0.3, 0.5).expect("should compute SV");
        let power = delay_and_sum_power(&cov, &sv).expect("should compute DAS power");
        assert!(power >= 0.0, "DAS power should be non-negative: {}", power);
    }

    #[test]
    fn test_das_power_dimension_mismatch() {
        let cov = vec![vec![Complex64::new(1.0, 0.0); 3]; 3];
        let sv = vec![Complex64::new(1.0, 0.0); 4];
        assert!(delay_and_sum_power(&cov, &sv).is_err());
    }

    #[test]
    fn test_das_filter_basic() {
        let n_samples = 100;
        let signals = vec![
            vec![1.0; n_samples],
            vec![1.0; n_samples],
            vec![1.0; n_samples],
        ];
        let sv = steering_vector_ula(3, 0.0, 0.5).expect("should compute SV");
        let output = delay_and_sum_filter(&signals, &sv).expect("should filter");
        assert_eq!(output.len(), n_samples);
        for &o in &output {
            assert_relative_eq!(o, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_das_filter_single_element() {
        let signal = vec![0.1, 0.5, -0.3, 0.8, -0.2];
        let signals = vec![signal.clone()];
        let sv = vec![Complex64::new(1.0, 0.0)];
        let output = delay_and_sum_filter(&signals, &sv).expect("should filter");
        for (o, &s) in output.iter().zip(signal.iter()) {
            assert_relative_eq!(*o, s, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_das_spectrum_has_peak_at_source() {
        // Create a signal from broadside (angle=0)
        let n_elements = 8;
        let n_samples = 200;
        let d = 0.5;
        let source_angle = 0.0; // broadside

        // All elements receive identical signal at broadside
        let signals: Vec<Vec<f64>> = (0..n_elements)
            .map(|_| {
                (0..n_samples)
                    .map(|k| (2.0 * PI * 0.1 * k as f64).sin())
                    .collect()
            })
            .collect();

        let scan = scan_angles_degrees(-60.0, 60.0, 121).expect("should create angles");
        let result = delay_and_sum_spectrum(&signals, &scan, d).expect("should compute spectrum");

        // Peak should be near broadside
        assert!(
            result.peak_angle.abs() < 0.15,
            "Peak angle should be near broadside, got {} rad",
            result.peak_angle
        );
    }

    #[test]
    fn test_das_beam_pattern_main_lobe() {
        let n = 8;
        let look = 0.2; // look direction
        let d = 0.5;
        let scan = scan_angles_degrees(-90.0, 90.0, 361).expect("should create angles");
        let pattern =
            delay_and_sum_beam_pattern(n, look, d, &scan).expect("should compute beam pattern");

        // Find peak
        let (peak_idx, _) = pattern
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("should find peak");

        let peak_angle = scan[peak_idx];
        assert!(
            (peak_angle - look).abs() < 0.05,
            "Beam pattern peak should be at look direction: peak={}, look={}",
            peak_angle,
            look
        );
    }

    #[test]
    fn test_das_frequency_domain_basic() {
        let n_elements = 4;
        let n_samples = 64;
        let signals: Vec<Vec<f64>> = (0..n_elements).map(|_| vec![1.0; n_samples]).collect();

        let result = delay_and_sum_frequency_domain(&signals, 0.0, 0.5, 1000.0, 100.0)
            .expect("should compute FD beamforming");
        assert_eq!(result.len(), n_samples);
    }
}
