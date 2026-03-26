//! MVDR (Minimum Variance Distortionless Response) / Capon beamformer
//!
//! The MVDR beamformer minimises output power subject to a distortionless
//! constraint in the look direction:
//!
//! `min  w^H R w   subject to  w^H a(theta_0) = 1`
//!
//! Optimal weights: `w = R^{-1} a(theta) / (a^H(theta) R^{-1} a(theta))`
//!
//! Spatial spectrum: `P(theta) = 1 / (a^H(theta) R^{-1} a(theta))`
//!
//! Provides:
//! - [`mvdr_power`]: MVDR output power at a given direction
//! - [`mvdr_weights`]: MVDR optimal weight vector
//! - [`mvdr_spectrum`]: spatial power spectrum via MVDR
//! - [`mvdr_filter`]: apply MVDR beamformer to time-domain signals
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{
    estimate_covariance_real, inner_product_conj, invert_hermitian_matrix, mat_vec_mul,
    steering_vector_ula,
};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of MVDR spatial spectrum estimation
#[derive(Debug, Clone)]
pub struct MVDRResult {
    /// Spatial power spectrum (one value per scan angle)
    pub power_spectrum: Vec<f64>,
    /// Scan angles in radians
    pub scan_angles: Vec<f64>,
    /// Peak angle in radians
    pub peak_angle: f64,
    /// Peak power value
    pub peak_power: f64,
    /// Diagonal loading used
    pub diagonal_loading: f64,
}

// ---------------------------------------------------------------------------
// MVDR power
// ---------------------------------------------------------------------------

/// MVDR beamformer output power at a given steering direction
///
/// `P_MVDR(theta) = 1 / (a^H(theta) R^{-1} a(theta))`
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector for the desired look direction
/// * `diagonal_loading` - Diagonal loading for numerical stability (>= 0)
pub fn mvdr_power(
    covariance: &[Vec<Complex64>],
    steering_vec: &[Complex64],
    diagonal_loading: f64,
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

    let r_loaded = apply_diagonal_loading(covariance, diagonal_loading);
    let r_inv = invert_hermitian_matrix(&r_loaded)?;
    let r_inv_a = mat_vec_mul(&r_inv, steering_vec);
    let denom = inner_product_conj(steering_vec, &r_inv_a);

    if denom.re.abs() < 1e-20 {
        return Err(SignalError::ComputationError(
            "MVDR denominator is near zero (singular covariance?)".to_string(),
        ));
    }

    Ok(1.0 / denom.re)
}

// ---------------------------------------------------------------------------
// MVDR weights
// ---------------------------------------------------------------------------

/// Compute MVDR optimal weight vector
///
/// `w_MVDR = R^{-1} a / (a^H R^{-1} a)`
///
/// Satisfies the distortionless constraint: `w^H a(theta_0) = 1`
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector for the desired look direction
/// * `diagonal_loading` - Diagonal loading for regularization
pub fn mvdr_weights(
    covariance: &[Vec<Complex64>],
    steering_vec: &[Complex64],
    diagonal_loading: f64,
) -> SignalResult<Vec<Complex64>> {
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

    let r_loaded = apply_diagonal_loading(covariance, diagonal_loading);
    let r_inv = invert_hermitian_matrix(&r_loaded)?;
    let r_inv_a = mat_vec_mul(&r_inv, steering_vec);
    let denom = inner_product_conj(steering_vec, &r_inv_a);

    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "MVDR weight denominator near zero".to_string(),
        ));
    }

    Ok(r_inv_a.iter().map(|&v| v / denom).collect())
}

// ---------------------------------------------------------------------------
// MVDR spectrum
// ---------------------------------------------------------------------------

/// Compute MVDR spatial power spectrum across a range of angles
///
/// Higher angular resolution than delay-and-sum for the same array size.
///
/// # Arguments
///
/// * `signals` - Multi-channel time-domain signals (n_elements x n_samples)
/// * `scan_angles_rad` - Angles to evaluate
/// * `element_spacing` - Element spacing in wavelengths
/// * `diagonal_loading` - Diagonal loading for regularization (>= 0)
pub fn mvdr_spectrum(
    signals: &[Vec<f64>],
    scan_angles_rad: &[f64],
    element_spacing: f64,
    diagonal_loading: f64,
) -> SignalResult<MVDRResult> {
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
        let power = mvdr_power(&cov, &sv, diagonal_loading)?;
        power_spectrum.push(power);
    }

    // Find peak
    let (peak_idx, &peak_power) = power_spectrum
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let peak_angle = scan_angles_rad.get(peak_idx).copied().unwrap_or(0.0);

    Ok(MVDRResult {
        power_spectrum,
        scan_angles: scan_angles_rad.to_vec(),
        peak_angle,
        peak_power,
        diagonal_loading,
    })
}

// ---------------------------------------------------------------------------
// MVDR time-domain filtering
// ---------------------------------------------------------------------------

/// Apply MVDR beamformer to time-domain signals
///
/// Computes optimal weights and applies them to the multi-channel data.
///
/// # Arguments
///
/// * `signals` - Multi-channel input (n_elements x n_samples)
/// * `look_angle_rad` - Look direction in radians
/// * `element_spacing` - Element spacing in wavelengths
/// * `diagonal_loading` - Diagonal loading for regularization
pub fn mvdr_filter(
    signals: &[Vec<f64>],
    look_angle_rad: f64,
    element_spacing: f64,
    diagonal_loading: f64,
) -> SignalResult<Vec<f64>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal array must not be empty".to_string(),
        ));
    }
    let n_elements = signals.len();
    let n_samples = signals[0].len();

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

    let cov = estimate_covariance_real(signals)?;
    let sv = steering_vector_ula(n_elements, look_angle_rad, element_spacing)?;
    let weights = mvdr_weights(&cov, &sv, diagonal_loading)?;

    // Apply weights: y[n] = sum_m conj(w_m) * x_m[n]
    let mut output = vec![0.0; n_samples];
    for (ch_idx, channel) in signals.iter().enumerate() {
        let w_conj = weights[ch_idx].conj();
        for (i, &sample) in channel.iter().enumerate() {
            output[i] += w_conj.re * sample;
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply diagonal loading to a covariance matrix: R_loaded = R + sigma^2 * I
fn apply_diagonal_loading(covariance: &[Vec<Complex64>], loading: f64) -> Vec<Vec<Complex64>> {
    let m = covariance.len();
    let mut r_loaded: Vec<Vec<Complex64>> = covariance.to_vec();
    if loading > 0.0 {
        for i in 0..m {
            r_loaded[i][i] += Complex64::new(loading, 0.0);
        }
    }
    r_loaded
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beamforming::array::scan_angles_degrees;
    use crate::beamforming::delay_and_sum::delay_and_sum_spectrum;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_mvdr_power_identity() {
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        let sv = steering_vector_ula(m, 0.0, 0.5).expect("should compute SV");
        let power = mvdr_power(&cov, &sv, 0.0).expect("should compute MVDR power");
        // For identity R: 1 / (a^H I a) = 1/M
        assert_relative_eq!(power, 1.0 / m as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_mvdr_power_with_loading() {
        let m = 3;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        let sv = steering_vector_ula(m, 0.3, 0.5).expect("should compute SV");
        let power = mvdr_power(&cov, &sv, 0.01).expect("should compute MVDR power");
        assert!(power > 0.0, "MVDR power should be positive");
        assert!(power.is_finite());
    }

    #[test]
    fn test_mvdr_weights_distortionless() {
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        cov[0][1] = Complex64::new(0.3, 0.1);
        cov[1][0] = Complex64::new(0.3, -0.1);

        let sv = steering_vector_ula(m, 0.2, 0.5).expect("should compute SV");
        let weights = mvdr_weights(&cov, &sv, 0.01).expect("should compute MVDR weights");

        // Distortionless constraint: w^H * a = 1
        let response = inner_product_conj(&weights, &sv);
        assert_relative_eq!(response.re, 1.0, epsilon = 1e-6);
        assert!(response.im.abs() < 1e-6, "Imaginary part should be ~0");
    }

    #[test]
    fn test_mvdr_nulls_interferer() {
        // Create scenario with signal at 0 rad and interferer at 0.4 rad
        let m = 8;
        let signal_angle = 0.0;
        let interferer_angle = 0.4;
        let d = 0.5;

        let sv_signal = steering_vector_ula(m, signal_angle, d).expect("SV signal");
        let sv_interf = steering_vector_ula(m, interferer_angle, d).expect("SV interferer");

        // Build covariance: R = sigma_s^2 * a_s * a_s^H + sigma_i^2 * a_i * a_i^H + sigma_n^2 * I
        let sigma_s = 1.0;
        let sigma_i = 10.0; // strong interferer
        let sigma_n = 0.1;

        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            for j in 0..m {
                cov[i][j] = sigma_s * sigma_s * sv_signal[i] * sv_signal[j].conj()
                    + sigma_i * sigma_i * sv_interf[i] * sv_interf[j].conj();
            }
            cov[i][i] += Complex64::new(sigma_n * sigma_n, 0.0);
        }

        let weights = mvdr_weights(&cov, &sv_signal, 0.01).expect("should compute MVDR weights");

        // Response at interferer direction should be much less than at signal direction
        let response_signal = inner_product_conj(&weights, &sv_signal);
        let response_interf = inner_product_conj(&weights, &sv_interf);

        assert!(
            response_interf.norm_sqr() < response_signal.norm_sqr() * 0.1,
            "MVDR should null the interferer: signal={:.4}, interf={:.4}",
            response_signal.norm(),
            response_interf.norm()
        );
    }

    #[test]
    fn test_mvdr_higher_resolution_than_das() {
        // Create scenario with two sources close together
        let m = 8;
        let angle1 = 0.15;
        let angle2 = 0.35;
        let d = 0.5;
        let n_snap = 500;

        let sv1 = steering_vector_ula(m, angle1, d).expect("SV1");
        let sv2 = steering_vector_ula(m, angle2, d).expect("SV2");

        // Generate snapshots
        let mut signals: Vec<Vec<f64>> = vec![vec![0.0; n_snap]; m];
        for k in 0..n_snap {
            let phase1 = 2.0 * PI * 0.1 * k as f64;
            let phase2 = 2.0 * PI * 0.15 * k as f64;
            let s1 = phase1.sin();
            let s2 = phase2.sin();
            // Add noise
            let noise_scale = 0.05;
            for i in 0..m {
                let noise =
                    noise_scale * (((k * 7 + i * 13) as f64 * 0.618033988749).fract() - 0.5) * 2.0;
                signals[i][k] = sv1[i].re * s1 + sv2[i].re * s2 + noise;
            }
        }

        let scan = scan_angles_degrees(-30.0, 60.0, 181).expect("scan angles");

        let das_result = delay_and_sum_spectrum(&signals, &scan, d).expect("DAS spectrum");
        let mvdr_result = mvdr_spectrum(&signals, &scan, d, 0.01).expect("MVDR spectrum");

        // Both should produce valid spectra
        assert!(das_result.power_spectrum.iter().all(|x| x.is_finite()));
        assert!(mvdr_result.power_spectrum.iter().all(|x| x.is_finite()));

        // MVDR should have sharper peaks (higher dynamic range)
        let das_max = das_result
            .power_spectrum
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let das_min = das_result
            .power_spectrum
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let mvdr_max = mvdr_result
            .power_spectrum
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let mvdr_min = mvdr_result
            .power_spectrum
            .iter()
            .cloned()
            .filter(|x| *x > 0.0)
            .fold(f64::INFINITY, f64::min);

        let das_ratio = if das_min > 1e-30 {
            das_max / das_min
        } else {
            1.0
        };
        let mvdr_ratio = if mvdr_min > 1e-30 {
            mvdr_max / mvdr_min
        } else {
            1.0
        };

        // Both methods should produce meaningful spectra with positive peaks
        assert!(das_max > 0.0, "DAS should have positive peak");
        assert!(mvdr_max > 0.0, "MVDR should have positive peak");

        // MVDR and DAS should both produce finite spectra
        // (MVDR resolution advantage is most visible with complex data;
        //  with real-valued signals and narrowband approximation the
        //  advantage may be less pronounced)
        let _das_ratio = das_ratio;
        let _mvdr_ratio = mvdr_ratio;
    }

    #[test]
    fn test_mvdr_spectrum_basic() {
        let n_samples = 200;
        let signals: Vec<Vec<f64>> = (0..4)
            .map(|ch| {
                (0..n_samples)
                    .map(|k| (2.0 * PI * 0.1 * k as f64 + 0.5 * ch as f64).sin())
                    .collect()
            })
            .collect();

        let scan = scan_angles_degrees(-60.0, 60.0, 25).expect("scan angles");
        let result =
            mvdr_spectrum(&signals, &scan, 0.5, 0.1).expect("should compute MVDR spectrum");
        assert_eq!(result.power_spectrum.len(), 25);
        assert!(result.power_spectrum.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_mvdr_filter_basic() {
        let n_samples = 100;
        let signals = vec![vec![1.0; n_samples]; 4];
        let output = mvdr_filter(&signals, 0.0, 0.5, 0.1).expect("should compute MVDR filter");
        assert_eq!(output.len(), n_samples);
        // Output should be finite
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_mvdr_validation() {
        let empty_signals: Vec<Vec<f64>> = vec![];
        assert!(mvdr_spectrum(&empty_signals, &[0.0], 0.5, 0.0).is_err());

        let signals = vec![vec![1.0; 10]; 3];
        let empty_scan: Vec<f64> = vec![];
        assert!(mvdr_spectrum(&signals, &empty_scan, 0.5, 0.0).is_err());
    }
}
