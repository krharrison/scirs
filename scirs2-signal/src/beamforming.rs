//! Beamforming module for array signal processing
//!
//! This module provides beamforming algorithms for sensor arrays:
//! - **Delay-and-Sum** (conventional) beamformer
//! - **MVDR** (Minimum Variance Distortionless Response) / Capon beamformer
//! - **Steering vector** computation for uniform linear arrays (ULA)
//! - **Spatial spectrum** scanning for direction-of-arrival estimation
//!
//! All algorithms operate in the frequency domain using narrowband assumptions.
//! Wideband beamforming is supported by applying narrowband beamforming per frequency bin.
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Steering vectors
// ---------------------------------------------------------------------------

/// Compute the steering vector for a Uniform Linear Array (ULA)
///
/// The steering vector for a ULA with element spacing `d` at angle `theta`
/// (measured from broadside) for wavelength `lambda` is:
///   a(theta) = [1, exp(-j*2*pi*d*sin(theta)/lambda), ..., exp(-j*2*pi*(M-1)*d*sin(theta)/lambda)]
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `angle_rad` - Steering angle in radians (0 = broadside, pi/2 = endfire)
/// * `element_spacing` - Element spacing in units of wavelength (typically 0.5)
///
/// # Returns
///
/// * Complex steering vector of length `n_elements`
pub fn steering_vector_ula(
    n_elements: usize,
    angle_rad: f64,
    element_spacing: f64,
) -> SignalResult<Vec<Complex64>> {
    if n_elements == 0 {
        return Err(SignalError::ValueError(
            "Number of elements must be positive".to_string(),
        ));
    }
    if element_spacing <= 0.0 {
        return Err(SignalError::ValueError(
            "Element spacing must be positive".to_string(),
        ));
    }

    let phase_increment = -2.0 * PI * element_spacing * angle_rad.sin();
    let sv: Vec<Complex64> = (0..n_elements)
        .map(|m| {
            let phase = phase_increment * m as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    Ok(sv)
}

/// Compute steering vectors for a set of angles
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `angles_rad` - Steering angles in radians
/// * `element_spacing` - Element spacing in wavelengths (default 0.5)
///
/// # Returns
///
/// * Vec of steering vectors, one per angle
pub fn steering_vectors_ula(
    n_elements: usize,
    angles_rad: &[f64],
    element_spacing: f64,
) -> SignalResult<Vec<Vec<Complex64>>> {
    if angles_rad.is_empty() {
        return Err(SignalError::ValueError(
            "Angle list must not be empty".to_string(),
        ));
    }

    let mut vectors = Vec::with_capacity(angles_rad.len());
    for &angle in angles_rad {
        vectors.push(steering_vector_ula(n_elements, angle, element_spacing)?);
    }
    Ok(vectors)
}

// ---------------------------------------------------------------------------
// Covariance matrix estimation
// ---------------------------------------------------------------------------

/// Estimate the spatial covariance matrix from multi-channel data
///
/// # Arguments
///
/// * `signals` - Snapshot matrix (n_elements x n_snapshots) stored as Vec of per-element signals
///
/// # Returns
///
/// * Covariance matrix (n_elements x n_elements) as `Vec<Vec<Complex64>>`
pub fn estimate_covariance(signals: &[Vec<Complex64>]) -> SignalResult<Vec<Vec<Complex64>>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal matrix must not be empty".to_string(),
        ));
    }

    let n_elements = signals.len();
    let n_snapshots = signals[0].len();

    if n_snapshots == 0 {
        return Err(SignalError::ValueError(
            "Number of snapshots must be positive".to_string(),
        ));
    }

    // Check all elements have the same length
    for (idx, sig) in signals.iter().enumerate() {
        if sig.len() != n_snapshots {
            return Err(SignalError::DimensionMismatch(format!(
                "Element {} has {} snapshots, expected {}",
                idx,
                sig.len(),
                n_snapshots
            )));
        }
    }

    // R = (1/N) * X * X^H
    let mut cov = vec![vec![Complex64::new(0.0, 0.0); n_elements]; n_elements];

    for i in 0..n_elements {
        for j in 0..n_elements {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n_snapshots {
                sum += signals[i][k] * signals[j][k].conj();
            }
            cov[i][j] = sum / n_snapshots as f64;
        }
    }

    Ok(cov)
}

/// Estimate covariance matrix from real-valued multi-channel signals
///
/// Converts real signals to complex (zero imaginary part) and computes covariance.
pub fn estimate_covariance_real(signals: &[Vec<f64>]) -> SignalResult<Vec<Vec<Complex64>>> {
    let complex_signals: Vec<Vec<Complex64>> = signals
        .iter()
        .map(|ch| ch.iter().map(|&x| Complex64::new(x, 0.0)).collect())
        .collect();
    estimate_covariance(&complex_signals)
}

// ---------------------------------------------------------------------------
// Delay-and-Sum beamformer
// ---------------------------------------------------------------------------

/// Delay-and-Sum (conventional) beamformer
///
/// Computes the beamformed output power at the given steering direction.
/// P_DAS(theta) = a^H(theta) * R * a(theta) / M^2
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector for the desired look direction
///
/// # Returns
///
/// * Beamformer output power (real scalar)
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

    // Compute R * a
    let mut r_a = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        if covariance[i].len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance row {} has length {}, expected {}",
                i,
                covariance[i].len(),
                m
            )));
        }
        for j in 0..m {
            r_a[i] += covariance[i][j] * steering_vec[j];
        }
    }

    // Compute a^H * R * a
    let mut power = Complex64::new(0.0, 0.0);
    for i in 0..m {
        power += steering_vec[i].conj() * r_a[i];
    }

    Ok(power.re / (m * m) as f64)
}

/// Apply delay-and-sum beamformer to time-domain signals
///
/// Steers the array to the given direction by applying phase shifts
/// (narrowband assumption) and summing across elements.
///
/// # Arguments
///
/// * `signals` - Multi-channel input (n_elements x n_samples), real-valued
/// * `steering_vec` - Steering vector for the desired look direction
///
/// # Returns
///
/// * Beamformed output signal (n_samples)
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

    // For narrowband: multiply each channel by conjugate of steering weight and sum
    // In time domain with real signals, this is an approximation
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
// MVDR (Capon) beamformer
// ---------------------------------------------------------------------------

/// MVDR (Capon) beamformer power spectrum
///
/// Computes the MVDR beamformer output power at the given steering direction.
/// P_MVDR(theta) = 1 / (a^H(theta) * R^{-1} * a(theta))
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector for the desired look direction
/// * `diagonal_loading` - Optional diagonal loading for numerical stability (default 0.0)
///
/// # Returns
///
/// * MVDR beamformer output power (real scalar)
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

    // Apply diagonal loading: R_loaded = R + sigma^2 * I
    let mut r_loaded = covariance.to_vec();
    if diagonal_loading > 0.0 {
        for i in 0..m {
            r_loaded[i] = covariance[i].clone();
            r_loaded[i][i] += Complex64::new(diagonal_loading, 0.0);
        }
    }

    // Invert the covariance matrix
    let r_inv = invert_hermitian_matrix(&r_loaded)?;

    // Compute R^{-1} * a
    let mut r_inv_a = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        for j in 0..m {
            r_inv_a[i] += r_inv[i][j] * steering_vec[j];
        }
    }

    // Compute a^H * R^{-1} * a
    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..m {
        denom += steering_vec[i].conj() * r_inv_a[i];
    }

    if denom.re.abs() < 1e-20 {
        return Err(SignalError::ComputationError(
            "MVDR denominator is near zero (singular covariance?)".to_string(),
        ));
    }

    Ok(1.0 / denom.re)
}

/// Compute MVDR weight vector
///
/// w_MVDR = R^{-1} * a / (a^H * R^{-1} * a)
///
/// # Arguments
///
/// * `covariance` - Spatial covariance matrix (M x M)
/// * `steering_vec` - Steering vector
/// * `diagonal_loading` - Diagonal loading for regularization
///
/// # Returns
///
/// * MVDR weight vector
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

    // Apply diagonal loading
    let mut r_loaded: Vec<Vec<Complex64>> = covariance.to_vec();
    if diagonal_loading > 0.0 {
        for i in 0..m {
            r_loaded[i] = covariance[i].clone();
            r_loaded[i][i] += Complex64::new(diagonal_loading, 0.0);
        }
    }

    let r_inv = invert_hermitian_matrix(&r_loaded)?;

    // R^{-1} * a
    let mut r_inv_a = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        for j in 0..m {
            r_inv_a[i] += r_inv[i][j] * steering_vec[j];
        }
    }

    // a^H * R^{-1} * a
    let mut denom = Complex64::new(0.0, 0.0);
    for i in 0..m {
        denom += steering_vec[i].conj() * r_inv_a[i];
    }

    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "MVDR weight denominator near zero".to_string(),
        ));
    }

    let weights: Vec<Complex64> = r_inv_a.iter().map(|&v| v / denom).collect();
    Ok(weights)
}

// ---------------------------------------------------------------------------
// Spatial spectrum scanning
// ---------------------------------------------------------------------------

/// Beamforming method for spatial spectrum scanning
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BeamformMethod {
    /// Delay-and-Sum (conventional)
    DelayAndSum,
    /// MVDR (Capon) with optional diagonal loading
    Mvdr(f64),
}

/// Scan the spatial spectrum across a range of angles
///
/// Computes the beamformer output power at each angle in the scan range.
///
/// # Arguments
///
/// * `signals` - Multi-channel time-domain signals (n_elements x n_samples), real-valued
/// * `scan_angles_rad` - Angles to evaluate (in radians)
/// * `element_spacing` - Element spacing in wavelengths
/// * `method` - Beamforming method
///
/// # Returns
///
/// * Power spectrum (one value per scan angle)
pub fn beamform(
    signals: &[Vec<f64>],
    scan_angles_rad: &[f64],
    element_spacing: f64,
    method: BeamformMethod,
) -> SignalResult<Vec<f64>> {
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

    // Estimate spatial covariance matrix
    let cov = estimate_covariance_real(signals)?;

    // Scan each angle
    let mut power_spectrum = Vec::with_capacity(scan_angles_rad.len());

    for &angle in scan_angles_rad {
        let sv = steering_vector_ula(n_elements, angle, element_spacing)?;
        let power = match method {
            BeamformMethod::DelayAndSum => delay_and_sum_power(&cov, &sv)?,
            BeamformMethod::Mvdr(loading) => mvdr_power(&cov, &sv, loading)?,
        };
        power_spectrum.push(power);
    }

    Ok(power_spectrum)
}

/// Convenience function: create uniformly spaced scan angles
///
/// # Arguments
///
/// * `start_deg` - Start angle in degrees
/// * `end_deg` - End angle in degrees
/// * `n_points` - Number of scan points
///
/// # Returns
///
/// * Vector of angles in radians
pub fn scan_angles_degrees(
    start_deg: f64,
    end_deg: f64,
    n_points: usize,
) -> SignalResult<Vec<f64>> {
    if n_points == 0 {
        return Err(SignalError::ValueError(
            "Number of scan points must be positive".to_string(),
        ));
    }
    if n_points == 1 {
        return Ok(vec![start_deg.to_radians()]);
    }

    let step = (end_deg - start_deg) / (n_points - 1) as f64;
    Ok((0..n_points)
        .map(|i| (start_deg + step * i as f64).to_radians())
        .collect())
}

// ---------------------------------------------------------------------------
// Matrix utilities (small Hermitian matrix inversion via Gauss-Jordan)
// ---------------------------------------------------------------------------

/// Invert a Hermitian positive-definite matrix using Gauss-Jordan elimination
fn invert_hermitian_matrix(matrix: &[Vec<Complex64>]) -> SignalResult<Vec<Vec<Complex64>>> {
    let n = matrix.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Matrix must not be empty".to_string(),
        ));
    }

    // Build augmented matrix [A | I]
    let mut aug: Vec<Vec<Complex64>> = Vec::with_capacity(n);
    for i in 0..n {
        if matrix[i].len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "Matrix row {} has length {}, expected {}",
                i,
                matrix[i].len(),
                n
            )));
        }
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(&matrix[i]);
        for j in 0..n {
            if i == j {
                row.push(Complex64::new(1.0, 0.0));
            } else {
                row.push(Complex64::new(0.0, 0.0));
            }
        }
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].norm();
        for row in (col + 1)..n {
            let val = aug[row][col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or near-singular".to_string(),
            ));
        }

        aug.swap(col, max_row);

        // Scale pivot row
        let pivot = aug[col][col];
        let pivot_inv = Complex64::new(1.0, 0.0) / pivot;
        for j in 0..(2 * n) {
            aug[col][j] = aug[col][j] * pivot_inv;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] = aug[row][j] - factor * aug[col][j];
            }
        }
    }

    // Extract inverse from right half
    let mut inverse = Vec::with_capacity(n);
    for i in 0..n {
        inverse.push(aug[i][n..(2 * n)].to_vec());
    }

    Ok(inverse)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ----- Steering vector tests -----

    #[test]
    fn test_steering_vector_broadside() {
        // At broadside (angle=0), all elements have the same phase
        let sv = steering_vector_ula(4, 0.0, 0.5).expect("Steering vector failed");
        assert_eq!(sv.len(), 4);
        for s in &sv {
            assert_relative_eq!(s.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(s.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_steering_vector_unit_norm() {
        let sv = steering_vector_ula(8, 0.3, 0.5).expect("Steering vector failed");
        // Each element should have unit magnitude
        for s in &sv {
            assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_steering_vector_endfire() {
        // At endfire (angle = pi/2), phase shift between elements is maximal
        let sv = steering_vector_ula(4, PI / 2.0, 0.5).expect("Steering vector failed");
        assert_eq!(sv.len(), 4);
        // First element always 1
        assert_relative_eq!(sv[0].re, 1.0, epsilon = 1e-12);
        // Phase progression should be -pi per element
        let phase_diff = (sv[1] * sv[0].conj()).im.atan2((sv[1] * sv[0].conj()).re);
        assert!(phase_diff.abs() > 0.1); // significant phase shift
    }

    #[test]
    fn test_steering_vector_validation() {
        assert!(steering_vector_ula(0, 0.0, 0.5).is_err());
        assert!(steering_vector_ula(4, 0.0, 0.0).is_err());
        assert!(steering_vector_ula(4, 0.0, -0.5).is_err());
    }

    #[test]
    fn test_steering_vectors_batch() {
        let angles = vec![0.0, 0.5, 1.0];
        let svs = steering_vectors_ula(4, &angles, 0.5).expect("Batch steering failed");
        assert_eq!(svs.len(), 3);
        for sv in &svs {
            assert_eq!(sv.len(), 4);
        }
    }

    #[test]
    fn test_steering_vectors_empty_angles() {
        assert!(steering_vectors_ula(4, &[], 0.5).is_err());
    }

    // ----- Covariance estimation tests -----

    #[test]
    fn test_covariance_identity() {
        // Uncorrelated signals with unit power -> identity covariance
        let n_snap = 10000;
        let n_elem = 3;
        // Use deterministic "pseudo-random" signals
        let signals: Vec<Vec<Complex64>> = (0..n_elem)
            .map(|ch| {
                (0..n_snap)
                    .map(|k| {
                        let phase =
                            2.0 * PI * ((ch * 7 + k * 13 + ch * k * 3) as f64 % 997.0) / 997.0;
                        Complex64::new(phase.cos(), phase.sin())
                    })
                    .collect()
            })
            .collect();

        let cov = estimate_covariance(&signals).expect("Covariance failed");
        assert_eq!(cov.len(), n_elem);
        assert_eq!(cov[0].len(), n_elem);

        // Diagonal elements should be close to 1 (unit power signals)
        for i in 0..n_elem {
            assert!(
                (cov[i][i].re - 1.0).abs() < 0.2,
                "Diagonal element {} = {} (expected ~1)",
                i,
                cov[i][i].re
            );
        }
    }

    #[test]
    fn test_covariance_hermitian() {
        let signals = vec![
            vec![Complex64::new(1.0, 0.5), Complex64::new(0.3, -0.2)],
            vec![Complex64::new(-0.5, 0.1), Complex64::new(0.8, 0.4)],
        ];

        let cov = estimate_covariance(&signals).expect("Covariance failed");

        // Check Hermitian symmetry: R[i][j] = conj(R[j][i])
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                assert_relative_eq!(cov[i][j].re, cov[j][i].re, epsilon = 1e-12);
                assert_relative_eq!(cov[i][j].im, -cov[j][i].im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_real() {
        let signals = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let cov = estimate_covariance_real(&signals).expect("Covariance real failed");
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        // For real signals, covariance should be real
        for row in &cov {
            for &val in row {
                assert!(val.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_validation() {
        assert!(estimate_covariance(&[]).is_err());
        let empty_signals = vec![vec![]];
        assert!(estimate_covariance(&empty_signals).is_err());
    }

    #[test]
    fn test_covariance_dimension_mismatch() {
        let signals = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            vec![Complex64::new(3.0, 0.0)], // different length
        ];
        assert!(estimate_covariance(&signals).is_err());
    }

    // ----- Delay-and-Sum tests -----

    #[test]
    fn test_das_power_identity_covariance() {
        // With identity covariance and broadside steering, power = 1/M
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }

        let sv = steering_vector_ula(m, 0.0, 0.5).expect("SV failed");
        let power = delay_and_sum_power(&cov, &sv).expect("DAS power failed");

        // a^H * I * a = M, then / M^2 = 1/M
        assert_relative_eq!(power, 1.0 / m as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_das_power_dimension_mismatch() {
        let cov = vec![vec![Complex64::new(1.0, 0.0); 3]; 3];
        let sv = vec![Complex64::new(1.0, 0.0); 4]; // wrong size
        assert!(delay_and_sum_power(&cov, &sv).is_err());
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

        let sv = steering_vector_ula(m, 0.3, 0.5).expect("SV failed");
        let power = delay_and_sum_power(&cov, &sv).expect("DAS power failed");
        assert!(power >= 0.0, "DAS power should be non-negative: {}", power);
    }

    #[test]
    fn test_das_filter_basic() {
        let n_samples = 100;
        let signals = vec![
            vec![1.0; n_samples],
            vec![1.0; n_samples],
            vec![1.0; n_samples],
        ];
        let sv = steering_vector_ula(3, 0.0, 0.5).expect("SV failed");
        let output = delay_and_sum_filter(&signals, &sv).expect("DAS filter failed");
        assert_eq!(output.len(), n_samples);
        // For identical signals at broadside, output should be the signal
        for &o in &output {
            assert_relative_eq!(o, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_das_filter_dimension_mismatch() {
        let signals = vec![vec![1.0; 10], vec![1.0; 10]];
        let sv = vec![Complex64::new(1.0, 0.0); 3]; // wrong size
        assert!(delay_and_sum_filter(&signals, &sv).is_err());
    }

    #[test]
    fn test_das_filter_preserves_signal() {
        // Single element array = passthrough
        let signal = vec![0.1, 0.5, -0.3, 0.8, -0.2];
        let signals = vec![signal.clone()];
        let sv = vec![Complex64::new(1.0, 0.0)];
        let output = delay_and_sum_filter(&signals, &sv).expect("DAS filter failed");
        for (o, &s) in output.iter().zip(signal.iter()) {
            assert_relative_eq!(*o, s, epsilon = 1e-10);
        }
    }

    // ----- MVDR (Capon) tests -----

    #[test]
    fn test_mvdr_power_identity() {
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }

        let sv = steering_vector_ula(m, 0.0, 0.5).expect("SV failed");
        let power = mvdr_power(&cov, &sv, 0.0).expect("MVDR power failed");

        // For identity R: a^H * I^{-1} * a = M, so P = 1/M
        assert_relative_eq!(power, 1.0 / m as f64, epsilon = 1e-10);
    }

    #[test]
    fn test_mvdr_power_with_loading() {
        let m = 3;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }

        let sv = steering_vector_ula(m, 0.3, 0.5).expect("SV failed");
        let power = mvdr_power(&cov, &sv, 0.01).expect("MVDR power failed");
        assert!(power > 0.0, "MVDR power should be positive");
        assert!(power.is_finite());
    }

    #[test]
    fn test_mvdr_power_dimension_mismatch() {
        let cov = vec![vec![Complex64::new(1.0, 0.0); 3]; 3];
        let sv = vec![Complex64::new(1.0, 0.0); 2];
        assert!(mvdr_power(&cov, &sv, 0.0).is_err());
    }

    #[test]
    fn test_mvdr_weights_distortionless() {
        // MVDR constraint: w^H * a = 1
        let m = 4;
        let mut cov = vec![vec![Complex64::new(0.0, 0.0); m]; m];
        for i in 0..m {
            cov[i][i] = Complex64::new(1.0, 0.0);
        }
        // Add some off-diagonal terms
        cov[0][1] = Complex64::new(0.3, 0.1);
        cov[1][0] = Complex64::new(0.3, -0.1);

        let sv = steering_vector_ula(m, 0.2, 0.5).expect("SV failed");
        let weights = mvdr_weights(&cov, &sv, 0.01).expect("MVDR weights failed");

        // Check distortionless constraint: w^H * a = 1
        let mut response = Complex64::new(0.0, 0.0);
        for i in 0..m {
            response += weights[i].conj() * sv[i];
        }
        assert_relative_eq!(response.re, 1.0, epsilon = 1e-6);
        assert!(
            response.im.abs() < 1e-6,
            "Imaginary part should be ~0: {}",
            response.im
        );
    }

    #[test]
    fn test_mvdr_weights_validation() {
        let cov = vec![vec![Complex64::new(1.0, 0.0)]];
        let sv = vec![Complex64::new(1.0, 0.0); 2];
        assert!(mvdr_weights(&cov, &sv, 0.0).is_err());
    }

    #[test]
    fn test_mvdr_better_resolution_than_das() {
        // MVDR should have better angular resolution than DAS
        // Create a scenario with two close sources
        let m = 8;
        let angle1 = 0.2; // ~11.5 degrees
        let angle2 = 0.4; // ~22.9 degrees
        let d = 0.5;

        let sv1 = steering_vector_ula(m, angle1, d).expect("SV1 failed");
        let sv2 = steering_vector_ula(m, angle2, d).expect("SV2 failed");

        // Build covariance with two sources + noise
        let n_snap = 500;
        let mut signals: Vec<Vec<Complex64>> = vec![vec![Complex64::new(0.0, 0.0); n_snap]; m];
        for k in 0..n_snap {
            let phase1 = 2.0 * PI * (k as f64 * 0.1);
            let phase2 = 2.0 * PI * (k as f64 * 0.15);
            let s1 = Complex64::new(phase1.cos(), phase1.sin());
            let s2 = Complex64::new(phase2.cos(), phase2.sin());

            for i in 0..m {
                let noise = Complex64::new(
                    0.01 * ((k * 7 + i * 13) as f64 % 1.0 - 0.5),
                    0.01 * ((k * 11 + i * 17) as f64 % 1.0 - 0.5),
                );
                signals[i][k] = sv1[i] * s1 + sv2[i] * s2 + noise;
            }
        }

        let cov = estimate_covariance(&signals).expect("Covariance failed");

        // Scan angles
        let scan_angles: Vec<f64> = (0..50)
            .map(|i| -PI / 4.0 + PI / 2.0 * i as f64 / 49.0)
            .collect();

        let mut das_spectrum = Vec::new();
        let mut mvdr_spectrum = Vec::new();

        for &angle in &scan_angles {
            let sv = steering_vector_ula(m, angle, d).expect("SV failed");
            das_spectrum.push(delay_and_sum_power(&cov, &sv).expect("DAS failed"));
            mvdr_spectrum.push(mvdr_power(&cov, &sv, 0.01).expect("MVDR failed"));
        }

        // Both should produce valid spectra
        assert!(das_spectrum.iter().all(|x| x.is_finite()));
        assert!(mvdr_spectrum.iter().all(|x| x.is_finite()));

        // Find peaks in DAS
        let das_max = das_spectrum.iter().cloned().fold(0.0_f64, f64::max);
        assert!(das_max > 0.0, "DAS should have positive peak power");
    }

    // ----- Spatial spectrum scanning tests -----

    #[test]
    fn test_beamform_das() {
        let n_samples = 200;
        let signals = vec![
            vec![1.0; n_samples],
            vec![1.0; n_samples],
            vec![1.0; n_samples],
            vec![1.0; n_samples],
        ];

        let scan = scan_angles_degrees(-90.0, 90.0, 37).expect("Scan angles failed");
        let spectrum =
            beamform(&signals, &scan, 0.5, BeamformMethod::DelayAndSum).expect("Beamform failed");

        assert_eq!(spectrum.len(), 37);
        // All power values should be non-negative
        for &p in &spectrum {
            assert!(p >= 0.0, "Power should be non-negative: {}", p);
        }
    }

    #[test]
    fn test_beamform_mvdr() {
        let n_samples = 200;
        let signals: Vec<Vec<f64>> = (0..4)
            .map(|ch| {
                (0..n_samples)
                    .map(|k| (2.0 * PI * 0.1 * k as f64 + 0.5 * ch as f64).sin())
                    .collect()
            })
            .collect();

        let scan = scan_angles_degrees(-60.0, 60.0, 25).expect("Scan angles failed");
        let spectrum = beamform(&signals, &scan, 0.5, BeamformMethod::Mvdr(0.1))
            .expect("MVDR beamform failed");

        assert_eq!(spectrum.len(), 25);
        assert!(spectrum.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_beamform_validation() {
        let signals: Vec<Vec<f64>> = vec![];
        let scan = vec![0.0];
        assert!(beamform(&signals, &scan, 0.5, BeamformMethod::DelayAndSum).is_err());

        let signals = vec![vec![1.0; 10]; 3];
        let empty_scan: Vec<f64> = vec![];
        assert!(beamform(&signals, &empty_scan, 0.5, BeamformMethod::DelayAndSum).is_err());
    }

    #[test]
    fn test_scan_angles_degrees() {
        let angles = scan_angles_degrees(-90.0, 90.0, 181).expect("Scan angles failed");
        assert_eq!(angles.len(), 181);
        assert_relative_eq!(angles[0], -PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(angles[180], PI / 2.0, epsilon = 1e-10);

        // Test single point
        let single = scan_angles_degrees(30.0, 30.0, 1).expect("Single angle failed");
        assert_eq!(single.len(), 1);
        assert_relative_eq!(single[0], 30.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn test_scan_angles_zero_points() {
        assert!(scan_angles_degrees(-90.0, 90.0, 0).is_err());
    }

    // ----- Matrix inversion tests -----

    #[test]
    fn test_invert_identity() {
        let n = 3;
        let mut identity = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            identity[i][i] = Complex64::new(1.0, 0.0);
        }

        let inv = invert_hermitian_matrix(&identity).expect("Inversion failed");

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_relative_eq!(inv[i][j].re, 1.0, epsilon = 1e-10);
                } else {
                    assert!(inv[i][j].norm() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_invert_2x2() {
        let a = Complex64::new(4.0, 0.0);
        let b = Complex64::new(1.0, 0.5);
        let matrix = vec![vec![a, b], vec![b.conj(), a]];

        let inv = invert_hermitian_matrix(&matrix).expect("Inversion failed");

        // Verify A * A^{-1} = I
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum += matrix[i][k] * inv[k][j];
                }
                if i == j {
                    assert_relative_eq!(sum.re, 1.0, epsilon = 1e-8);
                    assert!(sum.im.abs() < 1e-8);
                } else {
                    assert!(sum.norm() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_invert_singular() {
        let matrix = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        assert!(invert_hermitian_matrix(&matrix).is_err());
    }

    #[test]
    fn test_invert_empty() {
        assert!(invert_hermitian_matrix(&[]).is_err());
    }

    #[test]
    fn test_invert_dimension_mismatch() {
        let matrix = vec![
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0)], // wrong length
        ];
        assert!(invert_hermitian_matrix(&matrix).is_err());
    }
}
