//! Beamforming algorithms for array signal processing
//!
//! This module provides comprehensive beamforming and Direction-of-Arrival (DOA)
//! estimation algorithms for sensor arrays:
//!
//! | Module | Algorithm | Description |
//! |--------|-----------|-------------|
//! | `array` | Array geometry | ULA, UCA, arbitrary arrays, steering vectors |
//! | [`delay_and_sum`] | Delay-and-Sum | Conventional beamformer with uniform weights |
//! | [`mvdr`] | MVDR/Capon | Minimum variance distortionless response |
//! | [`music`] | MUSIC | DOA via noise subspace pseudo-spectrum |
//! | [`esprit`] | ESPRIT | DOA via rotational invariance (shift structure) |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_signal::beamforming::{
//!     array::{UniformLinearArray, ArrayGeometry, scan_angles_degrees},
//!     delay_and_sum::delay_and_sum_spectrum,
//! };
//!
//! // Create an 8-element ULA with half-wavelength spacing
//! let ula = UniformLinearArray::new(8, 0.5).expect("ULA creation should succeed");
//!
//! // Generate some signals (8 channels, 200 samples each)
//! let signals: Vec<Vec<f64>> = (0..8)
//!     .map(|_| (0..200).map(|k| (0.1 * k as f64).sin()).collect())
//!     .collect();
//!
//! // Scan angles from -60 to 60 degrees
//! let scan = scan_angles_degrees(-60.0, 60.0, 121).expect("angle creation should succeed");
//!
//! // Compute delay-and-sum spatial spectrum
//! let result = delay_and_sum_spectrum(&signals, &scan, 0.5).expect("spectrum should succeed");
//! assert_eq!(result.power_spectrum.len(), 121);
//! ```
//!
//! Pure Rust, no unwrap(), snake_case naming.

pub mod array;
pub mod delay_and_sum;
pub mod diagonal_loading;
pub mod esprit;
pub mod music;
pub mod mvdr;
pub mod stap;
pub mod subarray;

// Re-export common types
pub use array::{
    estimate_covariance, estimate_covariance_real, scan_angles_degrees, steering_vector_ula,
    steering_vectors_ula, ArbitraryArray, ArrayGeometry, ArrayManifoldData, UniformCircularArray,
    UniformLinearArray,
};

pub use delay_and_sum::{
    delay_and_sum_beam_pattern, delay_and_sum_filter, delay_and_sum_frequency_domain,
    delay_and_sum_power, delay_and_sum_spectrum, delay_and_sum_weights, DelayAndSumResult,
};

pub use mvdr::{mvdr_filter, mvdr_power, mvdr_spectrum, mvdr_weights, MVDRResult};

pub use music::{
    estimate_num_sources, MUSICConfig, MUSICDOAResult, MUSICEstimator, RootMUSIC, RootMUSICResult,
    SourceNumberEstimate,
};

pub use esprit::{ESPRITResult, TlsESPRIT, ESPRIT};

/// Beamforming method for spatial spectrum scanning (convenience enum)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BeamformMethod {
    /// Delay-and-Sum (conventional)
    DelayAndSum,
    /// MVDR (Capon) with diagonal loading
    Mvdr(f64),
}

/// Convenience function: scan spatial spectrum using the selected method
///
/// # Arguments
///
/// * `signals` - Multi-channel time-domain signals (n_elements x n_samples)
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
) -> crate::error::SignalResult<Vec<f64>> {
    use crate::error::SignalError;

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
        let power = match method {
            BeamformMethod::DelayAndSum => delay_and_sum_power(&cov, &sv)?,
            BeamformMethod::Mvdr(loading) => mvdr_power(&cov, &sv, loading)?,
        };
        power_spectrum.push(power);
    }

    Ok(power_spectrum)
}
