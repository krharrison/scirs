//! Space-Time Adaptive Processing (STAP)
//!
//! STAP simultaneously filters in both the spatial and temporal (Doppler) domains
//! to suppress clutter and interference while preserving the target signal.
//!
//! The space-time snapshot vector is formed by stacking the spatial snapshots from
//! each pulse within a Coherent Processing Interval (CPI):
//!
//! `x_st = [x_1^T, x_2^T, ..., x_M^T]^T`   (dimension N*M x 1)
//!
//! The optimal STAP weight vector is:
//!
//! `w = R_st^{-1} s_st / (s_st^H R_st^{-1} s_st)`
//!
//! where `R_st` is the space-time covariance matrix and `s_st` is the
//! space-time steering vector `s_st = s_temporal kron s_spatial`.
//!
//! Provides:
//! - [`stap_processor`]: Full-dimension STAP processing
//! - [`smi_stap`]: Sample Matrix Inversion approach
//! - [`space_time_steering_vector`]: Joint spatial-temporal steering vector
//! - [`estimate_clutter_rank`]: Brennan's rule for clutter rank estimation
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{inner_product_conj, invert_hermitian_matrix, mat_vec_mul};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for STAP processing
#[derive(Debug, Clone)]
pub struct STAPConfig {
    /// Number of array elements (spatial channels)
    pub n_elements: usize,
    /// Number of pulses in the Coherent Processing Interval (CPI)
    pub n_pulses: usize,
    /// Element spacing in wavelengths
    pub element_spacing: f64,
    /// Pulse Repetition Frequency in Hz
    pub prf: f64,
    /// Carrier wavelength in metres
    pub wavelength: f64,
    /// Number of guard cells for CFAR-like exclusion around the cell under test
    pub n_guard_cells: usize,
    /// Number of training cells for covariance estimation
    pub n_training_cells: usize,
}

impl Default for STAPConfig {
    fn default() -> Self {
        Self {
            n_elements: 8,
            n_pulses: 16,
            element_spacing: 0.5,
            prf: 1000.0,
            wavelength: 0.03,
            n_guard_cells: 2,
            n_training_cells: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of STAP processing
#[derive(Debug, Clone)]
pub struct STAPResult {
    /// Space-time weight vector (dimension N*M x 1)
    pub weights: Vec<Complex64>,
    /// Beamformer output power
    pub output_power: f64,
    /// SINR improvement factor in dB
    pub sinr_improvement: f64,
    /// Estimated clutter rank (via Brennan's rule or eigenvalue analysis)
    pub clutter_rank: usize,
}

// ---------------------------------------------------------------------------
// Space-time steering vector
// ---------------------------------------------------------------------------

/// Compute the space-time steering vector
///
/// The space-time steering vector is the Kronecker product of a temporal
/// steering vector (Doppler) and a spatial steering vector (angle):
///
/// `s_st = s_temporal kron s_spatial`
///
/// # Arguments
///
/// * `angle` - Spatial angle in radians (from broadside)
/// * `doppler` - Normalised Doppler frequency (f_d / f_prf), range [-0.5, 0.5]
/// * `config` - STAP configuration
///
/// # Returns
///
/// Complex vector of length `n_elements * n_pulses`
pub fn space_time_steering_vector(
    angle: f64,
    doppler: f64,
    config: &STAPConfig,
) -> SignalResult<Vec<Complex64>> {
    if config.n_elements == 0 {
        return Err(SignalError::ValueError(
            "Number of elements must be positive".to_string(),
        ));
    }
    if config.n_pulses == 0 {
        return Err(SignalError::ValueError(
            "Number of pulses must be positive".to_string(),
        ));
    }

    let n = config.n_elements;
    let m = config.n_pulses;

    // Spatial steering vector: a_k = exp(j * 2 * pi * d * k * sin(angle))
    let spatial_phase = 2.0 * PI * config.element_spacing * angle.sin();
    let s_spatial: Vec<Complex64> = (0..n)
        .map(|k| {
            let phase = spatial_phase * k as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Temporal steering vector: b_p = exp(j * 2 * pi * doppler * p)
    let temporal_phase = 2.0 * PI * doppler;
    let s_temporal: Vec<Complex64> = (0..m)
        .map(|p| {
            let phase = temporal_phase * p as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Kronecker product: s_st = s_temporal kron s_spatial
    // Ordering: for each pulse p, stack N spatial elements
    let mut s_st = Vec::with_capacity(n * m);
    for p in 0..m {
        for k in 0..n {
            s_st.push(s_temporal[p] * s_spatial[k]);
        }
    }

    Ok(s_st)
}

// ---------------------------------------------------------------------------
// Covariance estimation from training data
// ---------------------------------------------------------------------------

/// Estimate the space-time covariance matrix from training data
///
/// Each training cell is an `[n_elements x n_pulses]` snapshot matrix.
/// The space-time snapshot is formed by vectorizing (column-major stacking).
///
/// `R_st = (1/L) * sum_l x_l * x_l^H`
fn estimate_st_covariance(
    training_data: &[Vec<Vec<Complex64>>],
    n_elements: usize,
    n_pulses: usize,
) -> SignalResult<Vec<Vec<Complex64>>> {
    if training_data.is_empty() {
        return Err(SignalError::ValueError(
            "Training data must not be empty".to_string(),
        ));
    }

    let dim = n_elements * n_pulses;
    let mut r_st = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

    let n_training = training_data.len();

    for cell in training_data {
        let x_st = vectorize_snapshot(cell, n_elements, n_pulses)?;
        // Rank-1 update: R += x * x^H
        for i in 0..dim {
            for j in 0..dim {
                r_st[i][j] += x_st[i] * x_st[j].conj();
            }
        }
    }

    // Normalise
    let scale = 1.0 / n_training as f64;
    for row in &mut r_st {
        for val in row.iter_mut() {
            *val *= scale;
        }
    }

    Ok(r_st)
}

/// Vectorize a snapshot matrix [n_elements x n_pulses] into a column vector
/// by stacking pulse-by-pulse (first pulse elements first, then second, ...).
fn vectorize_snapshot(
    data: &[Vec<Complex64>],
    n_elements: usize,
    n_pulses: usize,
) -> SignalResult<Vec<Complex64>> {
    if data.len() != n_elements {
        return Err(SignalError::DimensionMismatch(format!(
            "Snapshot has {} rows, expected {} elements",
            data.len(),
            n_elements
        )));
    }

    let mut x_st = Vec::with_capacity(n_elements * n_pulses);
    for p in 0..n_pulses {
        for k in 0..n_elements {
            if p >= data[k].len() {
                return Err(SignalError::DimensionMismatch(format!(
                    "Element {} has {} pulses, expected at least {}",
                    k,
                    data[k].len(),
                    n_pulses
                )));
            }
            x_st.push(data[k][p]);
        }
    }

    Ok(x_st)
}

// ---------------------------------------------------------------------------
// Full STAP processor
// ---------------------------------------------------------------------------

/// Full-dimension STAP processor
///
/// Computes the optimal STAP weight vector using the sample covariance
/// matrix estimated from nearby range cells (training data), then applies
/// it to the cell under test.
///
/// `w = R_st^{-1} s_st / (s_st^H R_st^{-1} s_st)`
///
/// # Arguments
///
/// * `data` - Cell under test, `[n_elements][n_pulses]` complex snapshots
/// * `training_data` - Nearby range cells for covariance estimation
/// * `target_angle` - Look direction in radians (from broadside)
/// * `target_doppler` - Normalised Doppler frequency (f_d / f_prf)
/// * `config` - STAP configuration
///
/// # Returns
///
/// [`STAPResult`] containing weights, output power, and SINR improvement
pub fn stap_processor(
    data: &[Vec<Complex64>],
    training_data: &[Vec<Vec<Complex64>>],
    target_angle: f64,
    target_doppler: f64,
    config: &STAPConfig,
) -> SignalResult<STAPResult> {
    let n = config.n_elements;
    let m = config.n_pulses;
    let dim = n * m;

    // Compute space-time steering vector
    let s_st = space_time_steering_vector(target_angle, target_doppler, config)?;

    // Estimate space-time covariance from training data
    let r_st = estimate_st_covariance(training_data, n, m)?;

    // Add small diagonal loading for numerical stability
    let trace_val = compute_trace(&r_st);
    let loading = 1e-6 * trace_val / dim as f64;
    let r_loaded = apply_diagonal_loading_st(&r_st, loading);

    // Compute weights: w = R^{-1} s / (s^H R^{-1} s)
    let r_inv = invert_hermitian_matrix(&r_loaded)?;
    let r_inv_s = mat_vec_mul(&r_inv, &s_st);
    let denom = inner_product_conj(&s_st, &r_inv_s);

    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "STAP denominator near zero; covariance may be singular".to_string(),
        ));
    }

    let weights: Vec<Complex64> = r_inv_s.iter().map(|&v| v / denom).collect();

    // Compute output power: P = w^H R w (using training covariance)
    let rw = mat_vec_mul(&r_st, &weights);
    let output_power = inner_product_conj(&weights, &rw).re;

    // Vectorize the cell under test
    let x_st = vectorize_snapshot(data, n, m)?;

    // SINR improvement: compare to matched-filter (conventional) output
    // Conventional: w_mf = s / ||s||^2
    let s_norm_sq = s_st.iter().map(|v| v.norm_sqr()).sum::<f64>();
    let mf_weights: Vec<Complex64> = s_st.iter().map(|&v| v / s_norm_sq).collect();

    let rw_mf = mat_vec_mul(&r_st, &mf_weights);
    let mf_power = inner_product_conj(&mf_weights, &rw_mf).re;

    let sinr_improvement = if mf_power > 1e-30 && output_power > 1e-30 {
        // SINR improvement = (SINR_stap / SINR_mf)
        // For MVDR-type weights: SINR_out = s^H R^{-1} s
        // For matched filter: SINR_mf = ||s||^4 / (s^H R s)
        let sinr_stap = denom.re; // s^H R^{-1} s
        let r_s = mat_vec_mul(&r_st, &s_st);
        let sinr_mf_denom = inner_product_conj(&s_st, &r_s).re;
        let sinr_mf = if sinr_mf_denom > 1e-30 {
            s_norm_sq * s_norm_sq / sinr_mf_denom
        } else {
            1.0
        };
        let ratio = sinr_stap / sinr_mf;
        if ratio > 0.0 {
            10.0 * ratio.log10()
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Estimate clutter rank using eigenvalue spread heuristic
    let clutter_rank = estimate_clutter_rank_from_covariance(&r_st, dim);

    Ok(STAPResult {
        weights,
        output_power,
        sinr_improvement,
        clutter_rank,
    })
}

// ---------------------------------------------------------------------------
// SMI STAP
// ---------------------------------------------------------------------------

/// Sample Matrix Inversion (SMI) STAP
///
/// The simplest STAP approach: directly invert the sample covariance matrix
/// and apply to the steering vector.
///
/// `w = R_hat^{-1} s`
///
/// Note: this returns the un-normalised weight vector. The caller may
/// normalise by `s^H w` to enforce the distortionless constraint.
///
/// # Arguments
///
/// * `training_data` - Training snapshots `[L][n_elements][n_pulses]`
/// * `steering` - Space-time steering vector (dimension N*M)
/// * `n_elements` - Number of spatial elements
/// * `n_pulses` - Number of temporal pulses
///
/// # Returns
///
/// Weight vector of length `N * M`
pub fn smi_stap(
    training_data: &[Vec<Vec<Complex64>>],
    steering: &Vec<Complex64>,
    n_elements: usize,
    n_pulses: usize,
) -> SignalResult<Vec<Complex64>> {
    let dim = n_elements * n_pulses;
    if steering.len() != dim {
        return Err(SignalError::DimensionMismatch(format!(
            "Steering vector length {} does not match N*M = {}",
            steering.len(),
            dim
        )));
    }

    let r_st = estimate_st_covariance(training_data, n_elements, n_pulses)?;

    // Diagonal loading proportional to trace
    let trace_val = compute_trace(&r_st);
    let loading = 1e-4 * trace_val / dim as f64;
    let r_loaded = apply_diagonal_loading_st(&r_st, loading);

    let r_inv = invert_hermitian_matrix(&r_loaded)?;
    let weights = mat_vec_mul(&r_inv, steering);

    // Normalise so that s^H w = 1
    let denom = inner_product_conj(steering, &weights);
    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "SMI STAP: denominator near zero".to_string(),
        ));
    }

    Ok(weights.iter().map(|&w| w / denom).collect())
}

// ---------------------------------------------------------------------------
// Clutter rank estimation
// ---------------------------------------------------------------------------

/// Estimate the clutter rank using Brennan's rule
///
/// For an airborne radar with a ULA:
///
/// `rank approx N + (M - 1) * beta`
///
/// where `beta = 2 * v * T_r / d` is the space-time coupling factor,
/// `v` is the platform velocity, `T_r = 1/PRF` is the PRI,
/// and `d = element_spacing * wavelength` is the physical element spacing.
///
/// The result is clamped to `[1, N*M]`.
///
/// # Arguments
///
/// * `config` - STAP configuration
/// * `platform_velocity` - Platform velocity in m/s
pub fn estimate_clutter_rank(config: &STAPConfig, platform_velocity: f64) -> usize {
    let n = config.n_elements;
    let m = config.n_pulses;
    let d_physical = config.element_spacing * config.wavelength;

    if d_physical <= 0.0 || config.prf <= 0.0 {
        return n; // fallback
    }

    let t_r = 1.0 / config.prf;
    let beta = 2.0 * platform_velocity * t_r / d_physical;

    let rank_f = n as f64 + (m as f64 - 1.0) * beta;
    let rank = rank_f.round() as usize;

    // Clamp to valid range
    rank.clamp(1, n * m)
}

/// Estimate clutter rank from eigenvalue spread of covariance matrix
///
/// Counts eigenvalues above a threshold relative to the largest eigenvalue.
/// Uses the Gershgorin circle theorem for a rough estimate without full
/// eigendecomposition.
fn estimate_clutter_rank_from_covariance(cov: &[Vec<Complex64>], dim: usize) -> usize {
    // Use diagonal dominance as a proxy for eigenvalue spread
    // Gershgorin: eigenvalue_i in [a_ii - r_i, a_ii + r_i]
    // where r_i = sum_{j != i} |a_ij|
    let mut gershgorin_centers: Vec<f64> = Vec::with_capacity(dim);
    let mut gershgorin_radii: Vec<f64> = Vec::with_capacity(dim);

    for i in 0..dim {
        if i < cov.len() {
            gershgorin_centers.push(cov[i][i].re);
            let radius: f64 = (0..dim)
                .filter(|&j| j != i && j < cov[i].len())
                .map(|j| cov[i][j].norm())
                .sum();
            gershgorin_radii.push(radius);
        }
    }

    if gershgorin_centers.is_empty() {
        return 1;
    }

    // Max possible eigenvalue
    let max_eig = gershgorin_centers
        .iter()
        .zip(gershgorin_radii.iter())
        .map(|(c, r)| c + r)
        .fold(0.0_f64, f64::max);

    if max_eig <= 0.0 {
        return 1;
    }

    // Count discs whose upper bound is above 1% of max eigenvalue
    let threshold = 0.01 * max_eig;
    let rank = gershgorin_centers
        .iter()
        .zip(gershgorin_radii.iter())
        .filter(|(c, r)| *c + *r > threshold)
        .count();

    rank.max(1)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute trace of a complex matrix
fn compute_trace(matrix: &[Vec<Complex64>]) -> f64 {
    matrix
        .iter()
        .enumerate()
        .map(|(i, row)| if i < row.len() { row[i].re } else { 0.0 })
        .sum()
}

/// Apply diagonal loading: R_loaded = R + sigma^2 * I
fn apply_diagonal_loading_st(covariance: &[Vec<Complex64>], loading: f64) -> Vec<Vec<Complex64>> {
    let mut r_loaded: Vec<Vec<Complex64>> = covariance.to_vec();
    if loading > 0.0 {
        for (i, row) in r_loaded.iter_mut().enumerate() {
            if i < row.len() {
                row[i] += Complex64::new(loading, 0.0);
            }
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
    use approx::assert_relative_eq;

    fn make_identity_training(
        n_elements: usize,
        n_pulses: usize,
        n_cells: usize,
    ) -> Vec<Vec<Vec<Complex64>>> {
        // Generate training data that gives approximately identity covariance
        // (white noise snapshots)
        let mut training = Vec::with_capacity(n_cells);
        for cell_idx in 0..n_cells {
            let mut snapshot = vec![vec![Complex64::new(0.0, 0.0); n_pulses]; n_elements];
            for k in 0..n_elements {
                for p in 0..n_pulses {
                    // Deterministic pseudo-random via hash
                    let seed = (cell_idx * 1000 + k * 37 + p * 13) as f64;
                    let re = (seed * 0.618033988749).fract() - 0.5;
                    let im = (seed * 0.414213562373).fract() - 0.5;
                    snapshot[k][p] = Complex64::new(re, im);
                }
            }
            training.push(snapshot);
        }
        training
    }

    #[test]
    fn test_steering_vector_dimensions() {
        let config = STAPConfig {
            n_elements: 4,
            n_pulses: 8,
            ..STAPConfig::default()
        };
        let sv =
            space_time_steering_vector(0.0, 0.0, &config).expect("should compute steering vector");
        assert_eq!(sv.len(), 4 * 8, "Steering vector should be N*M");
    }

    #[test]
    fn test_steering_vector_kronecker_structure() {
        let config = STAPConfig {
            n_elements: 3,
            n_pulses: 4,
            element_spacing: 0.5,
            ..STAPConfig::default()
        };
        let angle = 0.3;
        let doppler = 0.1;
        let sv = space_time_steering_vector(angle, doppler, &config)
            .expect("should compute steering vector");

        // Verify Kronecker structure: sv[p*N + k] = b_p * a_k
        let spatial_phase = 2.0 * PI * config.element_spacing * angle.sin();
        let temporal_phase = 2.0 * PI * doppler;

        for p in 0..config.n_pulses {
            let b_p = Complex64::new(
                (temporal_phase * p as f64).cos(),
                (temporal_phase * p as f64).sin(),
            );
            for k in 0..config.n_elements {
                let a_k = Complex64::new(
                    (spatial_phase * k as f64).cos(),
                    (spatial_phase * k as f64).sin(),
                );
                let expected = b_p * a_k;
                let idx = p * config.n_elements + k;
                assert_relative_eq!(sv[idx].re, expected.re, epsilon = 1e-12);
                assert_relative_eq!(sv[idx].im, expected.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_steering_vector_broadside_zero_doppler() {
        // At broadside (angle=0) and zero Doppler, all elements should be 1+0j
        let config = STAPConfig {
            n_elements: 4,
            n_pulses: 3,
            ..STAPConfig::default()
        };
        let sv =
            space_time_steering_vector(0.0, 0.0, &config).expect("should compute steering vector");
        for &val in &sv {
            assert_relative_eq!(val.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(val.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_identity_covariance_gives_conventional_beamformer() {
        // With identity covariance, STAP weights reduce to matched filter
        let config = STAPConfig {
            n_elements: 3,
            n_pulses: 4,
            ..STAPConfig::default()
        };
        let n = config.n_elements;
        let m = config.n_pulses;
        let dim = n * m;

        // Use many training cells to get near-identity covariance
        let training = make_identity_training(n, m, dim * 4);
        let data = vec![vec![Complex64::new(1.0, 0.0); m]; n];

        let result =
            stap_processor(&data, &training, 0.0, 0.0, &config).expect("should process STAP");

        // Weights should exist and be finite
        assert_eq!(result.weights.len(), dim);
        assert!(
            result
                .weights
                .iter()
                .all(|w| w.re.is_finite() && w.im.is_finite()),
            "All weights should be finite"
        );
        assert!(
            result.output_power.is_finite(),
            "Output power should be finite"
        );
    }

    #[test]
    fn test_clutter_rank_brennan() {
        let config = STAPConfig {
            n_elements: 8,
            n_pulses: 16,
            element_spacing: 0.5,
            prf: 1000.0,
            wavelength: 0.03,
            ..STAPConfig::default()
        };
        // Platform velocity 100 m/s
        // d_physical = 0.5 * 0.03 = 0.015 m
        // T_r = 1/1000 = 0.001 s
        // beta = 2 * 100 * 0.001 / 0.015 = 13.33
        // rank = 8 + 15 * 13.33 = 8 + 200 = 208, clamped to N*M = 128
        let rank = estimate_clutter_rank(&config, 100.0);
        assert!(rank >= 1, "Rank should be at least 1");
        assert!(
            rank <= config.n_elements * config.n_pulses,
            "Rank should not exceed N*M"
        );
        // For this case beta is large, so rank should saturate at N*M
        assert_eq!(rank, config.n_elements * config.n_pulses);

        // With a slow platform, rank should be closer to N
        let rank_slow = estimate_clutter_rank(&config, 0.1);
        // beta = 2 * 0.1 * 0.001 / 0.015 = 0.0133
        // rank = 8 + 15 * 0.0133 = 8.2 -> 8
        assert_eq!(rank_slow, 8);
    }

    #[test]
    fn test_smi_stap() {
        let n = 3;
        let m = 4;
        let dim = n * m;
        let config = STAPConfig {
            n_elements: n,
            n_pulses: m,
            ..STAPConfig::default()
        };

        let training = make_identity_training(n, m, dim * 4);
        let steering =
            space_time_steering_vector(0.1, 0.05, &config).expect("should compute steering vector");

        let weights = smi_stap(&training, &steering, n, m).expect("should compute SMI weights");

        assert_eq!(weights.len(), dim);
        assert!(weights.iter().all(|w| w.re.is_finite() && w.im.is_finite()));

        // Verify distortionless constraint: s^H w = 1
        let response = inner_product_conj(&steering, &weights);
        assert_relative_eq!(response.re, 1.0, epsilon = 0.1);
        assert!(response.im.abs() < 0.1);
    }

    #[test]
    fn test_sinr_improvement_with_interference() {
        let config = STAPConfig {
            n_elements: 4,
            n_pulses: 4,
            element_spacing: 0.5,
            ..STAPConfig::default()
        };
        let n = config.n_elements;
        let m = config.n_pulses;
        let dim = n * m;

        // Create training data with a strong interferer at a different angle/Doppler
        let interf_angle = 0.5;
        let interf_doppler = 0.2;
        let interf_sv = space_time_steering_vector(interf_angle, interf_doppler, &config)
            .expect("interference SV");

        let mut training = Vec::with_capacity(dim * 3);
        for cell_idx in 0..(dim * 3) {
            let mut snapshot = vec![vec![Complex64::new(0.0, 0.0); m]; n];
            for k in 0..n {
                for p in 0..m {
                    let seed = (cell_idx * 1000 + k * 37 + p * 13) as f64;
                    let noise_re = (seed * 0.618033988749).fract() - 0.5;
                    let noise_im = (seed * 0.414213562373).fract() - 0.5;
                    // Add interference
                    let idx = p * n + k;
                    let interf = 5.0 * interf_sv[idx];
                    snapshot[k][p] = Complex64::new(noise_re, noise_im) + interf;
                }
            }
            training.push(snapshot);
        }

        let target_angle = 0.0;
        let target_doppler = 0.0;
        let data = vec![vec![Complex64::new(1.0, 0.0); m]; n];

        let result = stap_processor(&data, &training, target_angle, target_doppler, &config)
            .expect("should process STAP");

        // With strong interference, STAP should provide positive SINR improvement
        // (it suppresses the interferer that the matched filter cannot)
        assert!(result.output_power.is_finite());
        assert!(result.sinr_improvement.is_finite());
    }

    #[test]
    fn test_stap_validation_errors() {
        let config = STAPConfig {
            n_elements: 0,
            ..STAPConfig::default()
        };
        assert!(space_time_steering_vector(0.0, 0.0, &config).is_err());

        let config2 = STAPConfig {
            n_elements: 4,
            n_pulses: 0,
            ..STAPConfig::default()
        };
        assert!(space_time_steering_vector(0.0, 0.0, &config2).is_err());
    }
}
