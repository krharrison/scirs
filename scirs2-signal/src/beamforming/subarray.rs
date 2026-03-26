//! Reduced-dimension STAP via subarray processing
//!
//! Full-dimension STAP requires inverting an `N*M x N*M` covariance matrix,
//! which is computationally expensive and requires many training samples
//! (the Reed-Mallet-Brennan rule requires `L >= 2*N*M` i.i.d. samples).
//!
//! Subarray-based methods reduce the dimensionality by partitioning the
//! spatial and/or temporal domains into overlapping subarrays/sub-pulses,
//! then processing within each subarray before combining.
//!
//! Provides:
//! - [`jdl_stap`]: Joint Domain Localized (JDL) processing
//! - [`factored_stap`]: Factored (separable) STAP
//! - [`dimension_reduction_ratio`]: Compare full vs reduced dimensions
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{inner_product_conj, invert_hermitian_matrix, mat_vec_mul};
use crate::beamforming::stap::{space_time_steering_vector, STAPConfig, STAPResult};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for subarray-based reduced-dimension STAP
#[derive(Debug, Clone)]
pub struct SubarrayConfig {
    /// Number of spatial subarrays
    pub spatial_subarrays: usize,
    /// Number of Doppler taps (temporal sub-pulses) per subarray
    pub temporal_taps: usize,
    /// Overlap between adjacent spatial subarrays (in elements)
    pub overlap: usize,
}

impl Default for SubarrayConfig {
    fn default() -> Self {
        Self {
            spatial_subarrays: 4,
            temporal_taps: 4,
            overlap: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// JDL STAP
// ---------------------------------------------------------------------------

/// Joint Domain Localized (JDL) STAP processing
///
/// JDL pre-filters the data using spatial beamforming (DFT beams) and
/// temporal filtering (Doppler bins), then applies adaptive processing
/// in a reduced-dimension space. This dramatically reduces the number
/// of degrees of freedom and the required training data.
///
/// The processing flow is:
/// 1. Form spatial beams (DFT across elements for each pulse)
/// 2. Form Doppler bins (DFT across pulses for each beam)
/// 3. Select a local processing region around the target angle/Doppler
/// 4. Estimate covariance in the reduced domain
/// 5. Apply adaptive weights in the reduced domain
///
/// # Arguments
///
/// * `data` - Cell under test `[n_elements][n_pulses]`
/// * `training_data` - Training cells `[L][n_elements][n_pulses]`
/// * `target_angle` - Look direction in radians
/// * `target_doppler` - Normalised Doppler frequency
/// * `stap_config` - STAP configuration
/// * `subarray_config` - Subarray configuration
pub fn jdl_stap(
    data: &[Vec<Complex64>],
    training_data: &[Vec<Vec<Complex64>>],
    target_angle: f64,
    target_doppler: f64,
    stap_config: &STAPConfig,
    subarray_config: &SubarrayConfig,
) -> SignalResult<STAPResult> {
    let n = stap_config.n_elements;
    let m = stap_config.n_pulses;
    let n_sub = subarray_config.spatial_subarrays;
    let n_taps = subarray_config.temporal_taps;

    validate_subarray_config(n, m, subarray_config)?;

    // Reduced dimension
    let reduced_dim = n_sub * n_taps;

    // Step 1-2: Transform data to beam-Doppler domain (reduced)
    let x_reduced =
        transform_to_beam_doppler(data, n, m, target_angle, target_doppler, subarray_config)?;

    // Transform training data
    let mut training_reduced = Vec::with_capacity(training_data.len());
    for cell in training_data {
        let t_reduced =
            transform_to_beam_doppler(cell, n, m, target_angle, target_doppler, subarray_config)?;
        training_reduced.push(t_reduced);
    }

    // Step 3: Estimate covariance in reduced domain
    let mut r_reduced = vec![vec![Complex64::new(0.0, 0.0); reduced_dim]; reduced_dim];
    let n_training = training_reduced.len();
    if n_training == 0 {
        return Err(SignalError::ValueError(
            "Training data must not be empty".to_string(),
        ));
    }

    for t_vec in &training_reduced {
        for i in 0..reduced_dim {
            for j in 0..reduced_dim {
                r_reduced[i][j] += t_vec[i] * t_vec[j].conj();
            }
        }
    }

    let scale = 1.0 / n_training as f64;
    for row in &mut r_reduced {
        for val in row.iter_mut() {
            *val *= scale;
        }
    }

    // Diagonal loading
    let trace_val: f64 = (0..reduced_dim).map(|i| r_reduced[i][i].re).sum();
    let loading = 1e-4 * trace_val / reduced_dim as f64;
    for i in 0..reduced_dim {
        r_reduced[i][i] += Complex64::new(loading, 0.0);
    }

    // Step 4: Steering vector in reduced domain
    let s_reduced =
        reduced_steering_vector(target_angle, target_doppler, subarray_config, stap_config)?;

    // Step 5: Compute weights in reduced domain
    let r_inv = invert_hermitian_matrix(&r_reduced)?;
    let r_inv_s = mat_vec_mul(&r_inv, &s_reduced);
    let denom = inner_product_conj(&s_reduced, &r_inv_s);

    if denom.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "JDL STAP: denominator near zero".to_string(),
        ));
    }

    let w_reduced: Vec<Complex64> = r_inv_s.iter().map(|&v| v / denom).collect();

    // Output power in reduced domain
    let rw = mat_vec_mul(&r_reduced, &w_reduced);
    let output_power = inner_product_conj(&w_reduced, &rw).re;

    // Map back to full-dimension weights for the result
    let full_weights = expand_weights_to_full(
        &w_reduced,
        n,
        m,
        target_angle,
        target_doppler,
        subarray_config,
    )?;

    // SINR improvement estimate
    let sinr_improvement = if denom.re > 1.0 {
        10.0 * denom.re.log10()
    } else {
        0.0
    };

    Ok(STAPResult {
        weights: full_weights,
        output_power: output_power.max(0.0),
        sinr_improvement,
        clutter_rank: reduced_dim.min(n * m),
    })
}

// ---------------------------------------------------------------------------
// Factored STAP
// ---------------------------------------------------------------------------

/// Factored STAP: separate spatial and temporal processing
///
/// Decomposes the joint space-time problem into two independent stages:
/// 1. Spatial beamforming (per pulse)
/// 2. Temporal (Doppler) filtering (per spatial beam output)
///
/// The overall weight vector approximates the Kronecker product:
/// `w approx w_temporal kron w_spatial`
///
/// This reduces the N*M-dimensional problem to an N-dimensional spatial
/// problem and an M-dimensional temporal problem.
///
/// # Arguments
///
/// * `data` - Cell under test `[n_elements][n_pulses]`
/// * `training_data` - Training cells `[L][n_elements][n_pulses]`
/// * `target_angle` - Look direction in radians
/// * `target_doppler` - Normalised Doppler frequency
/// * `config` - STAP configuration
pub fn factored_stap(
    data: &[Vec<Complex64>],
    training_data: &[Vec<Vec<Complex64>>],
    target_angle: f64,
    target_doppler: f64,
    config: &STAPConfig,
) -> SignalResult<STAPResult> {
    let n = config.n_elements;
    let m = config.n_pulses;

    if data.len() != n {
        return Err(SignalError::DimensionMismatch(format!(
            "Data has {} elements, expected {}",
            data.len(),
            n
        )));
    }
    if training_data.is_empty() {
        return Err(SignalError::ValueError(
            "Training data must not be empty".to_string(),
        ));
    }

    // Stage 1: Spatial processing
    // Estimate spatial covariance from training data (average over pulses)
    let mut r_spatial = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let mut spatial_count = 0usize;

    for cell in training_data {
        if cell.len() != n {
            continue;
        }
        for p in 0..m {
            // Extract spatial snapshot for pulse p
            let mut x_spatial = Vec::with_capacity(n);
            let mut valid = true;
            for k in 0..n {
                if p < cell[k].len() {
                    x_spatial.push(cell[k][p]);
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            for i in 0..n {
                for j in 0..n {
                    r_spatial[i][j] += x_spatial[i] * x_spatial[j].conj();
                }
            }
            spatial_count += 1;
        }
    }

    if spatial_count == 0 {
        return Err(SignalError::ComputationError(
            "No valid spatial snapshots in training data".to_string(),
        ));
    }

    let spatial_scale = 1.0 / spatial_count as f64;
    for row in &mut r_spatial {
        for val in row.iter_mut() {
            *val *= spatial_scale;
        }
    }

    // Spatial loading
    let trace_s: f64 = (0..n).map(|i| r_spatial[i][i].re).sum();
    let load_s = 1e-4 * trace_s / n as f64;
    for i in 0..n {
        r_spatial[i][i] += Complex64::new(load_s, 0.0);
    }

    // Spatial steering vector
    let spatial_phase = 2.0 * PI * config.element_spacing * target_angle.sin();
    let s_spatial: Vec<Complex64> = (0..n)
        .map(|k| {
            let phase = spatial_phase * k as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Spatial weights via MVDR
    let r_s_inv = invert_hermitian_matrix(&r_spatial)?;
    let r_s_inv_a = mat_vec_mul(&r_s_inv, &s_spatial);
    let denom_s = inner_product_conj(&s_spatial, &r_s_inv_a);

    if denom_s.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "Factored STAP: spatial denominator near zero".to_string(),
        ));
    }

    let w_spatial: Vec<Complex64> = r_s_inv_a.iter().map(|&v| v / denom_s).collect();

    // Stage 2: Temporal processing
    // Apply spatial weights to get beam output for each pulse, then estimate
    // temporal covariance

    let mut r_temporal = vec![vec![Complex64::new(0.0, 0.0); m]; m];
    let mut temporal_count = 0usize;

    for cell in training_data {
        if cell.len() != n {
            continue;
        }
        // Form beam output: y[p] = w_spatial^H * x[:,p]
        let mut y = Vec::with_capacity(m);
        let mut valid = true;
        for p in 0..m {
            let mut beam_out = Complex64::new(0.0, 0.0);
            for k in 0..n {
                if p < cell[k].len() {
                    beam_out += w_spatial[k].conj() * cell[k][p];
                } else {
                    valid = false;
                    break;
                }
            }
            if !valid {
                break;
            }
            y.push(beam_out);
        }
        if !valid || y.len() != m {
            continue;
        }

        for i in 0..m {
            for j in 0..m {
                r_temporal[i][j] += y[i] * y[j].conj();
            }
        }
        temporal_count += 1;
    }

    if temporal_count == 0 {
        return Err(SignalError::ComputationError(
            "No valid temporal snapshots in training data".to_string(),
        ));
    }

    let temporal_scale = 1.0 / temporal_count as f64;
    for row in &mut r_temporal {
        for val in row.iter_mut() {
            *val *= temporal_scale;
        }
    }

    // Temporal loading
    let trace_t: f64 = (0..m).map(|i| r_temporal[i][i].re).sum();
    let load_t = 1e-4 * trace_t / m as f64;
    for i in 0..m {
        r_temporal[i][i] += Complex64::new(load_t, 0.0);
    }

    // Temporal steering vector (Doppler)
    let temporal_phase = 2.0 * PI * target_doppler;
    let s_temporal: Vec<Complex64> = (0..m)
        .map(|p| {
            let phase = temporal_phase * p as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Temporal weights via MVDR
    let r_t_inv = invert_hermitian_matrix(&r_temporal)?;
    let r_t_inv_b = mat_vec_mul(&r_t_inv, &s_temporal);
    let denom_t = inner_product_conj(&s_temporal, &r_t_inv_b);

    if denom_t.norm() < 1e-20 {
        return Err(SignalError::ComputationError(
            "Factored STAP: temporal denominator near zero".to_string(),
        ));
    }

    let w_temporal: Vec<Complex64> = r_t_inv_b.iter().map(|&v| v / denom_t).collect();

    // Combine: w_st = w_temporal kron w_spatial
    let mut weights = Vec::with_capacity(n * m);
    for p in 0..m {
        for k in 0..n {
            weights.push(w_temporal[p] * w_spatial[k]);
        }
    }

    // Output power: apply combined weights to cell under test
    // x_st vectorised, then P = |w^H x|^2
    let mut x_st = Vec::with_capacity(n * m);
    for p in 0..m {
        for k in 0..n {
            if p < data[k].len() {
                x_st.push(data[k][p]);
            } else {
                x_st.push(Complex64::new(0.0, 0.0));
            }
        }
    }

    let beam_output = inner_product_conj(&weights, &x_st);
    let output_power = beam_output.norm_sqr();

    // SINR improvement estimate
    let sinr_s = denom_s.re;
    let sinr_t = denom_t.re;
    let sinr_improvement = if sinr_s > 0.0 && sinr_t > 0.0 {
        10.0 * (sinr_s * sinr_t).log10()
    } else {
        0.0
    };

    Ok(STAPResult {
        weights,
        output_power,
        sinr_improvement,
        clutter_rank: n + m, // approximate for factored approach
    })
}

// ---------------------------------------------------------------------------
// Dimension reduction ratio
// ---------------------------------------------------------------------------

/// Compute the dimension reduction ratio: full STAP dimension / reduced dimension
///
/// Full dimension: `N * M` (n_elements * n_pulses)
/// Reduced dimension: `spatial_subarrays * temporal_taps`
///
/// A ratio > 1 indicates dimension reduction.
pub fn dimension_reduction_ratio(stap: &STAPConfig, subarray: &SubarrayConfig) -> f64 {
    let full_dim = stap.n_elements * stap.n_pulses;
    let reduced_dim = subarray.spatial_subarrays * subarray.temporal_taps;

    if reduced_dim == 0 {
        return f64::INFINITY;
    }

    full_dim as f64 / reduced_dim as f64
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate subarray configuration against STAP configuration
fn validate_subarray_config(
    n_elements: usize,
    n_pulses: usize,
    config: &SubarrayConfig,
) -> SignalResult<()> {
    if config.spatial_subarrays == 0 {
        return Err(SignalError::ValueError(
            "Number of spatial subarrays must be positive".to_string(),
        ));
    }
    if config.temporal_taps == 0 {
        return Err(SignalError::ValueError(
            "Number of temporal taps must be positive".to_string(),
        ));
    }
    if config.spatial_subarrays > n_elements {
        return Err(SignalError::ValueError(format!(
            "Number of spatial subarrays ({}) exceeds number of elements ({})",
            config.spatial_subarrays, n_elements
        )));
    }
    if config.temporal_taps > n_pulses {
        return Err(SignalError::ValueError(format!(
            "Number of temporal taps ({}) exceeds number of pulses ({})",
            config.temporal_taps, n_pulses
        )));
    }
    Ok(())
}

/// Transform data from element-pulse domain to beam-Doppler domain (reduced)
///
/// Applies spatial DFT beams and temporal DFT bins, then selects a
/// local region around the target angle/Doppler.
fn transform_to_beam_doppler(
    data: &[Vec<Complex64>],
    n_elements: usize,
    n_pulses: usize,
    target_angle: f64,
    target_doppler: f64,
    config: &SubarrayConfig,
) -> SignalResult<Vec<Complex64>> {
    let n_sub = config.spatial_subarrays;
    let n_taps = config.temporal_taps;
    let reduced_dim = n_sub * n_taps;

    if data.len() != n_elements {
        return Err(SignalError::DimensionMismatch(format!(
            "Data has {} elements, expected {}",
            data.len(),
            n_elements
        )));
    }

    // Compute spatial DFT beams
    // Beam b = sum_k x_k * exp(-j*2*pi*b*k/N) for b = 0..N-1
    // Select n_sub beams centred around the target spatial frequency
    // Normalised spatial frequency u = d * sin(angle), DFT bin = u * N (mod N)
    let target_spatial_freq = target_angle.sin() * 0.5;
    let centre_beam = (target_spatial_freq * n_elements as f64)
        .round()
        .rem_euclid(n_elements as f64) as i64;

    let mut beams = Vec::with_capacity(n_sub * n_pulses);
    for sub_idx in 0..n_sub {
        let beam_idx_raw = centre_beam - (n_sub as i64 / 2) + sub_idx as i64;
        let beam_idx = beam_idx_raw.rem_euclid(n_elements as i64) as usize;

        for p in 0..n_pulses {
            let mut beam_val = Complex64::new(0.0, 0.0);
            for k in 0..n_elements {
                if p < data[k].len() {
                    let phase = -2.0 * PI * (beam_idx as f64) * (k as f64) / (n_elements as f64);
                    let twiddle = Complex64::new(phase.cos(), phase.sin());
                    beam_val += data[k][p] * twiddle;
                }
            }
            beams.push(beam_val);
        }
    }

    // Compute temporal DFT bins for each beam
    // Select n_taps Doppler bins centred around the target Doppler
    // DFT Doppler bin = f_d * M (mod M)
    let centre_doppler = (target_doppler * n_pulses as f64)
        .round()
        .rem_euclid(n_pulses as f64) as i64;

    let mut result = Vec::with_capacity(reduced_dim);
    for sub_idx in 0..n_sub {
        for tap_idx in 0..n_taps {
            let dop_idx_raw = centre_doppler - (n_taps as i64 / 2) + tap_idx as i64;
            let dop_idx = dop_idx_raw.rem_euclid(n_pulses as i64) as usize;

            let mut bin_val = Complex64::new(0.0, 0.0);
            for p in 0..n_pulses {
                let beam_sample = beams[sub_idx * n_pulses + p];
                let phase = -2.0 * PI * (dop_idx as f64) * (p as f64) / (n_pulses as f64);
                let twiddle = Complex64::new(phase.cos(), phase.sin());
                bin_val += beam_sample * twiddle;
            }
            result.push(bin_val);
        }
    }

    Ok(result)
}

/// Compute steering vector in the reduced beam-Doppler domain
fn reduced_steering_vector(
    target_angle: f64,
    target_doppler: f64,
    subarray_config: &SubarrayConfig,
    stap_config: &STAPConfig,
) -> SignalResult<Vec<Complex64>> {
    let n_sub = subarray_config.spatial_subarrays;
    let n_taps = subarray_config.temporal_taps;
    let n = stap_config.n_elements;
    let m = stap_config.n_pulses;
    let reduced_dim = n_sub * n_taps;

    let target_spatial_freq = target_angle.sin() * 0.5;
    let centre_beam = (target_spatial_freq * n as f64)
        .round()
        .rem_euclid(n as f64) as i64;
    let centre_doppler = (target_doppler * m as f64).round().rem_euclid(m as f64) as i64;

    // The steering vector in the DFT domain is a set of delta-like responses
    // at the target angle/Doppler. For the local region we use the DFT of
    // the original steering vector.
    let spatial_phase = 2.0 * PI * stap_config.element_spacing * target_angle.sin();
    let temporal_phase = 2.0 * PI * target_doppler;

    let mut s_reduced = Vec::with_capacity(reduced_dim);
    for sub_idx in 0..n_sub {
        let beam_idx_raw = centre_beam - (n_sub as i64 / 2) + sub_idx as i64;
        let beam_idx = beam_idx_raw.rem_euclid(n as i64) as usize;

        // Spatial DFT of steering vector at this beam index
        let mut s_beam = Complex64::new(0.0, 0.0);
        for k in 0..n {
            let sv_phase = spatial_phase * k as f64;
            let sv_k = Complex64::new(sv_phase.cos(), sv_phase.sin());
            let dft_phase = -2.0 * PI * (beam_idx as f64) * (k as f64) / (n as f64);
            let twiddle = Complex64::new(dft_phase.cos(), dft_phase.sin());
            s_beam += sv_k * twiddle;
        }

        for tap_idx in 0..n_taps {
            let dop_idx_raw = centre_doppler - (n_taps as i64 / 2) + tap_idx as i64;
            let dop_idx = dop_idx_raw.rem_euclid(m as i64) as usize;

            // Temporal DFT of steering vector at this Doppler bin
            let mut s_dop = Complex64::new(0.0, 0.0);
            for p in 0..m {
                let sv_phase_t = temporal_phase * p as f64;
                let sv_p = Complex64::new(sv_phase_t.cos(), sv_phase_t.sin());
                let dft_phase_t = -2.0 * PI * (dop_idx as f64) * (p as f64) / (m as f64);
                let twiddle_t = Complex64::new(dft_phase_t.cos(), dft_phase_t.sin());
                s_dop += sv_p * twiddle_t;
            }

            s_reduced.push(s_beam * s_dop);
        }
    }

    // Ensure the steering vector has non-trivial energy.
    // Normalise so that ||s||^2 = reduced_dim (similar to full-dim convention).
    let norm_sq: f64 = s_reduced.iter().map(|v| v.norm_sqr()).sum();
    if norm_sq > 1e-30 {
        let scale = (reduced_dim as f64 / norm_sq).sqrt();
        for v in &mut s_reduced {
            *v *= scale;
        }
    } else {
        // Fallback: uniform steering vector when DFT gives near-zero
        let val = Complex64::new(1.0, 0.0);
        for v in &mut s_reduced {
            *v = val;
        }
    }

    Ok(s_reduced)
}

/// Expand reduced-dimension weights back to full N*M dimension
fn expand_weights_to_full(
    w_reduced: &[Complex64],
    n_elements: usize,
    n_pulses: usize,
    target_angle: f64,
    target_doppler: f64,
    config: &SubarrayConfig,
) -> SignalResult<Vec<Complex64>> {
    let n_sub = config.spatial_subarrays;
    let n_taps = config.temporal_taps;
    let full_dim = n_elements * n_pulses;

    let target_spatial_freq = target_angle.sin() * 0.5;
    let centre_beam = (target_spatial_freq * n_elements as f64)
        .round()
        .rem_euclid(n_elements as f64) as i64;
    let centre_doppler = (target_doppler * n_pulses as f64)
        .round()
        .rem_euclid(n_pulses as f64) as i64;

    // Inverse DFT: map from beam-Doppler back to element-pulse domain
    let mut w_full = vec![Complex64::new(0.0, 0.0); full_dim];

    for sub_idx in 0..n_sub {
        let beam_idx_raw = centre_beam - (n_sub as i64 / 2) + sub_idx as i64;
        let beam_idx = beam_idx_raw.rem_euclid(n_elements as i64) as usize;

        for tap_idx in 0..n_taps {
            let dop_idx_raw = centre_doppler - (n_taps as i64 / 2) + tap_idx as i64;
            let dop_idx = dop_idx_raw.rem_euclid(n_pulses as i64) as usize;

            let w_val = w_reduced[sub_idx * n_taps + tap_idx];

            // Inverse spatial DFT
            for k in 0..n_elements {
                let phase_s = 2.0 * PI * (beam_idx as f64) * (k as f64) / (n_elements as f64);
                let twiddle_s = Complex64::new(phase_s.cos(), phase_s.sin());

                // Inverse temporal DFT
                for p in 0..n_pulses {
                    let phase_t = 2.0 * PI * (dop_idx as f64) * (p as f64) / (n_pulses as f64);
                    let twiddle_t = Complex64::new(phase_t.cos(), phase_t.sin());

                    let idx = p * n_elements + k;
                    w_full[idx] += w_val * twiddle_s * twiddle_t;
                }
            }
        }
    }

    // Normalise
    let norm_factor = 1.0 / ((n_elements * n_pulses) as f64);
    for w in &mut w_full {
        *w *= norm_factor;
    }

    Ok(w_full)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_training_data(
        n_elements: usize,
        n_pulses: usize,
        n_cells: usize,
    ) -> Vec<Vec<Vec<Complex64>>> {
        let mut training = Vec::with_capacity(n_cells);
        for cell_idx in 0..n_cells {
            let mut snapshot = vec![vec![Complex64::new(0.0, 0.0); n_pulses]; n_elements];
            for k in 0..n_elements {
                for p in 0..n_pulses {
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
    fn test_jdl_dimension_less_than_full() {
        let stap_config = STAPConfig {
            n_elements: 8,
            n_pulses: 8,
            ..STAPConfig::default()
        };
        let sub_config = SubarrayConfig {
            spatial_subarrays: 3,
            temporal_taps: 3,
            overlap: 1,
        };

        let full_dim = stap_config.n_elements * stap_config.n_pulses; // 64
        let reduced_dim = sub_config.spatial_subarrays * sub_config.temporal_taps; // 9

        assert!(
            reduced_dim < full_dim,
            "Reduced dim {} should be < full dim {}",
            reduced_dim,
            full_dim
        );

        let ratio = dimension_reduction_ratio(&stap_config, &sub_config);
        assert!(ratio > 1.0, "Ratio should be > 1, got {}", ratio);
        assert!((ratio - (64.0 / 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_jdl_stap_basic() {
        let stap_config = STAPConfig {
            n_elements: 4,
            n_pulses: 4,
            ..STAPConfig::default()
        };
        let sub_config = SubarrayConfig {
            spatial_subarrays: 2,
            temporal_taps: 2,
            overlap: 0,
        };

        let n = stap_config.n_elements;
        let m = stap_config.n_pulses;
        let training = make_training_data(n, m, 30);
        let data = vec![vec![Complex64::new(1.0, 0.0); m]; n];

        let result = jdl_stap(&data, &training, 0.0, 0.0, &stap_config, &sub_config)
            .expect("JDL STAP should succeed");

        assert_eq!(result.weights.len(), n * m);
        assert!(
            result.output_power >= 0.0,
            "Output power should be non-negative"
        );
        assert!(result.output_power.is_finite());
    }

    #[test]
    fn test_factored_stap_output_power_positive() {
        let config = STAPConfig {
            n_elements: 4,
            n_pulses: 4,
            element_spacing: 0.5,
            ..STAPConfig::default()
        };

        let n = config.n_elements;
        let m = config.n_pulses;
        let training = make_training_data(n, m, 30);
        let data = vec![vec![Complex64::new(1.0, 0.0); m]; n];

        let result = factored_stap(&data, &training, 0.1, 0.05, &config)
            .expect("Factored STAP should succeed");

        assert_eq!(result.weights.len(), n * m);
        assert!(
            result.output_power > 0.0,
            "Output power should be positive, got {}",
            result.output_power
        );
        assert!(result.output_power.is_finite());
    }

    #[test]
    fn test_dimension_reduction_ratio_correct() {
        let stap = STAPConfig {
            n_elements: 8,
            n_pulses: 16,
            ..STAPConfig::default()
        };
        let sub = SubarrayConfig {
            spatial_subarrays: 4,
            temporal_taps: 4,
            overlap: 1,
        };

        let ratio = dimension_reduction_ratio(&stap, &sub);
        // full = 128, reduced = 16, ratio = 8.0
        assert!((ratio - 8.0).abs() < 1e-10, "Expected 8.0, got {}", ratio);
    }

    #[test]
    fn test_subarray_validation() {
        assert!(validate_subarray_config(
            4,
            4,
            &SubarrayConfig {
                spatial_subarrays: 0,
                temporal_taps: 2,
                overlap: 0,
            }
        )
        .is_err());

        assert!(validate_subarray_config(
            4,
            4,
            &SubarrayConfig {
                spatial_subarrays: 2,
                temporal_taps: 0,
                overlap: 0,
            }
        )
        .is_err());

        assert!(validate_subarray_config(
            4,
            4,
            &SubarrayConfig {
                spatial_subarrays: 5,
                temporal_taps: 2,
                overlap: 0,
            }
        )
        .is_err());
    }

    #[test]
    fn test_factored_stap_validation() {
        let config = STAPConfig {
            n_elements: 4,
            n_pulses: 4,
            ..STAPConfig::default()
        };

        // Empty training data
        let data = vec![vec![Complex64::new(1.0, 0.0); 4]; 4];
        let empty_training: Vec<Vec<Vec<Complex64>>> = vec![];
        assert!(factored_stap(&data, &empty_training, 0.0, 0.0, &config).is_err());

        // Wrong data dimensions
        let wrong_data = vec![vec![Complex64::new(1.0, 0.0); 4]; 2]; // 2 elements, expected 4
        let training = make_training_data(4, 4, 10);
        assert!(factored_stap(&wrong_data, &training, 0.0, 0.0, &config).is_err());
    }
}
