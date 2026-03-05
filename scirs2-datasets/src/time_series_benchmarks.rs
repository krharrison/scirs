//! Standard time series benchmark dataset generators.
//!
//! This module provides competition-grade synthetic time series generators:
//!
//! - [`m4_generate`]               – M4-competition-style univariate series.
//! - [`ett_generate`]              – ETT (Electricity Transformer Temperature) style
//!                                   multi-channel series.
//! - [`generate_multivariate_ts`]  – Correlated multivariate time series with
//!                                   a user-supplied correlation matrix.
//! - [`anomaly_injection`]         – Inject point, collective, or level-shift anomalies
//!                                   into an existing series.
//! - [`add_seasonality`]           – Superpose an additive sinusoidal seasonal component.
//!
//! All generators are fully deterministic given a seed.

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Sample `n` independent N(0,1) values.
fn standard_normals(n: usize, rng: &mut StdRng) -> Result<Vec<f64>> {
    let dist = scirs2_core::random::Normal::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })?;
    Ok((0..n).map(|_| dist.sample(rng)).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend type
// ─────────────────────────────────────────────────────────────────────────────

/// Trend component type for M4-style series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendType {
    /// No deterministic trend.
    None,
    /// Linear trend: `slope * t`.
    Linear,
    /// Exponential growth: `exp(rate * t)`.
    Exponential,
    /// Damped linear trend that flattens after `length / 3` steps.
    Damped,
}

// ─────────────────────────────────────────────────────────────────────────────
// Anomaly type
// ─────────────────────────────────────────────────────────────────────────────

/// Anomaly type for [`anomaly_injection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyKind {
    /// Point anomaly: single spike at the specified position.
    Point,
    /// Collective anomaly: a contiguous run of elevated values starting at the
    /// position, lasting `duration` steps.
    Collective {
        /// Length of the collective anomaly run in time steps.
        duration: usize,
    },
    /// Level-shift anomaly: the series mean shifts by the magnitude from the
    /// position to the end of the series.
    LevelShift,
}

// ─────────────────────────────────────────────────────────────────────────────
// m4_generate
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an M4-competition-style univariate time series.
///
/// The series combines:
/// 1. A deterministic **trend** component controlled by `trend_type`.
/// 2. A **seasonal** component with amplitude `seasonality * sin(2π t / seasonality_period)`.
/// 3. An **AR(1) noise** process: `u[t] = 0.7 * u[t-1] + ε[t]`, where `ε[t] ~ N(0, noise_level²)`.
///
/// The result is `trend[t] + seasonal[t] + noise_ar1[t]`.
///
/// # Arguments
///
/// * `length`             – Number of time steps (must be ≥ 1).
/// * `seasonality_period` – Period of the seasonal sinusoid in steps (must be ≥ 1).
/// * `trend_type`         – Type of deterministic trend to include.
/// * `noise_level`        – Standard deviation of the i.i.d. driving noise (≥ 0).
/// * `seed`               – Random seed.
///
/// # Returns
///
/// `Array1<f64>` of length `length`.
///
/// # Errors
///
/// Returns an error if `length == 0`, `seasonality_period == 0`, or `noise_level < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::time_series_benchmarks::{m4_generate, TrendType};
///
/// let ts = m4_generate(120, 12, TrendType::Linear, 1.0, 42).expect("m4 failed");
/// assert_eq!(ts.len(), 120);
/// ```
pub fn m4_generate(
    length: usize,
    seasonality_period: usize,
    trend_type: TrendType,
    noise_level: f64,
    seed: u64,
) -> Result<Array1<f64>> {
    if length == 0 {
        return Err(DatasetsError::InvalidFormat(
            "m4_generate: length must be >= 1".to_string(),
        ));
    }
    if seasonality_period == 0 {
        return Err(DatasetsError::InvalidFormat(
            "m4_generate: seasonality_period must be >= 1".to_string(),
        ));
    }
    if noise_level < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "m4_generate: noise_level must be >= 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let eps_vec = if noise_level > 0.0 {
        let dist = scirs2_core::random::Normal::new(0.0_f64, noise_level).map_err(|e| {
            DatasetsError::ComputationError(format!("Normal dist creation failed: {e}"))
        })?;
        (0..length).map(|_| dist.sample(&mut rng)).collect::<Vec<f64>>()
    } else {
        vec![0.0_f64; length]
    };

    // AR(1) noise: u[t] = 0.7 * u[t-1] + eps[t]
    let ar_coef = 0.7_f64;
    let mut ar_noise = vec![0.0_f64; length];
    if !eps_vec.is_empty() {
        ar_noise[0] = eps_vec[0];
    }
    for t in 1..length {
        ar_noise[t] = ar_coef * ar_noise[t - 1] + eps_vec[t];
    }

    let mut out = Array1::zeros(length);
    let period_f = seasonality_period as f64;

    for t in 0..length {
        let t_f = t as f64;

        // Trend component
        let trend_val = match trend_type {
            TrendType::None => 0.0,
            TrendType::Linear => 0.05 * t_f,
            TrendType::Exponential => {
                // rate = 0.005 → moderate growth
                (0.005 * t_f).exp() - 1.0
            }
            TrendType::Damped => {
                let pivot = length as f64 / 3.0;
                if t_f < pivot {
                    0.1 * t_f
                } else {
                    0.1 * pivot + 0.02 * (t_f - pivot)
                }
            }
        };

        // Seasonal component (amplitude scaled to ~1 unit)
        let seasonal_val = (2.0 * PI * t_f / period_f).sin();

        out[t] = trend_val + seasonal_val + ar_noise[t];
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// ett_generate
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an ETT (Electricity Transformer Temperature) style multivariate series.
///
/// The ETT dataset is a benchmark for long-range multivariate time series
/// forecasting.  This generator produces 7 synthetic channels modelled after
/// the original ETT columns:
///
/// | Channel | Description           | Base period (steps) |
/// |---------|----------------------|---------------------|
/// | 0       | Load (HUFL)          | 24 (daily)          |
/// | 1       | Load (HULL)          | 24                  |
/// | 2       | Load (MUFL)          | 24 × 7 (weekly)     |
/// | 3       | Load (MULL)          | 24 × 7              |
/// | 4       | Load (LUFL)          | 24 × 30 (monthly)   |
/// | 5       | Load (LULL)          | 24 × 30             |
/// | 6       | Oil Temperature (OT) | 24                  |
///
/// Each channel uses `m4_generate` with channel-specific parameters.
///
/// # Arguments
///
/// * `length`      – Number of time steps (must be ≥ 1).
/// * `noise_level` – Noise std-dev applied to all channels (≥ 0).
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of shape `(length, 7)`.
///
/// # Errors
///
/// Returns an error if `length == 0` or `noise_level < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::time_series_benchmarks::ett_generate;
///
/// let mat = ett_generate(200, 0.5, 42).expect("ett failed");
/// assert_eq!(mat.nrows(), 200);
/// assert_eq!(mat.ncols(), 7);
/// ```
pub fn ett_generate(length: usize, noise_level: f64, seed: u64) -> Result<Array2<f64>> {
    if length == 0 {
        return Err(DatasetsError::InvalidFormat(
            "ett_generate: length must be >= 1".to_string(),
        ));
    }
    if noise_level < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "ett_generate: noise_level must be >= 0".to_string(),
        ));
    }

    // Channel specifications: (seasonality_period, trend_type, noise_scale)
    let channel_specs: [(usize, TrendType, f64); 7] = [
        (24, TrendType::Linear, 1.0),
        (24, TrendType::Damped, 0.8),
        (24 * 7, TrendType::Linear, 1.2),
        (24 * 7, TrendType::None, 0.9),
        (24 * 30, TrendType::Exponential, 1.1),
        (24 * 30, TrendType::None, 1.0),
        (24, TrendType::Linear, 0.5),
    ];

    let n_channels = channel_specs.len();
    let mut out = Array2::zeros((length, n_channels));

    for (ch, &(period, trend, noise_scale)) in channel_specs.iter().enumerate() {
        let ch_seed = seed.wrapping_add(ch as u64 * 1_000_003);
        let ch_noise = noise_level * noise_scale;
        // Clamp period to length if length < period
        let eff_period = period.min(length).max(1);
        let ts = m4_generate(length, eff_period, trend, ch_noise, ch_seed)?;
        for t in 0..length {
            out[[t, ch]] = ts[t];
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_multivariate_ts
// ─────────────────────────────────────────────────────────────────────────────

/// Generate correlated multivariate time series using a user-supplied correlation matrix.
///
/// The process:
/// 1. For each series generate independent AR(1) noise: `u[t] = 0.5 * u[t-1] + e[t]`,
///    `e[t] ~ N(0, 1)`.
/// 2. Apply the Cholesky factor of the correlation matrix to introduce cross-series
///    correlations.
/// 3. Each series also receives a distinct sinusoidal seasonal component with period
///    `length / (i + 1)` (clipped to reasonable bounds).
///
/// The correlation matrix must be:
/// - Square of size `n_series × n_series`.
/// - Symmetric.
/// - Positive semi-definite.
///
/// A simple validity check (positive diagonal) is performed; full PSD verification
/// is the caller's responsibility.
///
/// # Arguments
///
/// * `n_series`           – Number of time series (must be ≥ 1).
/// * `length`             – Number of time steps (must be ≥ 1).
/// * `correlation_matrix` – `(n_series, n_series)` correlation matrix.
/// * `seed`               – Random seed.
///
/// # Returns
///
/// `Array2<f64>` of shape `(length, n_series)`.
///
/// # Errors
///
/// Returns an error if dimensions are inconsistent or the diagonal is non-positive.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::time_series_benchmarks::generate_multivariate_ts;
/// use scirs2_core::ndarray::array;
///
/// let corr = array![[1.0, 0.6], [0.6, 1.0]];
/// let ts = generate_multivariate_ts(2, 100, &corr, 42).expect("mvts failed");
/// assert_eq!(ts.shape(), &[100, 2]);
/// ```
pub fn generate_multivariate_ts(
    n_series: usize,
    length: usize,
    correlation_matrix: &Array2<f64>,
    seed: u64,
) -> Result<Array2<f64>> {
    if n_series == 0 {
        return Err(DatasetsError::InvalidFormat(
            "generate_multivariate_ts: n_series must be >= 1".to_string(),
        ));
    }
    if length == 0 {
        return Err(DatasetsError::InvalidFormat(
            "generate_multivariate_ts: length must be >= 1".to_string(),
        ));
    }
    let corr_shape = correlation_matrix.shape();
    if corr_shape[0] != n_series || corr_shape[1] != n_series {
        return Err(DatasetsError::InvalidFormat(format!(
            "generate_multivariate_ts: correlation_matrix must be ({n_series}, {n_series}), \
             got ({}, {})",
            corr_shape[0], corr_shape[1]
        )));
    }
    // Basic diagonal check
    for i in 0..n_series {
        if correlation_matrix[[i, i]] <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "generate_multivariate_ts: correlation_matrix diagonal element [{i},{i}] \
                 must be positive, got {}",
                correlation_matrix[[i, i]]
            )));
        }
    }

    // Cholesky decomposition (lower-triangular L such that L L^T = Sigma)
    // Using the standard outer-product algorithm.
    let chol = cholesky_lower(correlation_matrix, n_series)?;

    let mut rng = make_rng(seed);
    let raw_noise = standard_normals(n_series * length, &mut rng)?;

    // Independent AR(1) processes (shape: length × n_series)
    let ar_coef = 0.5_f64;
    let mut ar_mat = vec![0.0_f64; length * n_series];
    // t=0
    for s in 0..n_series {
        ar_mat[0 * n_series + s] = raw_noise[s];
    }
    for t in 1..length {
        for s in 0..n_series {
            ar_mat[t * n_series + s] =
                ar_coef * ar_mat[(t - 1) * n_series + s] + raw_noise[t * n_series + s];
        }
    }

    // Apply Cholesky: correlated[t, :] = L @ ar_mat[t, :]
    let mut out = Array2::zeros((length, n_series));
    for t in 0..length {
        for s in 0..n_series {
            let mut val = 0.0_f64;
            for k in 0..=s {
                val += chol[s * n_series + k] * ar_mat[t * n_series + k];
            }
            // Add a per-series seasonal component (period proportional to index)
            let period_f = ((length as f64) / ((s + 1) as f64)).max(2.0);
            let seasonal = 0.3 * (2.0 * PI * t as f64 / period_f).sin();
            out[[t, s]] = val + seasonal;
        }
    }

    Ok(out)
}

/// Lower-triangular Cholesky decomposition of a symmetric PSD matrix.
/// Returns a flat row-major `n×n` vector of the lower triangle `L` s.t. `L L^T = A`.
fn cholesky_lower(a: &Array2<f64>, n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag_sq = a[[i, i]] - sum;
                if diag_sq < 0.0 {
                    return Err(DatasetsError::ComputationError(format!(
                        "cholesky_lower: matrix not PSD at diagonal [{i},{i}], \
                         got negative radicand {diag_sq}"
                    )));
                }
                l[i * n + j] = diag_sq.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < 1e-15 {
                    l[i * n + j] = 0.0;
                } else {
                    l[i * n + j] = (a[[i, j]] - sum) / ljj;
                }
            }
        }
    }
    Ok(l)
}

// ─────────────────────────────────────────────────────────────────────────────
// anomaly_injection
// ─────────────────────────────────────────────────────────────────────────────

/// Inject anomalies into a univariate time series.
///
/// Anomalies are applied in-place on a **copy** of `ts`.  Each position in
/// `positions` receives an anomaly of the corresponding `magnitude` (may be
/// negative for dips).
///
/// `anomaly_kind` controls the shape:
/// - [`AnomalyKind::Point`]       – Single-sample spike.
/// - [`AnomalyKind::Collective`]  – `duration`-sample plateau.
/// - [`AnomalyKind::LevelShift`]  – Permanent offset from `position` to end.
///
/// # Arguments
///
/// * `ts`           – Input time series (not modified; a copy is returned).
/// * `anomaly_kind` – Shape of each injected anomaly.
/// * `positions`    – Indices at which to inject anomalies (each must be < ts.len()).
/// * `magnitudes`   – Anomaly magnitudes (same length as `positions`).
///
/// # Returns
///
/// `Array1<f64>` — copy of `ts` with anomalies added.
///
/// # Errors
///
/// Returns an error if `positions.len() != magnitudes.len()` or any position is
/// out-of-bounds.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::time_series_benchmarks::{AnomalyKind, anomaly_injection};
/// use scirs2_core::ndarray::Array1;
///
/// let base = Array1::zeros(100);
/// let result = anomaly_injection(
///     &base,
///     AnomalyKind::Point,
///     &[20, 60],
///     &[5.0, -3.0],
/// ).expect("anomaly injection failed");
/// assert!((result[20] - 5.0).abs() < 1e-12);
/// assert!((result[60] - (-3.0)).abs() < 1e-12);
/// ```
pub fn anomaly_injection(
    ts: &Array1<f64>,
    anomaly_kind: AnomalyKind,
    positions: &[usize],
    magnitudes: &[f64],
) -> Result<Array1<f64>> {
    if positions.len() != magnitudes.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "anomaly_injection: positions.len() ({}) must equal magnitudes.len() ({})",
            positions.len(),
            magnitudes.len()
        )));
    }
    let n = ts.len();
    for (i, &pos) in positions.iter().enumerate() {
        if pos >= n {
            return Err(DatasetsError::InvalidFormat(format!(
                "anomaly_injection: positions[{i}]={pos} is out-of-bounds for ts of length {n}"
            )));
        }
    }

    let mut out = ts.to_owned();

    for (&pos, &mag) in positions.iter().zip(magnitudes.iter()) {
        match anomaly_kind {
            AnomalyKind::Point => {
                out[pos] += mag;
            }
            AnomalyKind::Collective { duration } => {
                let end = (pos + duration).min(n);
                for t in pos..end {
                    out[t] += mag;
                }
            }
            AnomalyKind::LevelShift => {
                for t in pos..n {
                    out[t] += mag;
                }
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// add_seasonality
// ─────────────────────────────────────────────────────────────────────────────

/// Add a sinusoidal seasonal component to a time series.
///
/// The seasonal pattern `amplitude * sin(2π t / period)` is added element-wise
/// to a **copy** of `ts`.
///
/// # Arguments
///
/// * `ts`        – Input time series.
/// * `period`    – Period of the seasonal component in steps (must be ≥ 1).
/// * `amplitude` – Amplitude of the sinusoidal component.
///
/// # Returns
///
/// `Array1<f64>` of the same length as `ts`.
///
/// # Errors
///
/// Returns an error if `period == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::time_series_benchmarks::add_seasonality;
/// use scirs2_core::ndarray::Array1;
///
/// let base = Array1::zeros(48);
/// let seasonal = add_seasonality(&base, 12, 2.0).expect("seasonality failed");
/// assert_eq!(seasonal.len(), 48);
/// // Season starts at 0 → sin(0) = 0
/// assert!((seasonal[0] - 0.0).abs() < 1e-12);
/// ```
pub fn add_seasonality(
    ts: &Array1<f64>,
    period: usize,
    amplitude: f64,
) -> Result<Array1<f64>> {
    if period == 0 {
        return Err(DatasetsError::InvalidFormat(
            "add_seasonality: period must be >= 1".to_string(),
        ));
    }

    let n = ts.len();
    let period_f = period as f64;
    let mut out = ts.to_owned();
    for t in 0..n {
        out[t] += amplitude * (2.0 * PI * t as f64 / period_f).sin();
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── m4_generate ──────────────────────────────────────────────────────────

    #[test]
    fn test_m4_length() {
        let ts = m4_generate(120, 12, TrendType::Linear, 1.0, 42).expect("m4 failed");
        assert_eq!(ts.len(), 120);
    }

    #[test]
    fn test_m4_no_trend() {
        let ts = m4_generate(60, 6, TrendType::None, 0.0, 1).expect("m4 no trend");
        assert_eq!(ts.len(), 60);
        // With no trend and no noise, values should be pure seasonality (bounded)
        for &v in ts.iter() {
            assert!(
                v.abs() <= 1.5,
                "no-trend zero-noise value {v} unexpectedly large"
            );
        }
    }

    #[test]
    fn test_m4_exponential_trend_grows() {
        let ts = m4_generate(100, 10, TrendType::Exponential, 0.0, 5).expect("m4 exp");
        // Last value should exceed first
        assert!(ts[99] > ts[0], "exponential trend should be increasing");
    }

    #[test]
    fn test_m4_damped_trend_decelerates() {
        let ts = m4_generate(90, 10, TrendType::Damped, 0.0, 3).expect("m4 damped");
        // The increments before the pivot should be larger than after.
        // Pivot is at length/3 = 30.
        let pre_delta = ts[29] - ts[0];
        let post_delta = ts[89] - ts[59];
        // With zero noise and flat seasonality through 3 full periods: damped post-slope is smaller.
        assert!(
            pre_delta >= post_delta,
            "damped: pre_delta ({pre_delta:.3}) should be >= post_delta ({post_delta:.3})"
        );
    }

    #[test]
    fn test_m4_determinism() {
        let ts1 = m4_generate(50, 5, TrendType::Linear, 1.0, 42).expect("ts1");
        let ts2 = m4_generate(50, 5, TrendType::Linear, 1.0, 42).expect("ts2");
        for (a, b) in ts1.iter().zip(ts2.iter()) {
            assert!((a - b).abs() < 1e-12, "m4 determinism failed");
        }
    }

    #[test]
    fn test_m4_different_seeds_differ() {
        let ts1 = m4_generate(50, 5, TrendType::Linear, 1.0, 42).expect("ts1");
        let ts2 = m4_generate(50, 5, TrendType::Linear, 1.0, 99).expect("ts2");
        let diff: f64 = ts1.iter().zip(ts2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.1, "different seeds should produce different series");
    }

    #[test]
    fn test_m4_error_length_zero() {
        assert!(m4_generate(0, 12, TrendType::None, 1.0, 1).is_err());
    }

    #[test]
    fn test_m4_error_period_zero() {
        assert!(m4_generate(100, 0, TrendType::None, 1.0, 1).is_err());
    }

    #[test]
    fn test_m4_error_negative_noise() {
        assert!(m4_generate(100, 12, TrendType::None, -0.5, 1).is_err());
    }

    // ── ett_generate ─────────────────────────────────────────────────────────

    #[test]
    fn test_ett_shape() {
        let mat = ett_generate(200, 0.5, 42).expect("ett failed");
        assert_eq!(mat.nrows(), 200);
        assert_eq!(mat.ncols(), 7);
    }

    #[test]
    fn test_ett_error_length_zero() {
        assert!(ett_generate(0, 0.5, 1).is_err());
    }

    #[test]
    fn test_ett_determinism() {
        let m1 = ett_generate(50, 0.3, 7).expect("m1");
        let m2 = ett_generate(50, 0.3, 7).expect("m2");
        for t in 0..50 {
            for ch in 0..7 {
                assert!((m1[[t, ch]] - m2[[t, ch]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_ett_channels_differ() {
        let mat = ett_generate(100, 0.1, 1).expect("ett channels");
        // Different channels should not be identical
        let mut found_different = false;
        'outer: for t in 0..100 {
            for ch in 1..7 {
                if (mat[[t, 0]] - mat[[t, ch]]).abs() > 0.01 {
                    found_different = true;
                    break 'outer;
                }
            }
        }
        assert!(found_different, "ETT channels should differ");
    }

    // ── generate_multivariate_ts ─────────────────────────────────────────────

    #[test]
    fn test_mvts_shape() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]];
        let ts = generate_multivariate_ts(2, 80, &corr, 42).expect("mvts shape");
        assert_eq!(ts.shape(), &[80, 2]);
    }

    #[test]
    fn test_mvts_identity_uncorrelated() {
        // Identity matrix → uncorrelated series
        let corr = array![[1.0, 0.0], [0.0, 1.0]];
        let ts = generate_multivariate_ts(2, 100, &corr, 42).expect("mvts identity");
        assert_eq!(ts.shape(), &[100, 2]);
    }

    #[test]
    fn test_mvts_three_series() {
        let corr = array![[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]];
        let ts = generate_multivariate_ts(3, 60, &corr, 7).expect("mvts 3");
        assert_eq!(ts.shape(), &[60, 3]);
    }

    #[test]
    fn test_mvts_error_wrong_corr_size() {
        let corr = array![[1.0, 0.5], [0.5, 1.0]]; // 2×2 but n_series=3
        assert!(generate_multivariate_ts(3, 50, &corr, 1).is_err());
    }

    #[test]
    fn test_mvts_error_zero_n_series() {
        let corr: Array2<f64> = Array2::zeros((0, 0));
        assert!(generate_multivariate_ts(0, 50, &corr, 1).is_err());
    }

    #[test]
    fn test_mvts_error_non_psd() {
        // Negative diagonal → should fail
        let corr = array![[1.0, 0.5], [0.5, -1.0]];
        assert!(generate_multivariate_ts(2, 50, &corr, 1).is_err());
    }

    #[test]
    fn test_mvts_determinism() {
        let corr = array![[1.0, 0.3], [0.3, 1.0]];
        let ts1 = generate_multivariate_ts(2, 50, &corr, 42).expect("ts1");
        let ts2 = generate_multivariate_ts(2, 50, &corr, 42).expect("ts2");
        for t in 0..50 {
            for s in 0..2 {
                assert!((ts1[[t, s]] - ts2[[t, s]]).abs() < 1e-12);
            }
        }
    }

    // ── anomaly_injection ────────────────────────────────────────────────────

    #[test]
    fn test_anomaly_point_spike() {
        let base = Array1::zeros(100);
        let result = anomaly_injection(&base, AnomalyKind::Point, &[20, 60], &[5.0, -3.0])
            .expect("point spike");
        assert!((result[20] - 5.0).abs() < 1e-12);
        assert!((result[60] - (-3.0)).abs() < 1e-12);
        // All other points should remain zero
        for t in 0..100 {
            if t != 20 && t != 60 {
                assert!(result[t].abs() < 1e-12, "non-spike position {t} should be zero");
            }
        }
    }

    #[test]
    fn test_anomaly_collective() {
        let base = Array1::zeros(100);
        let result = anomaly_injection(
            &base,
            AnomalyKind::Collective { duration: 5 },
            &[30],
            &[2.0],
        )
        .expect("collective");
        for t in 30..35 {
            assert!((result[t] - 2.0).abs() < 1e-12, "collective span t={t}");
        }
        for t in 0..30 {
            assert!(result[t].abs() < 1e-12, "before collective t={t}");
        }
        for t in 35..100 {
            assert!(result[t].abs() < 1e-12, "after collective t={t}");
        }
    }

    #[test]
    fn test_anomaly_level_shift() {
        let base = Array1::zeros(100);
        let result =
            anomaly_injection(&base, AnomalyKind::LevelShift, &[50], &[3.0]).expect("level shift");
        for t in 0..50 {
            assert!(result[t].abs() < 1e-12, "before shift t={t}");
        }
        for t in 50..100 {
            assert!((result[t] - 3.0).abs() < 1e-12, "after shift t={t}");
        }
    }

    #[test]
    fn test_anomaly_does_not_modify_input() {
        let base = Array1::from_vec(vec![1.0; 50]);
        let _ = anomaly_injection(&base, AnomalyKind::Point, &[5], &[10.0]).expect("copy");
        // base should be unchanged
        assert!(base.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_anomaly_error_length_mismatch() {
        let base = Array1::zeros(100);
        assert!(anomaly_injection(&base, AnomalyKind::Point, &[10, 20], &[1.0]).is_err());
    }

    #[test]
    fn test_anomaly_error_out_of_bounds() {
        let base = Array1::zeros(100);
        assert!(anomaly_injection(&base, AnomalyKind::Point, &[100], &[1.0]).is_err());
    }

    // ── add_seasonality ──────────────────────────────────────────────────────

    #[test]
    fn test_seasonality_length() {
        let base = Array1::zeros(48);
        let s = add_seasonality(&base, 12, 2.0).expect("seasonality");
        assert_eq!(s.len(), 48);
    }

    #[test]
    fn test_seasonality_values() {
        let base = Array1::zeros(48);
        let s = add_seasonality(&base, 12, 1.0).expect("seasonality values");
        // At t=0: sin(0) = 0
        assert!((s[0] - 0.0).abs() < 1e-12);
        // At t=3: sin(2π·3/12) = sin(π/2) = 1.0
        assert!((s[3] - 1.0).abs() < 1e-10, "s[3]={}", s[3]);
        // At t=6: sin(π) ≈ 0
        assert!(s[6].abs() < 1e-10, "s[6]={}", s[6]);
    }

    #[test]
    fn test_seasonality_amplitude() {
        let base = Array1::zeros(20);
        let s = add_seasonality(&base, 4, 3.0).expect("amplitude");
        let max_val = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_val - 3.0).abs() < 1e-9, "amplitude mismatch max={max_val}");
    }

    #[test]
    fn test_seasonality_does_not_modify_input() {
        let base = Array1::from_vec(vec![5.0; 20]);
        let _ = add_seasonality(&base, 4, 2.0).expect("copy");
        assert!(base.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn test_seasonality_error_period_zero() {
        let base = Array1::zeros(20);
        assert!(add_seasonality(&base, 0, 1.0).is_err());
    }
}
