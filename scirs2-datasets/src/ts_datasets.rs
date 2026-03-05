//! Time series dataset generators.
//!
//! This module provides synthetic time series generators for testing
//! time series analysis, forecasting, and signal processing algorithms.
//!
//! # Generators
//!
//! - [`make_arma`] - ARMA(p,q) process
//! - [`make_lorenz`] - Lorenz chaotic attractor (3-D trajectory)
//! - [`make_van_der_pol`] - Van der Pol nonlinear oscillator
//! - [`make_seasonal`] - Seasonal / trend / noise decomposition series
//! - [`make_changepoint_series`] - Piecewise-constant series with changepoints

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a seeded [`StdRng`] from a `u64` seed.
fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Sample N(0,1) values into a `Vec<f64>` of length `n`.
fn standard_normals(n: usize, rng: &mut StdRng) -> Result<Vec<f64>> {
    let dist = scirs2_core::random::Normal::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })?;
    Ok((0..n).map(|_| dist.sample(rng)).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// make_arma
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a stationary ARMA(p, q) time series.
///
/// The process is defined by:
///
/// ```text
/// y[t] = ar[0]*y[t-1] + ... + ar[p-1]*y[t-p]
///       + noise[t]
///       + ma[0]*noise[t-1] + ... + ma[q-1]*noise[t-q]
/// ```
///
/// where `noise[t] ~ N(0, 1)`.
///
/// # Arguments
///
/// * `n`         – Number of time steps to return (must be > 0).
/// * `ar_params` – AR coefficients `[ar_1, …, ar_p]`.
/// * `ma_params` – MA coefficients `[ma_1, …, ma_q]`.
/// * `seed`      – Random seed for reproducibility.
///
/// # Returns
///
/// `Array1<f64>` of length `n`.
///
/// # Errors
///
/// Returns an error if `n == 0` or the distribution cannot be created.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::ts_datasets::make_arma;
///
/// let ar = [0.5, -0.25];
/// let ma = [0.3];
/// let y = make_arma(200, &ar, &ma, 42).expect("make_arma failed");
/// assert_eq!(y.len(), 200);
/// ```
pub fn make_arma(n: usize, ar_params: &[f64], ma_params: &[f64], seed: u64) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_arma: n must be > 0".to_string(),
        ));
    }

    let p = ar_params.len();
    let q = ma_params.len();
    let burn_in = (p.max(q) + 50).max(200); // warm-up samples
    let total = burn_in + n;

    let mut rng = make_rng(seed);
    let noise_raw = standard_normals(total, &mut rng)?;

    let mut y = vec![0.0_f64; total];
    let mut eps = vec![0.0_f64; total]; // noise history

    for t in 0..total {
        eps[t] = noise_raw[t];
        let mut val = eps[t];
        // AR component
        for (i, &coef) in ar_params.iter().enumerate() {
            if t > i {
                val += coef * y[t - 1 - i];
            }
        }
        // MA component
        for (i, &coef) in ma_params.iter().enumerate() {
            if t > i {
                val += coef * eps[t - 1 - i];
            }
        }
        y[t] = val;
    }

    // Return only the post-burn-in portion
    let result: Vec<f64> = y[burn_in..].to_vec();
    Ok(Array1::from_vec(result))
}

// ─────────────────────────────────────────────────────────────────────────────
// make_lorenz
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Lorenz attractor trajectory using 4th-order Runge-Kutta.
///
/// The Lorenz system:
/// ```text
/// dx/dt = σ (y − x)
/// dy/dt = x (ρ − z) − y
/// dz/dt = x y − β z
/// ```
///
/// # Arguments
///
/// * `n`     – Number of time steps (must be > 0).
/// * `dt`    – Integration step size (must be > 0).
/// * `sigma` – Lorenz σ parameter (classic: 10.0).
/// * `rho`   – Lorenz ρ parameter (classic: 28.0).
/// * `beta`  – Lorenz β parameter (classic: 8/3 ≈ 2.667).
///
/// # Returns
///
/// `Array2<f64>` of shape `(n, 3)` — columns are x, y, z.
///
/// # Errors
///
/// Returns an error if `n == 0` or `dt <= 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::ts_datasets::make_lorenz;
///
/// let traj = make_lorenz(1000, 0.01, 10.0, 28.0, 8.0 / 3.0).expect("make_lorenz failed");
/// assert_eq!(traj.nrows(), 1000);
/// assert_eq!(traj.ncols(), 3);
/// ```
pub fn make_lorenz(
    n: usize,
    dt: f64,
    sigma: f64,
    rho: f64,
    beta: f64,
) -> Result<Array2<f64>> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_lorenz: n must be > 0".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_lorenz: dt must be > 0".to_string(),
        ));
    }

    let lorenz_deriv = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    };

    // Classic initial conditions (slightly off the origin)
    let mut x = 1.0_f64;
    let mut y = 1.0_f64;
    let mut z = 1.0_f64;

    // Warm-up for 500 steps to land on the attractor
    for _ in 0..500 {
        let (k1x, k1y, k1z) = lorenz_deriv(x, y, z);
        let (k2x, k2y, k2z) = lorenz_deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
        let (k3x, k3y, k3z) = lorenz_deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
        let (k4x, k4y, k4z) = lorenz_deriv(x + dt * k3x, y + dt * k3y, z + dt * k3z);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }

    let mut out = Array2::zeros((n, 3));
    for i in 0..n {
        out[[i, 0]] = x;
        out[[i, 1]] = y;
        out[[i, 2]] = z;

        let (k1x, k1y, k1z) = lorenz_deriv(x, y, z);
        let (k2x, k2y, k2z) = lorenz_deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
        let (k3x, k3y, k3z) = lorenz_deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
        let (k4x, k4y, k4z) = lorenz_deriv(x + dt * k3x, y + dt * k3y, z + dt * k3z);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
        z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_van_der_pol
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Van der Pol oscillator trajectory using 4th-order Runge-Kutta.
///
/// The Van der Pol system:
/// ```text
/// dx/dt = y
/// dy/dt = μ (1 − x²) y − x
/// ```
///
/// # Arguments
///
/// * `n`  – Number of time steps (must be > 0).
/// * `mu` – Nonlinearity parameter (must be ≥ 0; `mu = 0` gives a simple harmonic oscillator).
/// * `dt` – Integration step size (must be > 0).
///
/// # Returns
///
/// `Array2<f64>` of shape `(n, 2)` — columns are x (displacement) and y (velocity).
///
/// # Errors
///
/// Returns an error if `n == 0`, `dt <= 0`, or `mu < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::ts_datasets::make_van_der_pol;
///
/// let traj = make_van_der_pol(500, 1.0, 0.01).expect("make_van_der_pol failed");
/// assert_eq!(traj.nrows(), 500);
/// assert_eq!(traj.ncols(), 2);
/// ```
pub fn make_van_der_pol(n: usize, mu: f64, dt: f64) -> Result<Array2<f64>> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_van_der_pol: n must be > 0".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_van_der_pol: dt must be > 0".to_string(),
        ));
    }
    if mu < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_van_der_pol: mu must be >= 0".to_string(),
        ));
    }

    let vdp_deriv = |x: f64, y: f64| -> (f64, f64) {
        (y, mu * (1.0 - x * x) * y - x)
    };

    let mut x = 2.0_f64;
    let mut y = 0.0_f64;

    let mut out = Array2::zeros((n, 2));
    for i in 0..n {
        out[[i, 0]] = x;
        out[[i, 1]] = y;

        let (k1x, k1y) = vdp_deriv(x, y);
        let (k2x, k2y) = vdp_deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
        let (k3x, k3y) = vdp_deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
        let (k4x, k4y) = vdp_deriv(x + dt * k3x, y + dt * k3y);
        x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_seasonal (new signature returning Array1<f64>)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a seasonal time series with linear trend and additive noise.
///
/// ```text
/// y[t] = trend * t + seasonality * sin(2π t / period) + N(0, noise²)
/// ```
///
/// # Arguments
///
/// * `n`            – Length of the series (must be > 0).
/// * `trend`        – Per-step linear trend coefficient.
/// * `seasonality`  – Amplitude of the seasonal sinusoidal component.
/// * `noise`        – Standard deviation of the additive Gaussian noise (must be ≥ 0).
/// * `period`       – Number of steps per seasonal cycle (must be > 0).
///
/// # Returns
///
/// `Array1<f64>` of length `n`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::ts_datasets::make_seasonal_ts;
///
/// let y = make_seasonal_ts(120, 0.1, 2.0, 0.5, 12, 42).expect("failed");
/// assert_eq!(y.len(), 120);
/// ```
pub fn make_seasonal_ts(
    n: usize,
    trend: f64,
    seasonality: f64,
    noise: f64,
    period: usize,
    seed: u64,
) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_seasonal_ts: n must be > 0".to_string(),
        ));
    }
    if period == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_seasonal_ts: period must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_seasonal_ts: noise must be >= 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let noise_dist = scirs2_core::random::Normal::new(0.0_f64, noise.max(1e-12)).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })?;

    let mut out = Array1::zeros(n);
    for t in 0..n {
        let t_f = t as f64;
        let seasonal = seasonality * (2.0 * PI * t_f / period as f64).sin();
        let eps = if noise > 0.0 {
            noise_dist.sample(&mut rng)
        } else {
            0.0
        };
        out[t] = trend * t_f + seasonal + eps;
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// make_changepoint_series
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a piecewise-constant time series with Gaussian noise at each segment.
///
/// `changepoints` specifies the indices where the mean shifts.  There must be
/// exactly `changepoints.len() + 1` means in `means`.
///
/// # Arguments
///
/// * `n`            – Total series length (must be > 0).
/// * `changepoints` – Sorted list of change-point indices (each must be < n).
/// * `means`        – Mean value for each segment (length == changepoints.len() + 1).
/// * `noise_std`    – Standard deviation of per-sample Gaussian noise (must be ≥ 0).
/// * `seed`         – Random seed.
///
/// # Returns
///
/// `Array1<f64>` of length `n`.
///
/// # Errors
///
/// Returns an error if the argument lengths are inconsistent or invalid.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::ts_datasets::make_changepoint_series;
///
/// let cps = [50, 150];
/// let means = [0.0, 5.0, -2.0];
/// let y = make_changepoint_series(300, &cps, &means, 1.0, 42).expect("failed");
/// assert_eq!(y.len(), 300);
/// ```
pub fn make_changepoint_series(
    n: usize,
    changepoints: &[usize],
    means: &[f64],
    noise_std: f64,
    seed: u64,
) -> Result<Array1<f64>> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_changepoint_series: n must be > 0".to_string(),
        ));
    }
    if means.len() != changepoints.len() + 1 {
        return Err(DatasetsError::InvalidFormat(format!(
            "make_changepoint_series: means.len() ({}) must be changepoints.len() + 1 ({})",
            means.len(),
            changepoints.len() + 1
        )));
    }
    if noise_std < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "make_changepoint_series: noise_std must be >= 0".to_string(),
        ));
    }
    // Validate that changepoints are strictly increasing and within bounds
    for (i, &cp) in changepoints.iter().enumerate() {
        if cp == 0 || cp >= n {
            return Err(DatasetsError::InvalidFormat(format!(
                "make_changepoint_series: changepoint[{i}]={cp} is out of (0, n={n}) range"
            )));
        }
        if i > 0 && cp <= changepoints[i - 1] {
            return Err(DatasetsError::InvalidFormat(
                "make_changepoint_series: changepoints must be strictly increasing".to_string(),
            ));
        }
    }

    let mut rng = make_rng(seed);
    let noise_dist =
        scirs2_core::random::Normal::new(0.0_f64, noise_std.max(1e-12)).map_err(|e| {
            DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
        })?;

    // Build boundary list: [0, cp0, cp1, ..., n]
    let mut boundaries: Vec<usize> = vec![0];
    boundaries.extend_from_slice(changepoints);
    boundaries.push(n);

    let mut out = Array1::zeros(n);
    for seg in 0..means.len() {
        let start = boundaries[seg];
        let end = boundaries[seg + 1];
        let mean = means[seg];
        for t in start..end {
            let eps = if noise_std > 0.0 {
                noise_dist.sample(&mut rng)
            } else {
                0.0
            };
            out[t] = mean + eps;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── make_arma ────────────────────────────────────────────────────────────

    #[test]
    fn test_arma_length() {
        let y = make_arma(200, &[0.5], &[], 1).expect("make_arma failed");
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_arma_arma11() {
        let y = make_arma(500, &[0.6], &[0.3], 7).expect("make_arma ARMA(1,1) failed");
        assert_eq!(y.len(), 500);
        // Series should not be constant
        let first = y[0];
        assert!(y.iter().any(|&v| (v - first).abs() > 1e-10));
    }

    #[test]
    fn test_arma_pure_ma() {
        let y = make_arma(100, &[], &[0.5, 0.3], 99).expect("MA(2) failed");
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_arma_determinism() {
        let y1 = make_arma(100, &[0.4], &[0.2], 42).expect("seed 1");
        let y2 = make_arma(100, &[0.4], &[0.2], 42).expect("seed 2");
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_arma_error_n_zero() {
        assert!(make_arma(0, &[0.5], &[], 1).is_err());
    }

    // ── make_lorenz ──────────────────────────────────────────────────────────

    #[test]
    fn test_lorenz_shape() {
        let traj = make_lorenz(1000, 0.01, 10.0, 28.0, 8.0 / 3.0).expect("lorenz failed");
        assert_eq!(traj.nrows(), 1000);
        assert_eq!(traj.ncols(), 3);
    }

    #[test]
    fn test_lorenz_not_constant() {
        let traj = make_lorenz(200, 0.01, 10.0, 28.0, 8.0 / 3.0).expect("lorenz");
        // x-coordinate should change
        let x0 = traj[[0, 0]];
        assert!(traj.column(0).iter().any(|&v| (v - x0).abs() > 0.1));
    }

    #[test]
    fn test_lorenz_error_n_zero() {
        assert!(make_lorenz(0, 0.01, 10.0, 28.0, 8.0 / 3.0).is_err());
    }

    #[test]
    fn test_lorenz_error_dt_nonpositive() {
        assert!(make_lorenz(100, 0.0, 10.0, 28.0, 8.0 / 3.0).is_err());
        assert!(make_lorenz(100, -0.01, 10.0, 28.0, 8.0 / 3.0).is_err());
    }

    // ── make_van_der_pol ─────────────────────────────────────────────────────

    #[test]
    fn test_vdp_shape() {
        let traj = make_van_der_pol(400, 1.0, 0.01).expect("vdp failed");
        assert_eq!(traj.nrows(), 400);
        assert_eq!(traj.ncols(), 2);
    }

    #[test]
    fn test_vdp_oscillatory() {
        let traj = make_van_der_pol(1000, 0.5, 0.01).expect("vdp oscillatory");
        let x_max = traj.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let x_min = traj.column(0).fold(f64::INFINITY, |a, &b| a.min(b));
        // Limit cycle amplitude should be order-1
        assert!(x_max > 0.5);
        assert!(x_min < -0.5);
    }

    #[test]
    fn test_vdp_error_n_zero() {
        assert!(make_van_der_pol(0, 1.0, 0.01).is_err());
    }

    #[test]
    fn test_vdp_error_mu_negative() {
        assert!(make_van_der_pol(100, -1.0, 0.01).is_err());
    }

    // ── make_seasonal_ts ─────────────────────────────────────────────────────

    #[test]
    fn test_seasonal_length() {
        let y = make_seasonal_ts(120, 0.1, 2.0, 0.5, 12, 42).expect("seasonal ts");
        assert_eq!(y.len(), 120);
    }

    #[test]
    fn test_seasonal_no_noise_deterministic() {
        let y1 = make_seasonal_ts(50, 0.0, 1.0, 0.0, 10, 0).expect("s1");
        let y2 = make_seasonal_ts(50, 0.0, 1.0, 0.0, 10, 0).expect("s2");
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_seasonal_trend_visible() {
        // With zero noise and positive trend the last value should exceed the first
        let y = make_seasonal_ts(100, 1.0, 0.0, 0.0, 10, 1).expect("seasonal trend");
        assert!(y[99] > y[0]);
    }

    #[test]
    fn test_seasonal_error_period_zero() {
        assert!(make_seasonal_ts(100, 0.0, 1.0, 0.0, 0, 1).is_err());
    }

    // ── make_changepoint_series ──────────────────────────────────────────────

    #[test]
    fn test_changepoint_length() {
        let y = make_changepoint_series(300, &[100, 200], &[0.0, 5.0, -3.0], 1.0, 42)
            .expect("changepoint");
        assert_eq!(y.len(), 300);
    }

    #[test]
    fn test_changepoint_means_visible() {
        // No noise → segment means should be exact
        let y = make_changepoint_series(300, &[100, 200], &[0.0, 10.0, -5.0], 0.0, 0)
            .expect("changepoint no noise");
        // Segment 0: indices 0..100 → mean ≈ 0.0
        let seg0_mean: f64 = y.slice(scirs2_core::ndarray::s![0..100]).mean().unwrap_or(0.0);
        assert!((seg0_mean - 0.0).abs() < 1e-9, "seg0 mean={seg0_mean}");
        // Segment 1: indices 100..200 → mean ≈ 10.0
        let seg1_mean: f64 = y.slice(scirs2_core::ndarray::s![100..200]).mean().unwrap_or(0.0);
        assert!((seg1_mean - 10.0).abs() < 1e-9, "seg1 mean={seg1_mean}");
        // Segment 2: indices 200..300 → mean ≈ -5.0
        let seg2_mean: f64 = y.slice(scirs2_core::ndarray::s![200..300]).mean().unwrap_or(0.0);
        assert!((seg2_mean - (-5.0)).abs() < 1e-9, "seg2 mean={seg2_mean}");
    }

    #[test]
    fn test_changepoint_error_wrong_means_len() {
        // 2 changepoints → 3 means required; give 2 → error
        assert!(make_changepoint_series(200, &[50, 150], &[0.0, 1.0], 0.5, 1).is_err());
    }

    #[test]
    fn test_changepoint_error_n_zero() {
        assert!(make_changepoint_series(0, &[], &[1.0], 0.5, 1).is_err());
    }

    #[test]
    fn test_changepoint_no_changepoints() {
        // Single segment
        let y = make_changepoint_series(100, &[], &[3.0], 0.0, 5).expect("single segment");
        assert_eq!(y.len(), 100);
        assert!((y[0] - 3.0).abs() < 1e-9);
        assert!((y[99] - 3.0).abs() < 1e-9);
    }
}
