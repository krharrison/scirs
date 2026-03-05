//! Time series dataset generators
//!
//! This module provides synthetic time series generators for benchmarking
//! and testing time series analysis algorithms. Includes:
//!
//! - **Sine wave**: periodic sinusoidal signal with configurable frequency/amplitude
//! - **Random walk**: cumulative sum process with drift and volatility
//! - **AR process**: autoregressive process with specified coefficients
//! - **Seasonal**: seasonal decomposition signal with trend and noise

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

/// Helper to create an RNG from an optional seed
fn create_rng(randomseed: Option<u64>) -> StdRng {
    match randomseed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    }
}

/// Generate a sine wave time series dataset
///
/// Creates a sinusoidal signal y(t) = amplitude * sin(2 * pi * frequency * t / n_samples) + noise.
///
/// # Arguments
///
/// * `n_samples` - Number of time steps to generate (must be > 0)
/// * `frequency` - Number of complete cycles across the time series (must be > 0)
/// * `amplitude` - Peak amplitude of the sine wave (must be > 0)
/// * `noise` - Standard deviation of additive Gaussian noise (must be >= 0)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_samples, 1) containing time indices
/// - `target` contains the sine wave values
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::time_series::make_sine_wave;
///
/// let ds = make_sine_wave(100, 2.0, 1.0, 0.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// ```
pub fn make_sine_wave(
    n_samples: usize,
    frequency: f64,
    amplitude: f64,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if frequency <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "frequency must be > 0".to_string(),
        ));
    }
    if amplitude <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "amplitude must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);

    let noise_dist = scirs2_core::random::Normal::new(0.0, if noise > 0.0 { noise } else { 1.0 })
        .map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut data = Array2::zeros((n_samples, 1));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i as f64;
        data[[i, 0]] = t;

        let value = amplitude * (2.0 * PI * frequency * t / n_samples as f64).sin();
        let noise_val = if noise > 0.0 {
            noise_dist.sample(&mut rng)
        } else {
            0.0
        };
        target[i] = value + noise_val;
    }

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(vec!["time".to_string()])
        .with_description(format!(
            "Sine wave time series: frequency={frequency}, amplitude={amplitude}"
        ))
        .with_metadata("generator", "make_sine_wave")
        .with_metadata("frequency", &frequency.to_string())
        .with_metadata("amplitude", &amplitude.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a random walk time series dataset
///
/// Creates a random walk: y(t) = y(t-1) + drift + volatility * N(0,1).
/// Starting value y(0) = 0.
///
/// # Arguments
///
/// * `n_samples` - Number of time steps (must be > 0)
/// * `drift` - Constant drift added at each step
/// * `volatility` - Standard deviation of the random increments (must be > 0)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_samples, 1) containing time indices
/// - `target` contains the random walk values
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::time_series::make_random_walk;
///
/// let ds = make_random_walk(200, 0.01, 1.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 200);
/// ```
pub fn make_random_walk(
    n_samples: usize,
    drift: f64,
    volatility: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if volatility <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "volatility must be > 0".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);

    let noise_dist = scirs2_core::random::Normal::new(0.0, volatility).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut data = Array2::zeros((n_samples, 1));
    let mut target = Array1::zeros(n_samples);

    let mut current_value = 0.0;
    for i in 0..n_samples {
        data[[i, 0]] = i as f64;
        target[i] = current_value;
        current_value += drift + noise_dist.sample(&mut rng);
    }

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(vec!["time".to_string()])
        .with_description(format!(
            "Random walk time series: drift={drift}, volatility={volatility}"
        ))
        .with_metadata("generator", "make_random_walk")
        .with_metadata("drift", &drift.to_string())
        .with_metadata("volatility", &volatility.to_string());

    Ok(dataset)
}

/// Generate an autoregressive (AR) process time series dataset
///
/// Creates an AR(p) process: y(t) = c1*y(t-1) + c2*y(t-2) + ... + cp*y(t-p) + noise_std * N(0,1).
/// The process is initialized with zeros and uses a burn-in period of `max(100, 2*p)` steps
/// (discarded) so the output is stationary.
///
/// # Arguments
///
/// * `n_samples` - Number of time steps to output (must be > 0)
/// * `coefficients` - AR coefficients [c1, c2, ..., cp] (must be non-empty)
/// * `noise_std` - Standard deviation of the innovation noise (must be > 0)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_samples, 1) containing time indices
/// - `target` contains the AR process values
///
/// # Notes
///
/// For a stable (stationary) AR process, the roots of the characteristic polynomial
/// 1 - c1*z - c2*z^2 - ... - cp*z^p should all lie outside the unit circle.
/// This function does not enforce stationarity; if coefficients lead to an unstable
/// process the values may diverge.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::time_series::make_ar_process;
///
/// // AR(2) process with coefficients [0.5, 0.3]
/// let ds = make_ar_process(300, &[0.5, 0.3], 1.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 300);
/// ```
pub fn make_ar_process(
    n_samples: usize,
    coefficients: &[f64],
    noise_std: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if coefficients.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "coefficients must be non-empty".to_string(),
        ));
    }
    if noise_std <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise_std must be > 0".to_string(),
        ));
    }

    let p = coefficients.len();
    let burn_in = std::cmp::max(100, 2 * p);
    let total_steps = n_samples + burn_in;

    let mut rng = create_rng(randomseed);
    let noise_dist = scirs2_core::random::Normal::new(0.0, noise_std).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Generate the full AR process (burn_in + n_samples)
    let mut values = vec![0.0_f64; total_steps];

    for t in p..total_steps {
        let mut val = noise_dist.sample(&mut rng);
        for (lag, coeff) in coefficients.iter().enumerate() {
            val += coeff * values[t - 1 - lag];
        }
        values[t] = val;
    }

    // Discard burn-in and build the dataset
    let mut data = Array2::zeros((n_samples, 1));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        data[[i, 0]] = i as f64;
        target[i] = values[burn_in + i];
    }

    let coef_str = format!("{coefficients:?}");
    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(vec!["time".to_string()])
        .with_description(format!(
            "AR({p}) process with coefficients {coef_str}, noise_std={noise_std}"
        ))
        .with_metadata("generator", "make_ar_process")
        .with_metadata("order", &p.to_string())
        .with_metadata("coefficients", &coef_str)
        .with_metadata("noise_std", &noise_std.to_string());

    Ok(dataset)
}

/// Generate a seasonal time series dataset
///
/// Creates a signal with: y(t) = trend * t + amplitude * sin(2*pi*t / period) + noise * N(0,1).
///
/// # Arguments
///
/// * `n_samples` - Number of time steps (must be > 0)
/// * `period` - Length of one seasonal cycle in samples (must be > 0)
/// * `amplitude` - Amplitude of the seasonal component (must be >= 0)
/// * `trend` - Linear trend slope (coefficient of t)
/// * `noise` - Standard deviation of additive noise (must be >= 0)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n_samples, 1) containing time indices
/// - `target` contains the seasonal signal values
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::time_series::make_seasonal;
///
/// let ds = make_seasonal(365, 30.0, 5.0, 0.01, 0.5, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 365);
/// ```
pub fn make_seasonal(
    n_samples: usize,
    period: f64,
    amplitude: f64,
    trend: f64,
    noise: f64,
    randomseed: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if period <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "period must be > 0".to_string(),
        ));
    }
    if amplitude < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "amplitude must be >= 0.0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0.0".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);

    let noise_dist = scirs2_core::random::Normal::new(0.0, if noise > 0.0 { noise } else { 1.0 })
        .map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut data = Array2::zeros((n_samples, 1));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i as f64;
        data[[i, 0]] = t;

        let trend_component = trend * t;
        let seasonal_component = amplitude * (2.0 * PI * t / period).sin();
        let noise_component = if noise > 0.0 {
            noise_dist.sample(&mut rng)
        } else {
            0.0
        };

        target[i] = trend_component + seasonal_component + noise_component;
    }

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(vec!["time".to_string()])
        .with_description(format!(
            "Seasonal time series: period={period}, amplitude={amplitude}, trend={trend}"
        ))
        .with_metadata("generator", "make_seasonal")
        .with_metadata("period", &period.to_string())
        .with_metadata("amplitude", &amplitude.to_string())
        .with_metadata("trend", &trend.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_sine_wave tests
    // =========================================================================

    #[test]
    fn test_sine_wave_shape() {
        let ds = make_sine_wave(100, 2.0, 1.0, 0.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 1);
        assert!(ds.target.is_some());
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 100);
    }

    #[test]
    fn test_sine_wave_properties_no_noise() {
        let ds = make_sine_wave(1000, 1.0, 3.0, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        // Without noise, all values should be in [-amplitude, amplitude]
        for &val in target.iter() {
            assert!(
                val >= -3.0 - 1e-10 && val <= 3.0 + 1e-10,
                "Value {val} out of range"
            );
        }

        // The mean of a full-cycle sine wave should be approximately 0
        let mean: f64 = target.iter().sum::<f64>() / target.len() as f64;
        assert!(
            mean.abs() < 0.1,
            "Mean {mean} should be near 0 for a complete cycle"
        );
    }

    #[test]
    fn test_sine_wave_reproducibility() {
        let ds1 = make_sine_wave(50, 2.0, 1.0, 0.3, Some(123)).expect("should succeed");
        let ds2 = make_sine_wave(50, 2.0, 1.0, 0.3, Some(123)).expect("should succeed");
        let t1 = ds1.target.as_ref().expect("target exists");
        let t2 = ds2.target.as_ref().expect("target exists");
        for i in 0..50 {
            assert!(
                (t1[i] - t2[i]).abs() < 1e-15,
                "Sample {i} differs: {} vs {}",
                t1[i],
                t2[i]
            );
        }
    }

    #[test]
    fn test_sine_wave_validation() {
        assert!(make_sine_wave(0, 1.0, 1.0, 0.0, None).is_err());
        assert!(make_sine_wave(10, 0.0, 1.0, 0.0, None).is_err());
        assert!(make_sine_wave(10, -1.0, 1.0, 0.0, None).is_err());
        assert!(make_sine_wave(10, 1.0, 0.0, 0.0, None).is_err());
        assert!(make_sine_wave(10, 1.0, 1.0, -0.1, None).is_err());
    }

    // =========================================================================
    // make_random_walk tests
    // =========================================================================

    #[test]
    fn test_random_walk_shape() {
        let ds = make_random_walk(200, 0.0, 1.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 200);
        assert_eq!(ds.n_features(), 1);
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 200);
    }

    #[test]
    fn test_random_walk_starts_at_zero() {
        let ds = make_random_walk(100, 0.0, 1.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");
        assert!(
            target[0].abs() < 1e-15,
            "Random walk should start at 0, got {}",
            target[0]
        );
    }

    #[test]
    fn test_random_walk_drift() {
        // With a strong positive drift, the final value should generally be positive
        let ds = make_random_walk(1000, 1.0, 0.1, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");
        assert!(
            target[999] > 0.0,
            "With drift=1.0, final value should be positive, got {}",
            target[999]
        );
    }

    #[test]
    fn test_random_walk_reproducibility() {
        let ds1 = make_random_walk(50, 0.01, 1.0, Some(99)).expect("should succeed");
        let ds2 = make_random_walk(50, 0.01, 1.0, Some(99)).expect("should succeed");
        let t1 = ds1.target.as_ref().expect("target exists");
        let t2 = ds2.target.as_ref().expect("target exists");
        for i in 0..50 {
            assert!(
                (t1[i] - t2[i]).abs() < 1e-15,
                "Sample {i} differs: {} vs {}",
                t1[i],
                t2[i]
            );
        }
    }

    #[test]
    fn test_random_walk_validation() {
        assert!(make_random_walk(0, 0.0, 1.0, None).is_err());
        assert!(make_random_walk(10, 0.0, 0.0, None).is_err());
        assert!(make_random_walk(10, 0.0, -1.0, None).is_err());
    }

    // =========================================================================
    // make_ar_process tests
    // =========================================================================

    #[test]
    fn test_ar_process_shape() {
        let ds = make_ar_process(300, &[0.5, 0.3], 1.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 300);
        assert_eq!(ds.n_features(), 1);
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 300);
    }

    #[test]
    fn test_ar_process_stationary_bounded() {
        // AR(1) with coefficient 0.5 is stable; values should stay bounded
        let ds = make_ar_process(1000, &[0.5], 1.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        let max_abs = target.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        // For a stationary AR(1) with phi=0.5, sigma_noise=1.0, the std is ~1.15
        // So values exceeding 20 would be extremely unlikely
        assert!(
            max_abs < 20.0,
            "Stationary AR(1) values should stay bounded, max_abs={}",
            max_abs
        );
    }

    #[test]
    fn test_ar_process_reproducibility() {
        let ds1 = make_ar_process(100, &[0.7, -0.2], 0.5, Some(77)).expect("should succeed");
        let ds2 = make_ar_process(100, &[0.7, -0.2], 0.5, Some(77)).expect("should succeed");
        let t1 = ds1.target.as_ref().expect("target exists");
        let t2 = ds2.target.as_ref().expect("target exists");
        for i in 0..100 {
            assert!(
                (t1[i] - t2[i]).abs() < 1e-15,
                "Sample {i} differs: {} vs {}",
                t1[i],
                t2[i]
            );
        }
    }

    #[test]
    fn test_ar_process_autocorrelation() {
        // AR(1) with coefficient 0.9 should have strong positive autocorrelation at lag 1
        let ds = make_ar_process(5000, &[0.9], 1.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        let mean = target.iter().sum::<f64>() / target.len() as f64;
        let var: f64 =
            target.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / target.len() as f64;

        if var > 1e-10 {
            let cov_lag1: f64 = (1..target.len())
                .map(|i| (target[i] - mean) * (target[i - 1] - mean))
                .sum::<f64>()
                / (target.len() - 1) as f64;
            let acf1 = cov_lag1 / var;

            // For AR(1) with phi=0.9, theoretical ACF(1) = 0.9
            assert!(
                acf1 > 0.7,
                "ACF at lag 1 should be high for AR(1) with phi=0.9, got {acf1}"
            );
        }
    }

    #[test]
    fn test_ar_process_validation() {
        assert!(make_ar_process(0, &[0.5], 1.0, None).is_err());
        assert!(make_ar_process(10, &[], 1.0, None).is_err());
        assert!(make_ar_process(10, &[0.5], 0.0, None).is_err());
        assert!(make_ar_process(10, &[0.5], -1.0, None).is_err());
    }

    // =========================================================================
    // make_seasonal tests
    // =========================================================================

    #[test]
    fn test_seasonal_shape() {
        let ds = make_seasonal(365, 30.0, 5.0, 0.01, 0.5, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 365);
        assert_eq!(ds.n_features(), 1);
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 365);
    }

    #[test]
    fn test_seasonal_no_noise_no_trend() {
        // Pure seasonal signal
        let ds = make_seasonal(360, 60.0, 2.0, 0.0, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        // All values should be in [-amplitude, amplitude]
        for &val in target.iter() {
            assert!(
                val >= -2.0 - 1e-10 && val <= 2.0 + 1e-10,
                "Value {val} out of range [-2, 2]"
            );
        }

        // Check periodicity: values should repeat with given period
        // y(t) = amplitude * sin(2*pi*t/period)
        // y(t + period) = amplitude * sin(2*pi*(t+period)/period) = amplitude * sin(2*pi*t/period + 2*pi) = y(t)
        let period = 60;
        for i in 0..(360 - period) {
            let diff = (target[i] - target[i + period]).abs();
            assert!(
                diff < 1e-10,
                "Seasonal signal should repeat with period {period}, diff={diff} at t={i}"
            );
        }
    }

    #[test]
    fn test_seasonal_trend() {
        // Signal with trend and no noise
        let ds = make_seasonal(100, 50.0, 0.0, 1.0, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        // With amplitude=0 and noise=0, y(t) = trend*t = 1.0*t
        for i in 0..100 {
            let expected = 1.0 * i as f64;
            let diff = (target[i] - expected).abs();
            assert!(
                diff < 1e-10,
                "Pure trend: expected {expected}, got {}, diff={diff}",
                target[i]
            );
        }
    }

    #[test]
    fn test_seasonal_reproducibility() {
        let ds1 = make_seasonal(100, 20.0, 3.0, 0.05, 1.0, Some(55)).expect("should succeed");
        let ds2 = make_seasonal(100, 20.0, 3.0, 0.05, 1.0, Some(55)).expect("should succeed");
        let t1 = ds1.target.as_ref().expect("target exists");
        let t2 = ds2.target.as_ref().expect("target exists");
        for i in 0..100 {
            assert!(
                (t1[i] - t2[i]).abs() < 1e-15,
                "Sample {i} differs: {} vs {}",
                t1[i],
                t2[i]
            );
        }
    }

    #[test]
    fn test_seasonal_validation() {
        assert!(make_seasonal(0, 10.0, 1.0, 0.0, 0.0, None).is_err());
        assert!(make_seasonal(10, 0.0, 1.0, 0.0, 0.0, None).is_err());
        assert!(make_seasonal(10, -5.0, 1.0, 0.0, 0.0, None).is_err());
        assert!(make_seasonal(10, 10.0, -1.0, 0.0, 0.0, None).is_err());
        assert!(make_seasonal(10, 10.0, 1.0, 0.0, -0.5, None).is_err());
    }
}
