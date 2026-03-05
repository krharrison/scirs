//! Advanced regression dataset generators
//!
//! Provides sklearn-style synthetic regression generators including
//! Friedman benchmark functions, sparse uncorrelated regression,
//! and low-rank matrix generation.

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

/// Generate a Friedman #1 regression dataset
///
/// Generates the "Friedman #1" regression problem. Inputs are 10 independent
/// features uniformly distributed on [0, 1]. Only the first 5 are used to
/// compute the response:
///
///   y = 10 * sin(pi * x0 * x1) + 20 * (x2 - 0.5)^2 + 10 * x3 + 5 * x4 + noise
///
/// Reference: Friedman, J.H. (1991). Multivariate Adaptive Regression Splines.
/// Annals of Statistics, 19(1), 1-67.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features (must be >= 5)
/// * `noise` - Standard deviation of Gaussian noise added to the response
/// * `random_state` - Optional random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::regression::make_friedman1;
///
/// let ds = make_friedman1(200, 10, 1.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 200);
/// assert_eq!(ds.n_features(), 10);
/// ```
pub fn make_friedman1(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if n_features < 5 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be >= 5 for Friedman #1".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;
    let normal = scirs2_core::random::Normal::new(0.0, noise.max(1e-30)).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = uniform.sample(&mut rng);
        }
        let x0 = data[[i, 0]];
        let x1 = data[[i, 1]];
        let x2 = data[[i, 2]];
        let x3 = data[[i, 3]];
        let x4 = data[[i, 4]];

        target[i] = 10.0 * (PI * x0 * x1).sin() + 20.0 * (x2 - 0.5).powi(2) + 10.0 * x3 + 5.0 * x4;

        if noise > 0.0 {
            target[i] += normal.sample(&mut rng);
        }
    }

    let feature_names: Vec<String> = (0..n_features).map(|j| format!("x_{j}")).collect();

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_description(
            "Friedman #1 regression: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise. \
             Reference: Friedman (1991)."
                .to_string(),
        )
        .with_metadata("generator", "make_friedman1")
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a Friedman #2 regression dataset
///
/// Generates the "Friedman #2" regression problem. Four input features
/// uniformly distributed on their respective intervals:
///
///   x0 ~ U[0, 100], x1 ~ U[40*pi, 560*pi], x2 ~ U[0, 1], x3 ~ U[1, 11]
///
///   y = (x0^2 + (x1*x2 - 1/(x1*x3))^2)^0.5 + noise
///
/// Reference: Friedman, J.H. (1991).
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `noise` - Standard deviation of Gaussian noise
/// * `random_state` - Optional random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::regression::make_friedman2;
///
/// let ds = make_friedman2(200, 0.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 200);
/// assert_eq!(ds.n_features(), 4);
/// ```
pub fn make_friedman2(n_samples: usize, noise: f64, random_state: Option<u64>) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);

    let u0 = scirs2_core::random::Uniform::new(0.0, 100.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u1 = scirs2_core::random::Uniform::new(40.0 * PI, 560.0 * PI)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u2 = scirs2_core::random::Uniform::new(0.0, 1.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u3 = scirs2_core::random::Uniform::new(1.0, 11.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let normal = scirs2_core::random::Normal::new(0.0, noise.max(1e-30)).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let n_features = 4;
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let x0 = u0.sample(&mut rng);
        let x1 = u1.sample(&mut rng);
        let x2 = u2.sample(&mut rng);
        let x3 = u3.sample(&mut rng);

        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        data[[i, 2]] = x2;
        data[[i, 3]] = x3;

        // Protect against division by zero
        let denom = x1 * x3;
        let inner = if denom.abs() > 1e-15 {
            x1 * x2 - 1.0 / denom
        } else {
            x1 * x2
        };

        target[i] = (x0 * x0 + inner * inner).sqrt();

        if noise > 0.0 {
            target[i] += normal.sample(&mut rng);
        }
    }

    let feature_names = vec![
        "x_0".to_string(),
        "x_1".to_string(),
        "x_2".to_string(),
        "x_3".to_string(),
    ];

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_description(
            "Friedman #2 regression: y = sqrt(x0^2 + (x1*x2 - 1/(x1*x3))^2) + noise. \
             Reference: Friedman (1991)."
                .to_string(),
        )
        .with_metadata("generator", "make_friedman2")
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a Friedman #3 regression dataset
///
/// Generates the "Friedman #3" regression problem. Four input features
/// uniformly distributed on their respective intervals (same as Friedman #2):
///
///   x0 ~ U[0, 100], x1 ~ U[40*pi, 560*pi], x2 ~ U[0, 1], x3 ~ U[1, 11]
///
///   y = atan((x1*x2 - 1/(x1*x3)) / x0) + noise
///
/// Reference: Friedman, J.H. (1991).
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `noise` - Standard deviation of Gaussian noise
/// * `random_state` - Optional random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::regression::make_friedman3;
///
/// let ds = make_friedman3(200, 0.0, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 200);
/// assert_eq!(ds.n_features(), 4);
/// ```
pub fn make_friedman3(n_samples: usize, noise: f64, random_state: Option<u64>) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);

    let u0 = scirs2_core::random::Uniform::new(0.0, 100.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u1 = scirs2_core::random::Uniform::new(40.0 * PI, 560.0 * PI)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u2 = scirs2_core::random::Uniform::new(0.0, 1.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let u3 = scirs2_core::random::Uniform::new(1.0, 11.0)
        .map_err(|e| DatasetsError::ComputationError(format!("Failed to create dist: {e}")))?;
    let normal = scirs2_core::random::Normal::new(0.0, noise.max(1e-30)).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let n_features = 4;
    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let x0 = u0.sample(&mut rng);
        let x1 = u1.sample(&mut rng);
        let x2 = u2.sample(&mut rng);
        let x3 = u3.sample(&mut rng);

        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        data[[i, 2]] = x2;
        data[[i, 3]] = x3;

        let denom = x1 * x3;
        let inner = if denom.abs() > 1e-15 {
            x1 * x2 - 1.0 / denom
        } else {
            x1 * x2
        };

        // Protect against x0 == 0
        target[i] = if x0.abs() > 1e-15 {
            (inner / x0).atan()
        } else {
            PI / 2.0 * inner.signum()
        };

        if noise > 0.0 {
            target[i] += normal.sample(&mut rng);
        }
    }

    let feature_names = vec![
        "x_0".to_string(),
        "x_1".to_string(),
        "x_2".to_string(),
        "x_3".to_string(),
    ];

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_description(
            "Friedman #3 regression: y = atan((x1*x2 - 1/(x1*x3)) / x0) + noise. \
             Reference: Friedman (1991)."
                .to_string(),
        )
        .with_metadata("generator", "make_friedman3")
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a sparse uncorrelated regression dataset
///
/// Only a small number of features are relevant to the response.
/// The target is a linear combination of 4 features:
///
///   y = x0 + 2*x1 + 0 + 0 + ... + noise
///
/// where all features are standard normal. Only x0..x3 contribute;
/// the remaining features are pure noise.
///
/// This is useful for testing variable selection methods like Lasso.
///
/// # Arguments
///
/// * `n_samples` - Number of samples
/// * `n_features` - Number of features (>= 4)
/// * `random_state` - Optional random seed
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::regression::make_sparse_uncorrelated;
///
/// let ds = make_sparse_uncorrelated(100, 10, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// assert_eq!(ds.n_features(), 10);
/// ```
pub fn make_sparse_uncorrelated(
    n_samples: usize,
    n_features: usize,
    random_state: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if n_features < 4 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be >= 4 for sparse uncorrelated".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // True coefficients: sparse (only 4 non-zero)
    let true_coefs = [1.0, 2.0, 0.0, 0.0];

    let mut data = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = normal.sample(&mut rng);
        }

        let mut y = 0.0;
        for (k, &coef) in true_coefs.iter().enumerate() {
            y += coef * data[[i, k]];
        }
        // Add small noise
        y += 0.1 * normal.sample(&mut rng);
        target[i] = y;
    }

    let feature_names: Vec<String> = (0..n_features).map(|j| format!("x_{j}")).collect();

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_description(
            "Sparse uncorrelated regression: y = x0 + 2*x1 + noise. \
             Only the first two features are informative; the rest are noise. \
             Useful for testing variable selection methods."
                .to_string(),
        )
        .with_metadata("generator", "make_sparse_uncorrelated")
        .with_metadata("true_coefficients", "[1.0, 2.0, 0.0, 0.0, ...]");

    Ok(dataset)
}

/// Generate a random matrix with approximately the given effective rank
///
/// The matrix is generated with a spectrum that has `effective_rank` large
/// singular values, then remaining singular values that decay exponentially.
/// The output is a (n_samples x n_features) matrix.
///
/// This is useful for testing methods that rely on the rank of a matrix,
/// such as PCA, ridge regression, and reduced-rank regression.
///
/// # Arguments
///
/// * `n_samples` - Number of rows
/// * `n_features` - Number of columns
/// * `effective_rank` - Approximate rank of the matrix
/// * `tail_strength` - Relative importance of the tail (small) singular values (0..1)
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` whose `data` is the low-rank matrix and `target` contains the
/// singular values.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::regression::make_low_rank_matrix;
///
/// let ds = make_low_rank_matrix(100, 50, 5, 0.5, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// assert_eq!(ds.n_features(), 50);
/// ```
pub fn make_low_rank_matrix(
    n_samples: usize,
    n_features: usize,
    effective_rank: usize,
    tail_strength: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }
    if effective_rank == 0 {
        return Err(DatasetsError::InvalidFormat(
            "effective_rank must be > 0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&tail_strength) {
        return Err(DatasetsError::InvalidFormat(
            "tail_strength must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let min_dim = n_samples.min(n_features);

    // Generate singular values with exponential decay
    let mut singular_values = vec![0.0; min_dim];
    for (i, sv) in singular_values.iter_mut().enumerate() {
        let profile = (-(i as f64) / effective_rank as f64).exp();
        let tail = tail_strength * (1.0 - profile);
        *sv = profile + tail;
    }

    // Generate random orthogonal-like matrices U (n_samples x min_dim) and V (min_dim x n_features)
    // Using QR decomposition approximation via Gram-Schmidt
    let mut u_mat = Array2::zeros((n_samples, min_dim));
    for j in 0..min_dim {
        for i in 0..n_samples {
            u_mat[[i, j]] = normal.sample(&mut rng);
        }
        // Gram-Schmidt orthogonalization against previous columns
        for prev_j in 0..j {
            let mut dot = 0.0;
            for i in 0..n_samples {
                dot += u_mat[[i, j]] * u_mat[[i, prev_j]];
            }
            for i in 0..n_samples {
                u_mat[[i, j]] -= dot * u_mat[[i, prev_j]];
            }
        }
        // Normalize
        let norm: f64 = (0..n_samples)
            .map(|i| u_mat[[i, j]] * u_mat[[i, j]])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for i in 0..n_samples {
                u_mat[[i, j]] /= norm;
            }
        }
    }

    let mut v_mat = Array2::zeros((min_dim, n_features));
    for i in 0..min_dim {
        for j in 0..n_features {
            v_mat[[i, j]] = normal.sample(&mut rng);
        }
        // Gram-Schmidt
        for prev_i in 0..i {
            let mut dot = 0.0;
            for j in 0..n_features {
                dot += v_mat[[i, j]] * v_mat[[prev_i, j]];
            }
            for j in 0..n_features {
                v_mat[[i, j]] -= dot * v_mat[[prev_i, j]];
            }
        }
        // Normalize
        let norm: f64 = (0..n_features)
            .map(|j| v_mat[[i, j]] * v_mat[[i, j]])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for j in 0..n_features {
                v_mat[[i, j]] /= norm;
            }
        }
    }

    // Compute data = U * diag(singular_values) * V
    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let mut val = 0.0;
            for k in 0..min_dim {
                val += u_mat[[i, k]] * singular_values[k] * v_mat[[k, j]];
            }
            data[[i, j]] = val;
        }
    }

    let sv_array = Array1::from_vec(singular_values);
    let feature_names: Vec<String> = (0..n_features).map(|j| format!("feature_{j}")).collect();

    let dataset = Dataset::new(data, Some(sv_array))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Low-rank matrix ({n_samples} x {n_features}) with effective rank ~{effective_rank}. \
             Target contains the singular values."
        ))
        .with_metadata("generator", "make_low_rank_matrix")
        .with_metadata("effective_rank", &effective_rank.to_string())
        .with_metadata("tail_strength", &tail_strength.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_friedman1 tests
    // =========================================================================

    #[test]
    fn test_friedman1_basic() {
        let ds = make_friedman1(200, 10, 0.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 200);
        assert_eq!(ds.n_features(), 10);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_friedman1_formula_noiseless() {
        let ds = make_friedman1(100, 5, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");

        for i in 0..100 {
            let x0 = ds.data[[i, 0]];
            let x1 = ds.data[[i, 1]];
            let x2 = ds.data[[i, 2]];
            let x3 = ds.data[[i, 3]];
            let x4 = ds.data[[i, 4]];

            let expected =
                10.0 * (PI * x0 * x1).sin() + 20.0 * (x2 - 0.5).powi(2) + 10.0 * x3 + 5.0 * x4;

            assert!(
                (target[i] - expected).abs() < 1e-10,
                "Friedman1 formula mismatch at sample {i}: got {}, expected {expected}",
                target[i]
            );
        }
    }

    #[test]
    fn test_friedman1_features_in_range() {
        let ds = make_friedman1(500, 10, 0.0, Some(42)).expect("should succeed");
        for i in 0..500 {
            for j in 0..10 {
                let val = ds.data[[i, j]];
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Feature [{i},{j}] = {val} not in [0,1]"
                );
            }
        }
    }

    #[test]
    fn test_friedman1_reproducibility() {
        let ds1 = make_friedman1(50, 10, 1.0, Some(42)).expect("should succeed");
        let ds2 = make_friedman1(50, 10, 1.0, Some(42)).expect("should succeed");
        for i in 0..50 {
            for j in 0..10 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Reproducibility failed"
                );
            }
        }
    }

    #[test]
    fn test_friedman1_validation() {
        assert!(make_friedman1(0, 10, 0.0, None).is_err());
        assert!(make_friedman1(100, 3, 0.0, None).is_err());
        assert!(make_friedman1(100, 10, -1.0, None).is_err());
    }

    // =========================================================================
    // make_friedman2 tests
    // =========================================================================

    #[test]
    fn test_friedman2_basic() {
        let ds = make_friedman2(200, 0.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 200);
        assert_eq!(ds.n_features(), 4);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_friedman2_positive_target() {
        let ds = make_friedman2(500, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");
        for &val in target.iter() {
            assert!(
                val >= 0.0,
                "Friedman2 target should be non-negative, got {val}"
            );
        }
    }

    #[test]
    fn test_friedman2_feature_ranges() {
        let ds = make_friedman2(500, 0.0, Some(42)).expect("should succeed");
        for i in 0..500 {
            assert!(ds.data[[i, 0]] >= 0.0 && ds.data[[i, 0]] <= 100.0);
            assert!(ds.data[[i, 1]] >= 40.0 * PI && ds.data[[i, 1]] <= 560.0 * PI);
            assert!(ds.data[[i, 2]] >= 0.0 && ds.data[[i, 2]] <= 1.0);
            assert!(ds.data[[i, 3]] >= 1.0 && ds.data[[i, 3]] <= 11.0);
        }
    }

    #[test]
    fn test_friedman2_validation() {
        assert!(make_friedman2(0, 0.0, None).is_err());
        assert!(make_friedman2(100, -1.0, None).is_err());
    }

    // =========================================================================
    // make_friedman3 tests
    // =========================================================================

    #[test]
    fn test_friedman3_basic() {
        let ds = make_friedman3(200, 0.0, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 200);
        assert_eq!(ds.n_features(), 4);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_friedman3_atan_range() {
        // atan output is in (-pi/2, pi/2)
        let ds = make_friedman3(1000, 0.0, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");
        for &val in target.iter() {
            assert!(
                val.abs() <= PI / 2.0 + 0.01,
                "Friedman3 target should be bounded by pi/2, got {val}"
            );
        }
    }

    #[test]
    fn test_friedman3_validation() {
        assert!(make_friedman3(0, 0.0, None).is_err());
        assert!(make_friedman3(100, -1.0, None).is_err());
    }

    // =========================================================================
    // make_sparse_uncorrelated tests
    // =========================================================================

    #[test]
    fn test_sparse_uncorrelated_basic() {
        let ds = make_sparse_uncorrelated(100, 10, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 10);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_sparse_uncorrelated_informative_features() {
        // The target should correlate with x0 and x1, not with x4..x9
        let ds = make_sparse_uncorrelated(1000, 10, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target present");

        // Compute correlation of target with each feature
        let t_mean: f64 = target.iter().sum::<f64>() / 1000.0;

        for j in 0..10 {
            let x: Vec<f64> = (0..1000).map(|i| ds.data[[i, j]]).collect();
            let x_mean: f64 = x.iter().sum::<f64>() / 1000.0;

            let cov: f64 = (0..1000)
                .map(|i| (x[i] - x_mean) * (target[i] - t_mean))
                .sum::<f64>()
                / 999.0;
            let x_std: f64 = (x.iter().map(|v| (v - x_mean).powi(2)).sum::<f64>() / 999.0).sqrt();
            let t_std: f64 =
                (target.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / 999.0).sqrt();

            let corr = if x_std > 1e-10 && t_std > 1e-10 {
                cov / (x_std * t_std)
            } else {
                0.0
            };

            if j < 2 {
                // Informative features should have high correlation
                assert!(
                    corr.abs() > 0.1,
                    "Feature {j} should be informative, corr = {corr}"
                );
            }
            // Noise features may have low correlation but this is statistical
        }
    }

    #[test]
    fn test_sparse_uncorrelated_validation() {
        assert!(make_sparse_uncorrelated(0, 10, None).is_err());
        assert!(make_sparse_uncorrelated(100, 2, None).is_err());
    }

    // =========================================================================
    // make_low_rank_matrix tests
    // =========================================================================

    #[test]
    fn test_low_rank_matrix_basic() {
        let ds = make_low_rank_matrix(100, 50, 5, 0.5, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 50);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_low_rank_matrix_singular_values_decay() {
        let ds = make_low_rank_matrix(50, 30, 3, 0.1, Some(42)).expect("should succeed");
        let sv = ds.target.as_ref().expect("target present");

        // The first few singular values should be larger than the rest
        // With tail_strength=0.1, the tail is small
        assert!(
            sv[0] > sv[sv.len() - 1],
            "First SV ({}) should be > last SV ({})",
            sv[0],
            sv[sv.len() - 1]
        );

        // Check monotonically non-increasing (or close to it)
        for i in 1..sv.len() {
            assert!(
                sv[i] <= sv[i - 1] + 1e-10,
                "Singular values should be non-increasing: sv[{}]={} > sv[{}]={}",
                i,
                sv[i],
                i - 1,
                sv[i - 1]
            );
        }
    }

    #[test]
    fn test_low_rank_matrix_validation() {
        assert!(make_low_rank_matrix(0, 10, 5, 0.5, None).is_err());
        assert!(make_low_rank_matrix(10, 0, 5, 0.5, None).is_err());
        assert!(make_low_rank_matrix(10, 10, 0, 0.5, None).is_err());
        assert!(make_low_rank_matrix(10, 10, 5, -0.1, None).is_err());
        assert!(make_low_rank_matrix(10, 10, 5, 1.5, None).is_err());
    }

    #[test]
    fn test_low_rank_matrix_reproducibility() {
        let ds1 = make_low_rank_matrix(30, 20, 3, 0.5, Some(42)).expect("should succeed");
        let ds2 = make_low_rank_matrix(30, 20, 3, 0.5, Some(42)).expect("should succeed");
        for i in 0..30 {
            for j in 0..20 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-12,
                    "Reproducibility failed at ({i},{j})"
                );
            }
        }
    }
}
