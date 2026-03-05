//! Regression benchmark dataset generators.
//!
//! This module provides well-known regression benchmark datasets as synthetic
//! generators with configurable noise:
//!
//! - [`friedman1`]              – Friedman #1 (10-D, 5 active features).
//! - [`friedman2`]              – Friedman #2 (4-D, non-linear formula).
//! - [`friedman3`]              – Friedman #3 (4-D, arctan formula).
//! - [`boston_housing_like`]    – Boston-housing-inspired 13-feature dataset.
//! - [`california_housing_like`] – California-housing-inspired 8-feature dataset.
//! - [`diabetes_like`]          – Diabetes-benchmark-inspired 10-feature dataset.
//!
//! All functions return `(Array2<f64>, Array1<f64>)` — design matrix and target vector —
//! without depending on any external dataset files.  All generators are fully
//! deterministic given a seed.

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

/// Validate common regression generator arguments.
fn validate_common(n: usize, noise: f64, fn_name: &str) -> Result<()> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: n must be >= 1"
        )));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{fn_name}: noise must be >= 0, got {noise}"
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// friedman1
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #1 benchmark regression dataset.
///
/// The dataset uses 10 independent features uniformly distributed on [0, 1].
/// Only the first 5 contribute to the response:
///
/// ```text
/// y = 10 * sin(π x₀ x₁) + 20 (x₂ − 0.5)² + 10 x₃ + 5 x₄ + N(0, noise²)
/// ```
///
/// Reference: Friedman, J.H. (1991). *Multivariate Adaptive Regression Splines*.
/// Annals of Statistics 19(1), 1–67.
///
/// # Arguments
///
/// * `n`     – Number of samples (must be ≥ 1).
/// * `noise` – Gaussian noise std-dev added to the target (≥ 0).
/// * `seed`  – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 10)` and `y` is `(n,)`.
///
/// # Errors
///
/// Returns an error if `n == 0` or `noise < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::friedman1;
///
/// let (x, y) = friedman1(200, 0.5, 42).expect("friedman1 failed");
/// assert_eq!(x.shape(), &[200, 10]);
/// assert_eq!(y.len(), 200);
/// ```
pub fn friedman1(n: usize, noise: f64, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    validate_common(n, noise, "friedman1")?;

    let n_features = 10usize;
    let mut rng = make_rng(seed);

    let uniform = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform dist failed: {e}"))
    })?;
    let normal_opt = if noise > 0.0 {
        Some(
            scirs2_core::random::Normal::new(0.0_f64, noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist failed: {e}"))
            })?,
        )
    } else {
        None
    };

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        for j in 0..n_features {
            x[[i, j]] = uniform.sample(&mut rng);
        }
        let x0 = x[[i, 0]];
        let x1 = x[[i, 1]];
        let x2 = x[[i, 2]];
        let x3 = x[[i, 3]];
        let x4 = x[[i, 4]];
        let eps = normal_opt
            .as_ref()
            .map(|d| d.sample(&mut rng))
            .unwrap_or(0.0);
        y[i] =
            10.0 * (PI * x0 * x1).sin() + 20.0 * (x2 - 0.5).powi(2) + 10.0 * x3 + 5.0 * x4
                + eps;
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// friedman2
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #2 benchmark regression dataset.
///
/// Four features drawn from non-overlapping intervals:
///
/// ```text
/// x₀ ~ U[0, 100]          x₁ ~ U[40π, 560π]
/// x₂ ~ U[0, 1]            x₃ ~ U[1, 11]
///
/// y = √(x₀² + (x₁ x₂ − 1 / (x₁ x₃))²) + N(0, noise²)
/// ```
///
/// Reference: Friedman (1991).
///
/// # Arguments
///
/// * `n`     – Number of samples (must be ≥ 1).
/// * `noise` – Gaussian noise std-dev (≥ 0).
/// * `seed`  – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 4)` and `y` is `(n,)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::friedman2;
///
/// let (x, y) = friedman2(150, 0.0, 42).expect("friedman2 failed");
/// assert_eq!(x.shape(), &[150, 4]);
/// ```
pub fn friedman2(n: usize, noise: f64, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    validate_common(n, noise, "friedman2")?;

    let n_features = 4usize;
    let mut rng = make_rng(seed);

    let u0 = scirs2_core::random::Uniform::new(0.0_f64, 100.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u0 failed: {e}"))
    })?;
    let u1 = scirs2_core::random::Uniform::new(40.0 * PI, 560.0 * PI).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u1 failed: {e}"))
    })?;
    let u2 = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u2 failed: {e}"))
    })?;
    let u3 = scirs2_core::random::Uniform::new(1.0_f64, 11.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u3 failed: {e}"))
    })?;
    let normal_opt = if noise > 0.0 {
        Some(
            scirs2_core::random::Normal::new(0.0_f64, noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist failed: {e}"))
            })?,
        )
    } else {
        None
    };

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x0 = u0.sample(&mut rng);
        let x1 = u1.sample(&mut rng);
        let x2 = u2.sample(&mut rng);
        let x3 = u3.sample(&mut rng);
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        x[[i, 2]] = x2;
        x[[i, 3]] = x3;

        let denom = x1 * x3;
        let inner = if denom.abs() > 1e-15 {
            x1 * x2 - 1.0 / denom
        } else {
            x1 * x2
        };
        let eps = normal_opt
            .as_ref()
            .map(|d| d.sample(&mut rng))
            .unwrap_or(0.0);
        y[i] = (x0 * x0 + inner * inner).sqrt() + eps;
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// friedman3
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #3 benchmark regression dataset.
///
/// Same feature distributions as Friedman #2, different response:
///
/// ```text
/// y = atan((x₁ x₂ − 1 / (x₁ x₃)) / x₀) + N(0, noise²)
/// ```
///
/// Reference: Friedman (1991).
///
/// # Arguments
///
/// * `n`     – Number of samples (must be ≥ 1).
/// * `noise` – Gaussian noise std-dev (≥ 0).
/// * `seed`  – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 4)` and `y` is `(n,)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::friedman3;
///
/// let (x, y) = friedman3(150, 0.0, 42).expect("friedman3 failed");
/// assert_eq!(x.shape(), &[150, 4]);
/// // target bounded by ±π/2
/// for &v in y.iter() {
///     assert!(v.abs() <= std::f64::consts::PI / 2.0 + 0.01);
/// }
/// ```
pub fn friedman3(n: usize, noise: f64, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    validate_common(n, noise, "friedman3")?;

    let n_features = 4usize;
    let mut rng = make_rng(seed);

    let u0 = scirs2_core::random::Uniform::new(0.0_f64, 100.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u0 failed: {e}"))
    })?;
    let u1 = scirs2_core::random::Uniform::new(40.0 * PI, 560.0 * PI).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u1 failed: {e}"))
    })?;
    let u2 = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u2 failed: {e}"))
    })?;
    let u3 = scirs2_core::random::Uniform::new(1.0_f64, 11.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform u3 failed: {e}"))
    })?;
    let normal_opt = if noise > 0.0 {
        Some(
            scirs2_core::random::Normal::new(0.0_f64, noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Normal dist failed: {e}"))
            })?,
        )
    } else {
        None
    };

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x0 = u0.sample(&mut rng);
        let x1 = u1.sample(&mut rng);
        let x2 = u2.sample(&mut rng);
        let x3 = u3.sample(&mut rng);
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        x[[i, 2]] = x2;
        x[[i, 3]] = x3;

        let denom = x1 * x3;
        let inner = if denom.abs() > 1e-15 {
            x1 * x2 - 1.0 / denom
        } else {
            x1 * x2
        };
        let response = if x0.abs() > 1e-15 {
            (inner / x0).atan()
        } else {
            PI / 2.0 * inner.signum()
        };
        let eps = normal_opt
            .as_ref()
            .map(|d| d.sample(&mut rng))
            .unwrap_or(0.0);
        y[i] = response + eps;
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// boston_housing_like
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Boston-housing-inspired synthetic regression dataset.
///
/// The generator produces 13 features inspired by the original Boston housing
/// variables (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO,
/// B, LSTAT).  A realistic target (MEDV proxy) is constructed as a non-linear
/// combination that echoes the statistical relationships in the original data.
///
/// # Arguments
///
/// * `n`   – Number of samples (must be ≥ 1).
/// * `rng` – Mutable reference to a `StdRng` (caller provides seed).
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 13)` and `y` is `(n,)` with approximate value
/// in [5, 50].
///
/// # Internal note
///
/// Not a public RNG-accepting function; the public API is [`boston_housing_like`].
fn boston_housing_like_inner(n: usize, rng: &mut StdRng) -> Result<(Array2<f64>, Array1<f64>)> {
    // Feature definitions: (distribution_type, lo, hi)
    // We draw each feature independently; correlations emerge through the formula.
    let n_features = 13usize;

    // Per-feature distributions (min, max) – all Uniform for simplicity
    let feature_ranges: [(f64, f64); 13] = [
        (0.006, 89.0),  // 0 CRIM  – per capita crime rate
        (0.0, 100.0),   // 1 ZN    – % residential land zoned large lots
        (0.46, 27.74),  // 2 INDUS – % non-retail business acres
        (0.0, 1.0),     // 3 CHAS  – Charles River dummy (binary; will round)
        (0.385, 0.871), // 4 NOX   – nitric oxide concentration
        (3.56, 8.78),   // 5 RM    – avg rooms per dwelling
        (2.9, 100.0),   // 6 AGE   – % owner-occupied built before 1940
        (1.13, 12.13),  // 7 DIS   – weighted distances to employment centres
        (1.0, 24.0),    // 8 RAD   – accessibility to radial highways
        (187.0, 711.0), // 9 TAX   – full-value property tax rate / $10k
        (12.6, 22.0),   // 10 PTRATIO – pupil–teacher ratio
        (0.32, 396.9),  // 11 B    – 1000(Bk − 0.63)²
        (1.73, 37.97),  // 12 LSTAT – % lower-status population
    ];

    let mut distributions = Vec::with_capacity(n_features);
    for (lo, hi) in feature_ranges.iter() {
        let dist = scirs2_core::random::Uniform::new(*lo, *hi).map_err(|e| {
            DatasetsError::ComputationError(format!("Boston feature dist failed: {e}"))
        })?;
        distributions.push(dist);
    }

    let noise_dist = scirs2_core::random::Normal::new(0.0_f64, 2.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Boston noise dist failed: {e}"))
    })?;

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        for j in 0..n_features {
            x[[i, j]] = distributions[j].sample(rng);
        }
        // CHAS: round to 0 or 1
        x[[i, 3]] = if x[[i, 3]] > 0.5 { 1.0 } else { 0.0 };

        let crim = x[[i, 0]];
        let nox = x[[i, 4]];
        let rm = x[[i, 5]];
        let age = x[[i, 6]];
        let dis = x[[i, 7]];
        let tax = x[[i, 9]];
        let ptratio = x[[i, 10]];
        let lstat = x[[i, 12]];

        // Mimick a non-linear housing price formula
        let medv = 22.0
            - 0.1 * crim.ln().max(-5.0)
            + 5.0 * (rm - 6.0)
            - 0.05 * age
            + 1.5 * dis.ln()
            - 0.01 * tax
            - 0.5 * ptratio
            - 10.0 * (lstat / 100.0).sqrt()
            - 3.0 * nox
            + noise_dist.sample(rng);

        y[i] = medv.max(5.0).min(50.0);
    }

    Ok((x, y))
}

/// Generate a Boston-housing-inspired synthetic regression dataset.
///
/// 13 features mimicking the statistical characteristics of the original
/// Boston Housing dataset (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS,
/// RAD, TAX, PTRATIO, B, LSTAT).  Target approximates median house value
/// (clamped to [5, 50]).
///
/// # Arguments
///
/// * `n`    – Number of samples (must be ≥ 1).
/// * `seed` – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 13)` and `y` is `(n,)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::boston_housing_like;
///
/// let (x, y) = boston_housing_like(200, 42).expect("boston failed");
/// assert_eq!(x.shape(), &[200, 13]);
/// assert_eq!(y.len(), 200);
/// ```
pub fn boston_housing_like(n: usize, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "boston_housing_like: n must be >= 1".to_string(),
        ));
    }
    let mut rng = make_rng(seed);
    boston_housing_like_inner(n, &mut rng)
}

// ─────────────────────────────────────────────────────────────────────────────
// california_housing_like
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a California-housing-inspired synthetic regression dataset.
///
/// 8 features inspired by the California Housing dataset:
///
/// | Feature | Description                            | Range         |
/// |---------|----------------------------------------|---------------|
/// | 0       | MedInc – median income (10k USD)       | [0.5, 15.0]   |
/// | 1       | HouseAge – median house age            | [1, 52]       |
/// | 2       | AveRooms – avg rooms per household     | [1.0, 10.0]   |
/// | 3       | AveBedrms – avg bedrooms per household | [0.8, 5.0]    |
/// | 4       | Population – block population         | [3, 35000]    |
/// | 5       | AveOccup – avg occupancy               | [1.0, 12.0]   |
/// | 6       | Latitude                               | [32.5, 42.0]  |
/// | 7       | Longitude                              | [-124.4, -114.3] |
///
/// Target (MedHouseVal proxy) ranges approximately in [0.15, 5.0] (×$100k).
///
/// # Arguments
///
/// * `n`    – Number of samples (must be ≥ 1).
/// * `seed` – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 8)` and `y` is `(n,)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::california_housing_like;
///
/// let (x, y) = california_housing_like(300, 42).expect("california failed");
/// assert_eq!(x.shape(), &[300, 8]);
/// assert_eq!(y.len(), 300);
/// ```
pub fn california_housing_like(n: usize, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "california_housing_like: n must be >= 1".to_string(),
        ));
    }

    let n_features = 8usize;
    let mut rng = make_rng(seed);

    let feature_ranges: [(f64, f64); 8] = [
        (0.5, 15.0),       // 0 MedInc
        (1.0, 52.0),       // 1 HouseAge
        (1.0, 10.0),       // 2 AveRooms
        (0.8, 5.0),        // 3 AveBedrms
        (3.0, 35000.0),    // 4 Population
        (1.0, 12.0),       // 5 AveOccup
        (32.5, 42.0),      // 6 Latitude
        (-124.4, -114.3),  // 7 Longitude
    ];

    let mut distributions = Vec::with_capacity(n_features);
    for (lo, hi) in feature_ranges.iter() {
        let dist = scirs2_core::random::Uniform::new(*lo, *hi).map_err(|e| {
            DatasetsError::ComputationError(format!("California feature dist failed: {e}"))
        })?;
        distributions.push(dist);
    }

    let noise_dist = scirs2_core::random::Normal::new(0.0_f64, 0.3_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("California noise dist failed: {e}"))
    })?;

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        for j in 0..n_features {
            x[[i, j]] = distributions[j].sample(&mut rng);
        }

        let med_inc = x[[i, 0]];
        let house_age = x[[i, 1]];
        let ave_rooms = x[[i, 2]];
        let ave_bedrms = x[[i, 3]];
        let population = x[[i, 4]];
        let ave_occup = x[[i, 5]];

        // Simplified but realistic formula
        let base_value = 0.5 * med_inc
            + 0.01 * house_age
            + 0.3 * (ave_rooms - ave_bedrms)
            - 0.05 * (population / 10000.0)
            - 0.1 * ave_occup
            + noise_dist.sample(&mut rng);

        y[i] = base_value.max(0.15).min(5.0);
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// diabetes_like
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a diabetes-benchmark-inspired 10-feature regression dataset.
///
/// The original Efron et al. (2004) diabetes dataset has 442 patients and 10
/// baseline features.  This generator reproduces the statistical character:
///
/// | Feature | Proxy description        | Distribution        |
/// |---------|--------------------------|---------------------|
/// | 0       | Age (normalised)         | N(0, 1) clipped     |
/// | 1       | Sex (binary)             | Bernoulli(0.5)      |
/// | 2       | BMI (normalised)         | N(0, 1)             |
/// | 3       | BP (avg blood pressure)  | N(0, 1)             |
/// | 4–9     | Six serum measurements   | N(0, 1) (raw)       |
///
/// The target is a non-linear combination of the features and reflects the
/// disease progression proxy (a quantitative measure 1 year after baseline),
/// scaled approximately to [25, 350].
///
/// # Arguments
///
/// * `n`     – Number of samples (must be ≥ 1).
/// * `noise` – Gaussian noise std-dev added to the target (≥ 0).
/// * `seed`  – Random seed.
///
/// # Returns
///
/// `(X, y)` where `X` is `(n, 10)` and `y` is `(n,)`.
///
/// # Reference
///
/// Efron, B., Hastie, T., Johnstone, I., Tibshirani, R. (2004). *Least Angle
/// Regression*. Annals of Statistics 32(2), 407–451.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::regression_benchmarks::diabetes_like;
///
/// let (x, y) = diabetes_like(200, 25.0, 42).expect("diabetes failed");
/// assert_eq!(x.shape(), &[200, 10]);
/// assert_eq!(y.len(), 200);
/// ```
pub fn diabetes_like(n: usize, noise: f64, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    validate_common(n, noise, "diabetes_like")?;

    let n_features = 10usize;
    let mut rng = make_rng(seed);

    let normal_std1 = scirs2_core::random::Normal::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal(0,1) dist failed: {e}"))
    })?;

    let noise_dist_opt = if noise > 0.0 {
        Some(
            scirs2_core::random::Normal::new(0.0_f64, noise).map_err(|e| {
                DatasetsError::ComputationError(format!("Noise dist failed: {e}"))
            })?,
        )
    } else {
        None
    };

    let uniform01 = scirs2_core::random::Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform dist failed: {e}"))
    })?;

    let mut x = Array2::zeros((n, n_features));
    let mut y = Array1::zeros(n);

    // True linear coefficients (inspired by Efron et al. Table 1)
    // Features: age, sex, bmi, bp, s1-s6 (but we only have 6 serum → 4+6=10)
    let coefs: [f64; 10] = [-10.0, -239.8, 519.8, 324.4, -792.2, 476.7, 101.0, 177.1, 751.3, 67.6];
    // We apply a scaled version to keep output in a reasonable range
    let coef_scale = 0.3_f64;

    for i in 0..n {
        // Feature 0: age ~ N(0,1) (normalised)
        x[[i, 0]] = normal_std1.sample(&mut rng).max(-3.0).min(3.0);
        // Feature 1: sex ~ Bernoulli(0.5), mapped to {-1, 1}
        x[[i, 1]] = if uniform01.sample(&mut rng) >= 0.5 { 1.0 } else { -1.0 };
        // Features 2-9: ~ N(0, 1)
        for j in 2..n_features {
            x[[i, j]] = normal_std1.sample(&mut rng);
        }

        // Target: non-linear mix (squared BMI dominates)
        let bmi = x[[i, 2]];
        let bp = x[[i, 3]];
        let s1 = x[[i, 4]];
        let s3 = x[[i, 6]];

        let mut target = 152.0; // intercept (approximate mean)
        for j in 0..n_features {
            target += coef_scale * coefs[j] * x[[i, j]];
        }
        // Add interaction terms (BMI × BP, BMI² − reflecting disease progression)
        target += 50.0 * bmi * bp - 30.0 * bmi.powi(2) + 20.0 * s1 * s3;

        let eps = noise_dist_opt
            .as_ref()
            .map(|d| d.sample(&mut rng))
            .unwrap_or(0.0);
        y[i] = (target + eps).max(25.0).min(350.0);
    }

    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── friedman1 ────────────────────────────────────────────────────────────

    #[test]
    fn test_friedman1_shape() {
        let (x, y) = friedman1(200, 0.5, 42).expect("friedman1");
        assert_eq!(x.shape(), &[200, 10]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_friedman1_features_in_range() {
        let (x, _) = friedman1(300, 0.0, 1).expect("friedman1 range");
        for v in x.iter() {
            assert!(
                (0.0..=1.0).contains(v),
                "feature value {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_friedman1_formula_noiseless() {
        let (x, y) = friedman1(100, 0.0, 42).expect("friedman1 formula");
        for i in 0..100 {
            let expected = 10.0 * (PI * x[[i, 0]] * x[[i, 1]]).sin()
                + 20.0 * (x[[i, 2]] - 0.5).powi(2)
                + 10.0 * x[[i, 3]]
                + 5.0 * x[[i, 4]];
            assert!(
                (y[i] - expected).abs() < 1e-10,
                "friedman1 formula mismatch at {i}: got {}, expected {}",
                y[i],
                expected
            );
        }
    }

    #[test]
    fn test_friedman1_determinism() {
        let (x1, y1) = friedman1(50, 1.0, 42).expect("f1a");
        let (x2, y2) = friedman1(50, 1.0, 42).expect("f1b");
        for i in 0..50 {
            for j in 0..10 {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-14);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_friedman1_error_n_zero() {
        assert!(friedman1(0, 1.0, 1).is_err());
    }

    #[test]
    fn test_friedman1_error_negative_noise() {
        assert!(friedman1(100, -0.1, 1).is_err());
    }

    // ── friedman2 ────────────────────────────────────────────────────────────

    #[test]
    fn test_friedman2_shape() {
        let (x, y) = friedman2(200, 0.0, 42).expect("friedman2");
        assert_eq!(x.shape(), &[200, 4]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_friedman2_positive_target() {
        let (_, y) = friedman2(500, 0.0, 7).expect("friedman2 positive");
        for &v in y.iter() {
            assert!(v >= 0.0, "friedman2 target should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_friedman2_feature_ranges() {
        let (x, _) = friedman2(400, 0.0, 3).expect("friedman2 ranges");
        for i in 0..400 {
            assert!(x[[i, 0]] >= 0.0 && x[[i, 0]] <= 100.0);
            assert!(x[[i, 1]] >= 40.0 * PI && x[[i, 1]] <= 560.0 * PI);
            assert!(x[[i, 2]] >= 0.0 && x[[i, 2]] <= 1.0);
            assert!(x[[i, 3]] >= 1.0 && x[[i, 3]] <= 11.0);
        }
    }

    #[test]
    fn test_friedman2_error_n_zero() {
        assert!(friedman2(0, 0.0, 1).is_err());
    }

    // ── friedman3 ────────────────────────────────────────────────────────────

    #[test]
    fn test_friedman3_shape() {
        let (x, y) = friedman3(200, 0.0, 42).expect("friedman3");
        assert_eq!(x.shape(), &[200, 4]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_friedman3_atan_range() {
        let (_, y) = friedman3(1000, 0.0, 42).expect("friedman3 atan");
        for &v in y.iter() {
            assert!(
                v.abs() <= PI / 2.0 + 1e-9,
                "friedman3 target {v} exceeds pi/2"
            );
        }
    }

    #[test]
    fn test_friedman3_error_n_zero() {
        assert!(friedman3(0, 0.0, 1).is_err());
    }

    #[test]
    fn test_friedman3_error_negative_noise() {
        assert!(friedman3(100, -1.0, 1).is_err());
    }

    // ── boston_housing_like ──────────────────────────────────────────────────

    #[test]
    fn test_boston_shape() {
        let (x, y) = boston_housing_like(200, 42).expect("boston");
        assert_eq!(x.shape(), &[200, 13]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_boston_target_in_range() {
        let (_, y) = boston_housing_like(300, 7).expect("boston target range");
        for &v in y.iter() {
            assert!(
                (5.0..=50.0).contains(&v),
                "boston target {v} outside [5, 50]"
            );
        }
    }

    #[test]
    fn test_boston_chas_binary() {
        let (x, _) = boston_housing_like(200, 1).expect("boston chas");
        for i in 0..200 {
            let chas = x[[i, 3]];
            assert!(
                chas == 0.0 || chas == 1.0,
                "CHAS feature should be binary, got {chas}"
            );
        }
    }

    #[test]
    fn test_boston_determinism() {
        let (x1, y1) = boston_housing_like(50, 42).expect("b1");
        let (x2, y2) = boston_housing_like(50, 42).expect("b2");
        for i in 0..50 {
            for j in 0..13 {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-14);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_boston_error_n_zero() {
        assert!(boston_housing_like(0, 1).is_err());
    }

    // ── california_housing_like ──────────────────────────────────────────────

    #[test]
    fn test_california_shape() {
        let (x, y) = california_housing_like(300, 42).expect("california");
        assert_eq!(x.shape(), &[300, 8]);
        assert_eq!(y.len(), 300);
    }

    #[test]
    fn test_california_target_in_range() {
        let (_, y) = california_housing_like(500, 7).expect("california target");
        for &v in y.iter() {
            assert!(
                (0.15..=5.0).contains(&v),
                "california target {v} outside [0.15, 5.0]"
            );
        }
    }

    #[test]
    fn test_california_feature_ranges() {
        let (x, _) = california_housing_like(200, 1).expect("california features");
        for i in 0..200 {
            assert!(x[[i, 0]] >= 0.5 && x[[i, 0]] <= 15.0, "MedInc OOB");
            assert!(x[[i, 6]] >= 32.5 && x[[i, 6]] <= 42.0, "Latitude OOB");
            assert!(x[[i, 7]] >= -124.4 && x[[i, 7]] <= -114.3, "Longitude OOB");
        }
    }

    #[test]
    fn test_california_determinism() {
        let (x1, y1) = california_housing_like(40, 42).expect("c1");
        let (x2, y2) = california_housing_like(40, 42).expect("c2");
        for i in 0..40 {
            for j in 0..8 {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-14);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_california_error_n_zero() {
        assert!(california_housing_like(0, 1).is_err());
    }

    // ── diabetes_like ────────────────────────────────────────────────────────

    #[test]
    fn test_diabetes_shape() {
        let (x, y) = diabetes_like(200, 25.0, 42).expect("diabetes");
        assert_eq!(x.shape(), &[200, 10]);
        assert_eq!(y.len(), 200);
    }

    #[test]
    fn test_diabetes_target_in_range() {
        let (_, y) = diabetes_like(500, 0.0, 7).expect("diabetes range");
        for &v in y.iter() {
            assert!(
                (25.0..=350.0).contains(&v),
                "diabetes target {v} outside [25, 350]"
            );
        }
    }

    #[test]
    fn test_diabetes_sex_feature_binary() {
        let (x, _) = diabetes_like(200, 0.0, 1).expect("diabetes sex");
        for i in 0..200 {
            let sex = x[[i, 1]];
            assert!(
                sex == 1.0 || sex == -1.0,
                "sex feature should be ±1, got {sex}"
            );
        }
    }

    #[test]
    fn test_diabetes_age_clipped() {
        let (x, _) = diabetes_like(200, 0.0, 3).expect("diabetes age");
        for i in 0..200 {
            let age = x[[i, 0]];
            assert!(
                (-3.0..=3.0).contains(&age),
                "normalised age {age} outside [-3, 3]"
            );
        }
    }

    #[test]
    fn test_diabetes_determinism() {
        let (x1, y1) = diabetes_like(50, 10.0, 42).expect("d1");
        let (x2, y2) = diabetes_like(50, 10.0, 42).expect("d2");
        for i in 0..50 {
            for j in 0..10 {
                assert!((x1[[i, j]] - x2[[i, j]]).abs() < 1e-14);
            }
            assert!((y1[i] - y2[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_diabetes_error_n_zero() {
        assert!(diabetes_like(0, 10.0, 1).is_err());
    }

    #[test]
    fn test_diabetes_error_negative_noise() {
        assert!(diabetes_like(100, -1.0, 1).is_err());
    }
}
