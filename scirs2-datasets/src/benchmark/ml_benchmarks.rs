//! ML benchmark dataset generators with raw-vector output and explicit RNG parameter.
//!
//! These functions return simple `Vec<Vec<f64>>` / `Vec<f64>` / `Vec<usize>` tuples
//! rather than the [`Dataset`](crate::utils::Dataset) wrapper, making them easy to use
//! in benchmarking and unit-testing contexts where low overhead is desired.
//!
//! All randomness is driven through an explicit `rng: &mut impl Rng` parameter so
//! callers can control reproducibility.
//!
//! # Functions
//!
//! | Function | Target type | Description |
//! |---|---|---|
//! | [`friedman1_bench`] | `Vec<f64>` | Friedman #1 regression |
//! | [`friedman2_bench`] | `Vec<f64>` | Friedman #2 regression |
//! | [`moons_bench`] | `Vec<usize>` | Two interleaving half-circles |
//! | [`circles_bench`] | `Vec<usize>` | Two concentric circles |
//! | [`swiss_roll_bench`] | `Vec<f64>` | Swiss roll 3-D manifold |
//! | [`s_curve_bench`] | `Vec<f64>` | S-shaped 3-D manifold |
//! | [`imbalanced_classification`] | `Vec<usize>` | Skewed class distribution |
//! | [`concept_drift`] | `Vec<usize>` | Concept-drifting decision boundary |

use crate::error::{DatasetsError, Result};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::{Distribution, Uniform};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample from a standard normal distribution (Box-Muller transform).
fn sample_normal(rng: &mut impl Rng) -> f64 {
    let u1 = rng.random::<f64>().max(f64::EPSILON);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Sample a vector of `n` standard-normal values.
fn randn_vec(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..n).map(|_| sample_normal(rng)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Friedman #1
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #1 regression benchmark dataset.
///
/// The target is:
/// ```text
/// y = 10·sin(π·x₁·x₂) + 20·(x₃ - 0.5)² + 10·x₄ + 5·x₅ + ε
/// ```
/// where `xᵢ ~ Uniform(0,1)` and `ε ~ N(0, noise²)`.
/// Any additional features beyond the first 5 are pure noise.
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × n_features` and `y` has length `n_samples`.
pub fn friedman1_bench(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "friedman1_bench: n_samples must be > 0".into(),
        ));
    }
    if n_features < 5 {
        return Err(DatasetsError::InvalidFormat(
            "friedman1_bench: n_features must be >= 5".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "friedman1_bench: noise must be >= 0".into(),
        ));
    }

    let uni = Uniform::new(0.0f64, 1.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("friedman1_bench: uniform init failed: {e}"))
    })?;

    let mut x_all = Vec::with_capacity(n_samples);
    let mut y_all = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let row: Vec<f64> = (0..n_features).map(|_| uni.sample(rng)).collect();
        let y_val = 10.0 * (PI * row[0] * row[1]).sin()
            + 20.0 * (row[2] - 0.5).powi(2)
            + 10.0 * row[3]
            + 5.0 * row[4]
            + noise * sample_normal(rng);
        x_all.push(row);
        y_all.push(y_val);
    }
    Ok((x_all, y_all))
}

// ─────────────────────────────────────────────────────────────────────────────
// Friedman #2
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #2 regression benchmark dataset (4 features).
///
/// The target is:
/// ```text
/// y = (x₁² + (x₂·x₃ - 1/(x₂·x₄))²)^(1/2) + ε
/// ```
/// where:
/// - `x₁ ~ Uniform(0, 100)`
/// - `x₂ ~ Uniform(40π, 560π)`
/// - `x₃ ~ Uniform(0, 1)`
/// - `x₄ ~ Uniform(1, 11)`
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × 4`.
pub fn friedman2_bench(
    n_samples: usize,
    noise: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "friedman2_bench: n_samples must be > 0".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "friedman2_bench: noise must be >= 0".into(),
        ));
    }

    let u1 = Uniform::new(0.0f64, 100.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("friedman2_bench: u1 init: {e}"))
    })?;
    let u2 = Uniform::new(40.0 * PI, 560.0 * PI).map_err(|e| {
        DatasetsError::InvalidFormat(format!("friedman2_bench: u2 init: {e}"))
    })?;
    let u3 = Uniform::new(0.0f64, 1.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("friedman2_bench: u3 init: {e}"))
    })?;
    let u4 = Uniform::new(1.0f64, 11.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("friedman2_bench: u4 init: {e}"))
    })?;

    let mut x_all = Vec::with_capacity(n_samples);
    let mut y_all = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x1 = u1.sample(rng);
        let x2 = u2.sample(rng);
        let x3 = u3.sample(rng);
        let x4 = u4.sample(rng);
        let y_val = (x1.powi(2) + (x2 * x3 - 1.0 / (x2 * x4)).powi(2)).sqrt()
            + noise * sample_normal(rng);
        x_all.push(vec![x1, x2, x3, x4]);
        y_all.push(y_val);
    }
    Ok((x_all, y_all))
}

// ─────────────────────────────────────────────────────────────────────────────
// Moons
// ─────────────────────────────────────────────────────────────────────────────

/// Generate two interleaving half-circle (moons) datasets.
///
/// Class 0 is the top half-circle, class 1 is the bottom half-circle
/// (shifted right and down).  Gaussian noise is added to both.
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × 2` and `y ∈ {0, 1}`.
pub fn moons_bench(
    n_samples: usize,
    noise: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "moons_bench: n_samples must be > 0".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "moons_bench: noise must be >= 0".into(),
        ));
    }

    let n0 = n_samples / 2;
    let n1 = n_samples - n0;

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    // Top half-circle: angles in [0, π]
    let step0 = PI / (n0 as f64 - 1.0).max(1.0);
    for i in 0..n0 {
        let angle = step0 * i as f64;
        let nx = noise * sample_normal(rng);
        let ny = noise * sample_normal(rng);
        x.push(vec![angle.cos() + nx, angle.sin() + ny]);
        y.push(0usize);
    }
    // Bottom half-circle: angles in [π, 2π], shifted
    let step1 = PI / (n1 as f64 - 1.0).max(1.0);
    for i in 0..n1 {
        let angle = PI + step1 * i as f64;
        let nx = noise * sample_normal(rng);
        let ny = noise * sample_normal(rng);
        x.push(vec![angle.cos() + 1.0 + nx, angle.sin() + 0.5 + ny]);
        y.push(1usize);
    }
    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Circles
// ─────────────────────────────────────────────────────────────────────────────

/// Generate two concentric circles dataset.
///
/// Class 0 is the outer circle (radius 1), class 1 is the inner circle
/// (radius `factor`).  Gaussian noise is added.
///
/// # Parameters
/// - `factor` — ratio of inner to outer radius, must be in `(0, 1)`.
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × 2` and `y ∈ {0, 1}`.
pub fn circles_bench(
    n_samples: usize,
    noise: f64,
    factor: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "circles_bench: n_samples must be > 0".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "circles_bench: noise must be >= 0".into(),
        ));
    }
    if factor <= 0.0 || factor >= 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "circles_bench: factor must be in (0, 1)".into(),
        ));
    }

    let n0 = n_samples / 2;
    let n1 = n_samples - n0;

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    let step0 = 2.0 * PI / n0 as f64;
    for i in 0..n0 {
        let angle = step0 * i as f64;
        let nx = noise * sample_normal(rng);
        let ny = noise * sample_normal(rng);
        x.push(vec![angle.cos() + nx, angle.sin() + ny]);
        y.push(0usize);
    }
    let step1 = 2.0 * PI / n1 as f64;
    for i in 0..n1 {
        let angle = step1 * i as f64;
        let nx = noise * sample_normal(rng);
        let ny = noise * sample_normal(rng);
        x.push(vec![factor * angle.cos() + nx, factor * angle.sin() + ny]);
        y.push(1usize);
    }
    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Swiss roll
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Swiss roll manifold embedded in 3-D.
///
/// The roll parameter `t ∈ [1.5π, 4.5π]` drives the helix, and `height`
/// is sampled uniformly in `[0, 21]`.
///
/// # Returns
/// `(X, t)` where `X` is `n_samples × 3` and `t` is the roll parameter (color label).
pub fn swiss_roll_bench(
    n_samples: usize,
    noise: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "swiss_roll_bench: n_samples must be > 0".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "swiss_roll_bench: noise must be >= 0".into(),
        ));
    }

    let t_dist = Uniform::new(1.5 * PI, 4.5 * PI).map_err(|e| {
        DatasetsError::InvalidFormat(format!("swiss_roll_bench: t_dist init: {e}"))
    })?;
    let h_dist = Uniform::new(0.0f64, 21.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("swiss_roll_bench: h_dist init: {e}"))
    })?;

    let mut x = Vec::with_capacity(n_samples);
    let mut t_vals = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let t = t_dist.sample(rng);
        let height = h_dist.sample(rng);
        let xv = t * t.cos() + noise * sample_normal(rng);
        let yv = height + noise * sample_normal(rng);
        let zv = t * t.sin() + noise * sample_normal(rng);
        x.push(vec![xv, yv, zv]);
        t_vals.push(t);
    }
    Ok((x, t_vals))
}

// ─────────────────────────────────────────────────────────────────────────────
// S-curve
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an S-shaped manifold in 3-D.
///
/// The manifold parameter `t ∈ [-3π/2, 3π/2]` drives the S-shape.
///
/// # Returns
/// `(X, t)` where `X` is `n_samples × 3` and `t` is the manifold parameter.
pub fn s_curve_bench(
    n_samples: usize,
    noise: f64,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "s_curve_bench: n_samples must be > 0".into(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "s_curve_bench: noise must be >= 0".into(),
        ));
    }

    let t_dist = Uniform::new(-1.5 * PI, 1.5 * PI).map_err(|e| {
        DatasetsError::InvalidFormat(format!("s_curve_bench: t_dist init: {e}"))
    })?;
    let h_dist = Uniform::new(0.0f64, 2.0).map_err(|e| {
        DatasetsError::InvalidFormat(format!("s_curve_bench: h_dist init: {e}"))
    })?;

    let mut x = Vec::with_capacity(n_samples);
    let mut t_vals = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let t = t_dist.sample(rng);
        let height = h_dist.sample(rng);

        // S-curve shape
        let xv = t.sin() + noise * sample_normal(rng);
        let yv = height + noise * sample_normal(rng);
        let zv = (if t > 0.0 { -1.0 } else { 1.0 }) * (1.0 - t.abs().cos())
            + noise * sample_normal(rng);
        x.push(vec![xv, yv, zv]);
        t_vals.push(t);
    }
    Ok((x, t_vals))
}

// ─────────────────────────────────────────────────────────────────────────────
// Imbalanced classification
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an imbalanced binary classification dataset.
///
/// Class 0 (majority) has `round(n_samples × (1 - 1/(1+imbalance_ratio)))` samples;
/// class 1 (minority) has the remainder.  Features are sampled from Gaussians with
/// different means:
/// - Class 0: `N(0, I)`
/// - Class 1: `N(1, I)` (shifted by 1 in every feature)
///
/// # Parameters
/// - `imbalance_ratio` — ratio `|majority| / |minority|` (must be ≥ 1)
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × n_features` and `y ∈ {0, 1}`.
pub fn imbalanced_classification(
    n_samples: usize,
    imbalance_ratio: f64,
    n_features: usize,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: n_samples must be > 0".into(),
        ));
    }
    if imbalance_ratio < 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: imbalance_ratio must be >= 1".into(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "imbalanced_classification: n_features must be > 0".into(),
        ));
    }

    // minority count = n_samples / (1 + imbalance_ratio)
    let n_minority = (n_samples as f64 / (1.0 + imbalance_ratio)).round() as usize;
    let n_minority = n_minority.max(1);
    let n_majority = n_samples - n_minority;

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    // Majority class (class 0): centred at origin
    for _ in 0..n_majority {
        let row: Vec<f64> = randn_vec(n_features, rng);
        x.push(row);
        y.push(0usize);
    }
    // Minority class (class 1): centred at 1 in all features
    for _ in 0..n_minority {
        let row: Vec<f64> = randn_vec(n_features, rng)
            .into_iter()
            .map(|xi| xi + 1.0)
            .collect();
        x.push(row);
        y.push(1usize);
    }
    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Concept drift
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a binary classification dataset with concept drift.
///
/// Before `drift_at`, the boundary is at `x₀ > 0` → class 1.
/// After `drift_at`, the boundary rotates to `x₁ > 0` → class 1.
///
/// Features are sampled from `N(0, I)` throughout.
///
/// # Parameters
/// - `drift_at` — sample index at which the decision boundary changes
///
/// # Returns
/// `(X, y)` where `X` is `n_samples × n_features` and `y ∈ {0, 1}`.
pub fn concept_drift(
    n_samples: usize,
    n_features: usize,
    drift_at: usize,
    rng: &mut impl Rng,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "concept_drift: n_samples must be > 0".into(),
        ));
    }
    if n_features < 2 {
        return Err(DatasetsError::InvalidFormat(
            "concept_drift: n_features must be >= 2".into(),
        ));
    }
    if drift_at >= n_samples {
        return Err(DatasetsError::InvalidFormat(format!(
            "concept_drift: drift_at ({drift_at}) must be < n_samples ({n_samples})"
        )));
    }

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let row = randn_vec(n_features, rng);
        // Decision boundary depends on phase
        let label = if i < drift_at {
            if row[0] > 0.0 { 1usize } else { 0usize }
        } else {
            if row[1] > 0.0 { 1usize } else { 0usize }
        };
        x.push(row);
        y.push(label);
    }
    Ok((x, y))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::StdRng;
    use scirs2_core::random::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_friedman1_shape() {
        let mut rng = make_rng();
        let (x, y) = friedman1_bench(100, 5, 0.1, &mut rng).expect("valid params");
        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);
        assert_eq!(x[0].len(), 5);
    }

    #[test]
    fn test_friedman1_too_few_features() {
        let mut rng = make_rng();
        assert!(friedman1_bench(10, 3, 0.1, &mut rng).is_err());
    }

    #[test]
    fn test_friedman2_shape() {
        let mut rng = make_rng();
        let (x, y) = friedman2_bench(50, 0.0, &mut rng).expect("valid params");
        assert_eq!(x.len(), 50);
        assert_eq!(y.len(), 50);
        assert_eq!(x[0].len(), 4);
        for &yv in &y {
            assert!(yv.is_finite() && yv >= 0.0, "friedman2 target: {yv}");
        }
    }

    #[test]
    fn test_moons_shape_and_classes() {
        let mut rng = make_rng();
        let (x, y) = moons_bench(200, 0.05, &mut rng).expect("valid params");
        assert_eq!(x.len(), 200);
        assert_eq!(y.len(), 200);
        assert!(y.iter().all(|&c| c < 2));
        let n0 = y.iter().filter(|&&c| c == 0).count();
        let n1 = y.iter().filter(|&&c| c == 1).count();
        assert_eq!(n0 + n1, 200);
    }

    #[test]
    fn test_circles_shape() {
        let mut rng = make_rng();
        let (x, y) = circles_bench(150, 0.05, 0.5, &mut rng).expect("valid params");
        assert_eq!(x.len(), 150);
        assert_eq!(y.len(), 150);
        assert_eq!(x[0].len(), 2);
    }

    #[test]
    fn test_circles_invalid_factor() {
        let mut rng = make_rng();
        assert!(circles_bench(10, 0.0, 0.0, &mut rng).is_err());
        assert!(circles_bench(10, 0.0, 1.0, &mut rng).is_err());
    }

    #[test]
    fn test_swiss_roll_shape() {
        let mut rng = make_rng();
        let (x, t) = swiss_roll_bench(300, 0.1, &mut rng).expect("valid params");
        assert_eq!(x.len(), 300);
        assert_eq!(t.len(), 300);
        assert_eq!(x[0].len(), 3);
        for &tv in &t {
            assert!((1.5 * PI..=4.5 * PI).contains(&tv), "t out of range: {tv}");
        }
    }

    #[test]
    fn test_s_curve_shape() {
        let mut rng = make_rng();
        let (x, t) = s_curve_bench(200, 0.1, &mut rng).expect("valid params");
        assert_eq!(x.len(), 200);
        assert_eq!(t.len(), 200);
        assert_eq!(x[0].len(), 3);
    }

    #[test]
    fn test_imbalanced_classification_ratio() {
        let mut rng = make_rng();
        let (x, y) = imbalanced_classification(110, 10.0, 4, &mut rng).expect("valid params");
        assert_eq!(x.len(), 110);
        let n1 = y.iter().filter(|&&c| c == 1).count();
        let n0 = y.iter().filter(|&&c| c == 0).count();
        // Minority should be roughly 10% = 10 samples
        assert!(n1 < n0, "minority ({n1}) should be < majority ({n0})");
    }

    #[test]
    fn test_imbalanced_classification_invalid() {
        let mut rng = make_rng();
        assert!(imbalanced_classification(10, 0.5, 3, &mut rng).is_err()); // ratio < 1
        assert!(imbalanced_classification(10, 5.0, 0, &mut rng).is_err()); // 0 features
    }

    #[test]
    fn test_concept_drift_labels_change() {
        let mut rng = make_rng();
        let n = 200;
        let drift = 100;
        let (x, y) = concept_drift(n, 3, drift, &mut rng).expect("valid params");
        assert_eq!(x.len(), n);
        assert_eq!(y.len(), n);
        // Before drift: label == (x[0][0] > 0)
        for i in 0..drift {
            let expected = if x[i][0] > 0.0 { 1 } else { 0 };
            assert_eq!(y[i], expected, "pre-drift label mismatch at i={i}");
        }
        // After drift: label == (x[i][1] > 0)
        for i in drift..n {
            let expected = if x[i][1] > 0.0 { 1 } else { 0 };
            assert_eq!(y[i], expected, "post-drift label mismatch at i={i}");
        }
    }

    #[test]
    fn test_concept_drift_invalid() {
        let mut rng = make_rng();
        assert!(concept_drift(100, 1, 50, &mut rng).is_err()); // n_features < 2
        assert!(concept_drift(100, 3, 100, &mut rng).is_err()); // drift_at >= n_samples
    }
}
