//! Synthetic benchmark dataset generators.
//!
//! These functions produce standard synthetic datasets widely used for
//! evaluating regression, classification, and manifold-learning algorithms.
//! All randomness is driven by a seeded LCG — no external `rand` crate required.
//!
//! | Function | Type | Description |
//! |---|---|---|
//! | [`make_friedman1`] | regression | Friedman #1 (10·sin(π·x₁·x₂) + …) |
//! | [`make_friedman2`] | regression | Friedman #2 (sqrt formula) |
//! | [`make_blobs`] | clustering | Isotropic Gaussian blobs |
//! | [`make_moons`] | binary classification | Two interleaved half-moons |
//! | [`make_swiss_roll`] | manifold | 3-D Swiss roll |
//! | [`make_checkerboard`] | binary classification | 2-D checkerboard pattern |

use crate::error::{DatasetsError, Result};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// LCG helpers (same parameters as sharding module)
// ─────────────────────────────────────────────────────────────────────────────

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform sample in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform sample in `[lo, hi)`.
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }

    /// Standard normal via Box-Muller (consumes two LCG draws).
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::EPSILON);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Normal with given mean and std deviation.
    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.standard_normal()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Friedman #1
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #1 regression benchmark.
///
/// ```text
/// y = 10·sin(π·x₁·x₂) + 20·(x₃ - 0.5)² + 10·x₄ + 5·x₅ + ε
/// ```
///
/// where `xᵢ ~ Uniform[0, 1]` for all `n_features` columns and
/// `ε ~ Normal(0, noise²)`.  Only the first five features affect the target.
///
/// # Errors
///
/// Returns [`DatasetsError::InvalidFormat`] if `n_features < 5`.
pub fn make_friedman1(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_features < 5 {
        return Err(DatasetsError::InvalidFormat(
            "make_friedman1: n_features must be >= 5".into(),
        ));
    }
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_friedman1: n_samples must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);
    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut y_data: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let row: Vec<f64> = (0..n_features).map(|_| rng.uniform()).collect();
        let x1 = row[0];
        let x2 = row[1];
        let x3 = row[2];
        let x4 = row[3];
        let x5 = row[4];
        let y = 10.0 * (PI * x1 * x2).sin()
            + 20.0 * (x3 - 0.5).powi(2)
            + 10.0 * x4
            + 5.0 * x5
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };
        x_data.push(row);
        y_data.push(y);
    }

    Ok((x_data, y_data))
}

// ─────────────────────────────────────────────────────────────────────────────
// Friedman #2
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the Friedman #2 regression benchmark.
///
/// ```text
/// y = sqrt(x₁² + (x₂·x₃ - 1/(x₂·x₄))²) + ε
/// ```
///
/// Feature ranges:
/// - `x₁ ~ Uniform[0, 100]`
/// - `x₂ ~ Uniform[40π, 560π]`
/// - `x₃ ~ Uniform[0, 1]`
/// - `x₄ ~ Uniform[1, 11]`
pub fn make_friedman2(
    n_samples: usize,
    noise: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_friedman2: n_samples must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);
    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut y_data: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x1 = rng.uniform_range(0.0, 100.0);
        let x2 = rng.uniform_range(40.0 * PI, 560.0 * PI);
        let x3 = rng.uniform_range(0.0, 1.0);
        let x4 = rng.uniform_range(1.0, 11.0);

        let inner = x2 * x3 - 1.0 / (x2 * x4);
        let y = (x1 * x1 + inner * inner).sqrt()
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };

        x_data.push(vec![x1, x2, x3, x4]);
        y_data.push(y);
    }

    Ok((x_data, y_data))
}

// ─────────────────────────────────────────────────────────────────────────────
// Blobs
// ─────────────────────────────────────────────────────────────────────────────

/// Generate isotropic Gaussian blobs for clustering benchmarks.
///
/// `n_centers` cluster centers are placed at random in `[0, 10]^n_features`.
/// Each sample is drawn from `Normal(center_k, cluster_std² · I)`.
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    n_centers: usize,
    cluster_std: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs: n_samples must be > 0".into(),
        ));
    }
    if n_centers == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs: n_centers must be > 0".into(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_blobs: n_features must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);

    // Generate cluster centers.
    let centers: Vec<Vec<f64>> = (0..n_centers)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.uniform_range(0.0, 10.0))
                .collect()
        })
        .collect();

    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let center_id = i % n_centers;
        let center = &centers[center_id];
        let row: Vec<f64> = center.iter().map(|&c| rng.normal(c, cluster_std)).collect();
        x_data.push(row);
        labels.push(center_id);
    }

    Ok((x_data, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Moons
// ─────────────────────────────────────────────────────────────────────────────

/// Generate two interleaved half-moon shapes.
///
/// - Class 0: outer moon with radius ≈ 1, centre at `(0, 0)`.
/// - Class 1: inner moon with radius ≈ 0.5, centre at `(0.5, -0.25)`.
///
/// Gaussian noise with standard deviation `noise` is added to all coordinates.
pub fn make_moons(n_samples: usize, noise: f64, seed: u64) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_moons: n_samples must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);
    let half = n_samples / 2;
    let remaining = n_samples - half;

    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(n_samples);

    // Class 0: upper half-moon (theta in [0, π]).
    for i in 0..half {
        let theta = PI * i as f64 / (half.max(1) - 1).max(1) as f64;
        let x = theta.cos()
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };
        let y = theta.sin()
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };
        x_data.push(vec![x, y]);
        labels.push(0);
    }

    // Class 1: lower half-moon (theta in [0, π]) shifted right and down.
    for i in 0..remaining {
        let theta = PI * i as f64 / (remaining.max(1) - 1).max(1) as f64;
        let x = 1.0 - theta.cos()
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };
        let y = -theta.sin()
            + 0.5
            + if noise > 0.0 {
                rng.normal(0.0, noise)
            } else {
                0.0
            };
        x_data.push(vec![x, y]);
        labels.push(1);
    }

    Ok((x_data, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Swiss roll
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a 3-D Swiss roll manifold.
///
/// `t ~ Uniform[1.5π, 4.5π]`,
/// `y_rand ~ Uniform[0, 21]`
///
/// ```text
/// x = t · cos(t)
/// y = y_rand
/// z = t · sin(t)
/// ```
///
/// Gaussian noise with std `noise` is added to all coordinates.
/// The returned labels are the continuous `t` values (useful for colouring).
pub fn make_swiss_roll(
    n_samples: usize,
    noise: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_swiss_roll: n_samples must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);
    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut t_values: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let t = rng.uniform_range(1.5 * PI, 4.5 * PI);
        let y_rand = rng.uniform_range(0.0, 21.0);

        let nx = if noise > 0.0 {
            rng.normal(0.0, noise)
        } else {
            0.0
        };
        let ny = if noise > 0.0 {
            rng.normal(0.0, noise)
        } else {
            0.0
        };
        let nz = if noise > 0.0 {
            rng.normal(0.0, noise)
        } else {
            0.0
        };

        x_data.push(vec![t * t.cos() + nx, y_rand + ny, t * t.sin() + nz]);
        t_values.push(t);
    }

    Ok((x_data, t_values))
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkerboard
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a 2-D checkerboard classification dataset.
///
/// Points are drawn uniformly from `[0, 1)²`.  The label of a point `(x, y)` is
///
/// ```text
/// label = (floor(x · n_squares) + floor(y · n_squares)) % 2
/// ```
///
/// so that the positive class (`label == 0`) occupies every other square.
pub fn make_checkerboard(
    n_samples: usize,
    n_squares: usize,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<usize>)> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_checkerboard: n_samples must be > 0".into(),
        ));
    }
    if n_squares == 0 {
        return Err(DatasetsError::InvalidFormat(
            "make_checkerboard: n_squares must be > 0".into(),
        ));
    }

    let mut rng = Lcg64::new(seed);
    let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
    let mut labels: Vec<usize> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x = rng.uniform();
        let y = rng.uniform();
        let label =
            ((x * n_squares as f64).floor() as usize + (y * n_squares as f64).floor() as usize) % 2;
        x_data.push(vec![x, y]);
        labels.push(label);
    }

    Ok((x_data, labels))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_friedman1_shape() {
        let (x, y) = make_friedman1(50, 7, 0.0, 0).expect("ok");
        assert_eq!(x.len(), 50);
        assert_eq!(y.len(), 50);
        assert_eq!(x[0].len(), 7);
    }

    #[test]
    fn test_friedman1_value_range() {
        let (x, _y) = make_friedman1(100, 5, 0.0, 42).expect("ok");
        for row in &x {
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "feature value {v} out of [0,1]");
            }
        }
    }

    #[test]
    fn test_friedman1_needs_5_features() {
        assert!(make_friedman1(10, 4, 0.0, 0).is_err());
    }

    #[test]
    fn test_friedman2_shape() {
        let (x, y) = make_friedman2(30, 0.0, 7).expect("ok");
        assert_eq!(x.len(), 30);
        assert_eq!(y.len(), 30);
        assert_eq!(x[0].len(), 4);
    }

    #[test]
    fn test_friedman2_positive_targets() {
        let (_, y) = make_friedman2(100, 0.0, 0).expect("ok");
        // The sqrt formula always yields a non-negative value.
        for &v in &y {
            assert!(v >= 0.0, "negative target {v}");
        }
    }

    #[test]
    fn test_make_blobs_shape() {
        let (x, labels) = make_blobs(120, 3, 4, 1.0, 99).expect("ok");
        assert_eq!(x.len(), 120);
        assert_eq!(labels.len(), 120);
        assert_eq!(x[0].len(), 3);
    }

    #[test]
    fn test_make_blobs_all_centers_present() {
        let n_centers = 5;
        let (_, labels) = make_blobs(100, 2, n_centers, 0.1, 0).expect("ok");
        // With 100 samples round-robin across 5 centers, each center has 20 samples.
        let mut present = vec![false; n_centers];
        for &l in &labels {
            present[l] = true;
        }
        assert!(
            present.iter().all(|&p| p),
            "not all centers were used: {present:?}"
        );
    }

    #[test]
    fn test_make_moons_two_classes() {
        let (x, labels) = make_moons(200, 0.0, 0).expect("ok");
        assert_eq!(x.len(), 200);
        assert_eq!(labels.len(), 200);
        let has_zero = labels.contains(&0);
        let has_one = labels.contains(&1);
        assert!(has_zero && has_one);
        // All labels are 0 or 1.
        assert!(labels.iter().all(|&l| l < 2));
    }

    #[test]
    fn test_make_moons_2d() {
        let (x, _) = make_moons(50, 0.0, 1).expect("ok");
        for row in &x {
            assert_eq!(row.len(), 2);
        }
    }

    #[test]
    fn test_make_swiss_roll_3d() {
        let (x, t) = make_swiss_roll(80, 0.0, 5).expect("ok");
        assert_eq!(x.len(), 80);
        assert_eq!(t.len(), 80);
        for row in &x {
            assert_eq!(row.len(), 3, "expected 3D coordinates");
        }
    }

    #[test]
    fn test_make_swiss_roll_t_range() {
        let (_, t) = make_swiss_roll(200, 0.0, 0).expect("ok");
        for &v in &t {
            assert!((1.5 * PI..=4.5 * PI).contains(&v), "t={v} out of range");
        }
    }

    #[test]
    fn test_make_checkerboard_shape() {
        let (x, labels) = make_checkerboard(100, 4, 0).expect("ok");
        assert_eq!(x.len(), 100);
        assert_eq!(labels.len(), 100);
        for row in &x {
            assert_eq!(row.len(), 2);
            assert!((0.0..1.0).contains(&row[0]));
            assert!((0.0..1.0).contains(&row[1]));
        }
    }

    #[test]
    fn test_make_checkerboard_both_classes() {
        let (_, labels) = make_checkerboard(400, 4, 42).expect("ok");
        let has_zero = labels.contains(&0);
        let has_one = labels.contains(&1);
        assert!(has_zero && has_one);
    }
}
