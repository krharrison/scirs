//! Automatic interpolation method selection based on data characteristics.
//!
//! `analyze_data` builds a `DataProfile` by examining the data's dimensionality,
//! size, smoothness, noise, and periodicity.  `recommend_method` then applies a
//! set of decision rules to select the most appropriate interpolation strategy.
//!
//! # Decision Rules (in priority order)
//!
//! | Condition | Method |
//! |-----------|--------|
//! | `n_dims == 1 && !noisy` | `CubicSpline` |
//! | `n_dims ≤ 4 && n_points < 500` | `RadialBasis` |
//! | `n_dims ≤ 6 && n_points < 10_000` | `TensorProduct` |
//! | `n_dims > 6 && n_points > 1_000` | `SparseGrid` |
//! | `n_dims > 10` | `TensorTrain` |
//! | default | `RadialBasis` |

use crate::error::InterpolateError;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics derived from data analysis.
#[derive(Debug, Clone)]
pub struct DataProfile {
    /// Number of data points.
    pub n_points: usize,
    /// Number of input dimensions.
    pub n_dims: usize,
    /// Estimated smoothness (ratio of second-difference norm to value norm;
    /// lower is smoother).
    pub smoothness_estimate: f64,
    /// Whether the data appears to be noisy.
    pub has_noise: bool,
    /// Whether the data appears to be periodic.
    pub is_periodic: bool,
}

/// Interpolation method recommendation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMethod {
    /// Piecewise linear interpolation (fast, first-order accurate).
    LinearSpline,
    /// Piecewise cubic spline (smooth, good for 1-D well-sampled data).
    CubicSpline,
    /// Radial basis function interpolation (flexible, handles scattered data).
    RadialBasis,
    /// Tensor-product interpolation on structured grids (efficient up to ~6-D).
    TensorProduct,
    /// Sparse-grid interpolation (Smolyak; handles moderate-dimensional spaces).
    SparseGrid,
    /// Tensor-train (TT/MPS) decomposition (handles very high dimensions).
    TensorTrain,
}

impl std::fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            InterpolationMethod::LinearSpline => "LinearSpline",
            InterpolationMethod::CubicSpline => "CubicSpline",
            InterpolationMethod::RadialBasis => "RadialBasis",
            InterpolationMethod::TensorProduct => "TensorProduct",
            InterpolationMethod::SparseGrid => "SparseGrid",
            InterpolationMethod::TensorTrain => "TensorTrain",
        };
        write!(f, "{}", name)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Analyse data to build a `DataProfile`.
///
/// # Arguments
/// * `x` – Input points, each of length `n_dims`.  Must be non-empty.
/// * `y` – Function values.  Must have the same length as `x`.
///
/// # Smoothness estimation
///
/// For 1-D data, the smoothness estimate is the RMS of second-order finite
/// differences normalised by the RMS of the values.  A small ratio (< 0.1)
/// indicates a smooth function; a large ratio indicates roughness or noise.
///
/// For multi-dimensional data we use only the first-coordinate ordering.
///
/// # Noise detection
///
/// The data is considered noisy if `smoothness_estimate > 0.3`.
///
/// # Periodicity detection
///
/// If `|y.first() - y.last()| / max(|y|)  < 0.05` the data is considered
/// (approximately) periodic.
pub fn analyze_data(x: &[Vec<f64>], y: &[f64]) -> DataProfile {
    let n_points = x.len();
    let n_dims = if n_points > 0 { x[0].len() } else { 0 };

    if n_points < 3 || n_dims == 0 {
        return DataProfile {
            n_points,
            n_dims,
            smoothness_estimate: 0.0,
            has_noise: false,
            is_periodic: false,
        };
    }

    // ── Smoothness: second-order finite differences on y (sorted by x[0]) ─

    // Sort by first coordinate.
    let mut order: Vec<usize> = (0..n_points).collect();
    order.sort_by(|&a, &b| {
        x[a][0]
            .partial_cmp(&x[b][0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let y_sorted: Vec<f64> = order.iter().map(|&i| y[i]).collect();

    let rms_y = (y_sorted.iter().map(|&v| v * v).sum::<f64>() / n_points as f64)
        .sqrt()
        .max(1e-12);

    let second_diff_rms = if n_points >= 3 {
        let n = y_sorted.len();
        let ss: f64 = (1..(n - 1))
            .map(|i| {
                let d2 = y_sorted[i + 1] - 2.0 * y_sorted[i] + y_sorted[i - 1];
                d2 * d2
            })
            .sum::<f64>();
        (ss / (n - 2) as f64).sqrt()
    } else {
        0.0
    };

    let smoothness_estimate = second_diff_rms / rms_y;

    // ── Noise detection ────────────────────────────────────────────────────
    let has_noise = smoothness_estimate > 0.3;

    // ── Periodicity detection ──────────────────────────────────────────────
    let y_max_abs = y_sorted
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let endpoint_diff = (y_sorted[0] - y_sorted[n_points - 1]).abs();
    let is_periodic = endpoint_diff / y_max_abs < 0.05;

    DataProfile {
        n_points,
        n_dims,
        smoothness_estimate,
        has_noise,
        is_periodic,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Method recommendation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply decision rules to select an interpolation method.
///
/// See the module-level documentation for the full rule table.
pub fn recommend_method(profile: &DataProfile) -> InterpolationMethod {
    let d = profile.n_dims;
    let n = profile.n_points;

    if d == 1 && !profile.has_noise {
        return InterpolationMethod::CubicSpline;
    }
    if d > 10 {
        return InterpolationMethod::TensorTrain;
    }
    if d <= 4 && n < 500 {
        return InterpolationMethod::RadialBasis;
    }
    if d <= 6 && n < 10_000 {
        return InterpolationMethod::TensorProduct;
    }
    if d > 6 && n > 1_000 {
        return InterpolationMethod::SparseGrid;
    }
    // Default
    InterpolationMethod::RadialBasis
}

/// Apply decision rules and return the chosen method together with a
/// human-readable rationale string.
pub fn recommend_with_rationale(profile: &DataProfile) -> (InterpolationMethod, String) {
    let d = profile.n_dims;
    let n = profile.n_points;

    if d == 1 && !profile.has_noise {
        return (
            InterpolationMethod::CubicSpline,
            format!(
                "1-D data ({n} points) without noise: CubicSpline gives smooth, \
                 C² interpolation at O(n) cost."
            ),
        );
    }
    if d > 10 {
        return (
            InterpolationMethod::TensorTrain,
            format!(
                "{d}-D data ({n} points): dimensionality exceeds 10; \
                 TensorTrain (TT-SVD/TT-cross) avoids the curse of dimensionality."
            ),
        );
    }
    if d <= 4 && n < 500 {
        return (
            InterpolationMethod::RadialBasis,
            format!(
                "{d}-D scattered data ({n} points): RBF provides flexible \
                 interpolation without a grid structure."
            ),
        );
    }
    if d <= 6 && n < 10_000 {
        return (
            InterpolationMethod::TensorProduct,
            format!(
                "{d}-D data ({n} points): a tensor-product grid is feasible \
                 and gives fast O(n) evaluation per dimension."
            ),
        );
    }
    if d > 6 && n > 1_000 {
        return (
            InterpolationMethod::SparseGrid,
            format!(
                "{d}-D data ({n} points): Smolyak sparse grid reduces the \
                 exponential cost of tensor-product methods in moderate dimensions."
            ),
        );
    }

    (
        InterpolationMethod::RadialBasis,
        format!(
            "Default choice for {d}-D data ({n} points): RBF interpolation \
             works well for general scattered data."
        ),
    )
}

/// Validate input slices for consistency (helper used internally).
#[allow(dead_code)]
pub(crate) fn validate_input(x: &[Vec<f64>], y: &[f64]) -> Result<(), InterpolateError> {
    if x.len() != y.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "x has {} points but y has {} values",
            x.len(),
            y.len()
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64]).collect();
        let y: Vec<f64> = x.iter().map(|p| p[0] * p[0]).collect();
        (x, y)
    }

    fn make_nd_data(n: usize, d: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64; d]).collect();
        let y: Vec<f64> = x.iter().map(|p| p.iter().sum::<f64>()).collect();
        (x, y)
    }

    #[test]
    fn test_1d_smooth_data_recommends_cubic_spline() {
        let (x, y) = make_1d_data(50);
        let profile = analyze_data(&x, &y);
        assert_eq!(profile.n_dims, 1);
        let method = recommend_method(&profile);
        assert_eq!(method, InterpolationMethod::CubicSpline);
    }

    #[test]
    fn test_high_dim_recommends_tensor_train() {
        let (x, y) = make_nd_data(2000, 15);
        let profile = analyze_data(&x, &y);
        let method = recommend_method(&profile);
        assert_eq!(method, InterpolationMethod::TensorTrain);
    }

    #[test]
    fn test_moderate_dim_recommends_sparse_grid() {
        // 8-D, 2000 points
        let (x, y) = make_nd_data(2000, 8);
        let profile = analyze_data(&x, &y);
        let method = recommend_method(&profile);
        assert_eq!(method, InterpolationMethod::SparseGrid);
    }

    #[test]
    fn test_small_4d_recommends_rbf() {
        let (x, y) = make_nd_data(100, 4);
        let profile = analyze_data(&x, &y);
        let method = recommend_method(&profile);
        assert_eq!(method, InterpolationMethod::RadialBasis);
    }

    #[test]
    fn test_recommend_with_rationale_returns_string() {
        let (x, y) = make_1d_data(20);
        let profile = analyze_data(&x, &y);
        let (method, reason) = recommend_with_rationale(&profile);
        assert_eq!(method, InterpolationMethod::CubicSpline);
        assert!(!reason.is_empty(), "rationale string should not be empty");
    }

    #[test]
    fn test_analyze_data_smoothness_for_noisy_data() {
        // Noisy data: add random-looking second differences.
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.1]).collect();
        // Alternating sign induces large second differences → noisy.
        let y: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let profile = analyze_data(&x, &y);
        assert!(
            profile.has_noise,
            "alternating data should be flagged as noisy"
        );
    }

    #[test]
    fn test_periodicity_detected() {
        // Sine on a closed interval [0, 2π] with equal endpoints (both ≈ 0).
        use std::f64::consts::PI;
        let n = 65_usize; // 65 points so first and last are both x=0 and x=2π
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64 * 2.0 * PI / (n - 1) as f64])
            .collect();
        let y: Vec<f64> = x.iter().map(|p| p[0].sin()).collect();
        // y[0] = sin(0) = 0.0, y[n-1] = sin(2π) ≈ 0.0
        let profile = analyze_data(&x, &y);
        assert!(
            profile.is_periodic,
            "sin data on [0,2π] should be detected as periodic; y[0]={:.4}, y[last]={:.4}",
            y[0],
            y[n - 1]
        );
    }

    #[test]
    fn test_empty_data_no_panic() {
        let profile = analyze_data(&[], &[]);
        assert_eq!(profile.n_points, 0);
        assert_eq!(profile.n_dims, 0);
    }
}
