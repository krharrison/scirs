//! Dynamic Time Warping (DTW)
//!
//! Computes the Dynamic Time Warping distance and alignment between time series,
//! with support for various constraints and distance functions.
//!
//! # Features
//!
//! - Classic DTW with configurable point-wise distance
//! - Sakoe-Chiba band constraint
//! - Itakura parallelogram constraint
//! - Full alignment path recovery
//! - Open-end DTW for subsequence matching
//! - Multidimensional DTW
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::distributional::dtw::{dtw_distance, DtwConfig};
//!
//! let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let config = DtwConfig::default();
//! let result = dtw_distance(&a, &b, &config).expect("should succeed");
//! assert!(result.distance < 1e-10);
//! ```

use crate::error::{MetricsError, Result};

/// Configuration for DTW computation.
#[derive(Debug, Clone)]
pub struct DtwConfig {
    /// Point-wise distance function to use.
    pub distance_fn: DistanceFunction,
    /// Constraint window to apply.
    pub constraint: DtwConstraint,
}

impl Default for DtwConfig {
    fn default() -> Self {
        Self {
            distance_fn: DistanceFunction::Euclidean,
            constraint: DtwConstraint::None,
        }
    }
}

/// Point-wise distance function for DTW.
#[derive(Debug, Clone, Copy)]
pub enum DistanceFunction {
    /// Euclidean (L2) distance: |a - b|
    Euclidean,
    /// Squared Euclidean: (a - b)^2
    SquaredEuclidean,
    /// Manhattan (L1) distance: |a - b|
    Manhattan,
}

impl DistanceFunction {
    /// Compute the distance between two scalars.
    fn compute(self, a: f64, b: f64) -> f64 {
        match self {
            DistanceFunction::Euclidean => (a - b).abs(),
            DistanceFunction::SquaredEuclidean => (a - b) * (a - b),
            DistanceFunction::Manhattan => (a - b).abs(),
        }
    }

    /// Compute the distance between two multidimensional points.
    fn compute_nd(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            DistanceFunction::Euclidean => {
                let mut sum = 0.0;
                for (ai, bi) in a.iter().zip(b.iter()) {
                    sum += (ai - bi) * (ai - bi);
                }
                sum.sqrt()
            }
            DistanceFunction::SquaredEuclidean => {
                let mut sum = 0.0;
                for (ai, bi) in a.iter().zip(b.iter()) {
                    sum += (ai - bi) * (ai - bi);
                }
                sum
            }
            DistanceFunction::Manhattan => {
                let mut sum = 0.0;
                for (ai, bi) in a.iter().zip(b.iter()) {
                    sum += (ai - bi).abs();
                }
                sum
            }
        }
    }
}

/// Constraint window for DTW.
#[derive(Debug, Clone, Copy)]
pub enum DtwConstraint {
    /// No constraint (full DTW matrix).
    None,
    /// Sakoe-Chiba band: |i/n - j/m| <= w (w is a fraction in [0, 1]).
    SakoeChibaBand(f64),
    /// Itakura parallelogram constraint with slope factor s.
    /// Constrains the path to lie within a parallelogram defined by
    /// slopes 1/s and s around the diagonal.
    ItakuraParallelogram(f64),
}

/// Result of DTW computation.
#[derive(Debug, Clone)]
pub struct DtwResult {
    /// The DTW distance.
    pub distance: f64,
    /// The alignment path as (i, j) index pairs.
    pub path: Vec<(usize, usize)>,
}

/// Computes the DTW distance between two 1D time series.
///
/// # Arguments
///
/// * `a` - First time series
/// * `b` - Second time series
/// * `config` - DTW configuration
///
/// # Returns
///
/// A `DtwResult` with the distance and alignment path.
pub fn dtw_distance(a: &[f64], b: &[f64], config: &DtwConfig) -> Result<DtwResult> {
    if a.is_empty() || b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }

    let n = a.len();
    let m = b.len();

    // Initialize cost matrix with infinity
    let mut dp = vec![f64::INFINITY; (n + 1) * (m + 1)];
    dp[0] = 0.0; // dp[0][0] = 0

    let idx = |i: usize, j: usize| -> usize { i * (m + 1) + j };

    for i in 1..=n {
        for j in 1..=m {
            if !is_within_constraint(i - 1, j - 1, n, m, &config.constraint) {
                continue;
            }

            let cost = config.distance_fn.compute(a[i - 1], b[j - 1]);
            let prev = dp[idx(i - 1, j)]
                .min(dp[idx(i, j - 1)])
                .min(dp[idx(i - 1, j - 1)]);
            dp[idx(i, j)] = cost + prev;
        }
    }

    let distance = dp[idx(n, m)];
    if distance.is_infinite() {
        return Err(MetricsError::CalculationError(
            "DTW distance is infinite - constraint window may be too narrow".to_string(),
        ));
    }

    // Traceback to find alignment path
    let path = traceback(&dp, n, m);

    Ok(DtwResult { distance, path })
}

/// Computes the DTW distance between two multidimensional time series.
///
/// # Arguments
///
/// * `a` - First time series, flattened row-major, shape [n, dim]
/// * `b` - Second time series, flattened row-major, shape [m, dim]
/// * `dim` - Dimensionality of each time step
/// * `config` - DTW configuration
///
/// # Returns
///
/// A `DtwResult` with the distance and alignment path.
pub fn dtw_distance_nd(a: &[f64], b: &[f64], dim: usize, config: &DtwConfig) -> Result<DtwResult> {
    if dim == 0 {
        return Err(MetricsError::InvalidInput(
            "dimension must be > 0".to_string(),
        ));
    }
    if a.is_empty() || b.is_empty() {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }
    if a.len() % dim != 0 || b.len() % dim != 0 {
        return Err(MetricsError::InvalidInput(format!(
            "time series length must be divisible by dim={dim}"
        )));
    }

    let n = a.len() / dim;
    let m = b.len() / dim;

    let mut dp = vec![f64::INFINITY; (n + 1) * (m + 1)];
    dp[0] = 0.0;

    let idx = |i: usize, j: usize| -> usize { i * (m + 1) + j };

    for i in 1..=n {
        for j in 1..=m {
            if !is_within_constraint(i - 1, j - 1, n, m, &config.constraint) {
                continue;
            }

            let a_slice = &a[(i - 1) * dim..i * dim];
            let b_slice = &b[(j - 1) * dim..j * dim];
            let cost = config.distance_fn.compute_nd(a_slice, b_slice);

            let prev = dp[idx(i - 1, j)]
                .min(dp[idx(i, j - 1)])
                .min(dp[idx(i - 1, j - 1)]);
            dp[idx(i, j)] = cost + prev;
        }
    }

    let distance = dp[idx(n, m)];
    if distance.is_infinite() {
        return Err(MetricsError::CalculationError(
            "DTW distance is infinite - constraint window may be too narrow".to_string(),
        ));
    }

    let path = traceback(&dp, n, m);

    Ok(DtwResult { distance, path })
}

/// Computes open-end DTW for subsequence matching.
///
/// Finds the best alignment of `query` within `reference`, where
/// the end of `reference` does not need to be matched. This is useful
/// for finding a subsequence of `reference` that best matches `query`.
///
/// # Arguments
///
/// * `reference` - The longer reference time series
/// * `query` - The query/pattern to search for
/// * `config` - DTW configuration
///
/// # Returns
///
/// A tuple of (distance, best_end_index) where best_end_index is the
/// position in `reference` where the best match ends.
pub fn dtw_open_end(reference: &[f64], query: &[f64], config: &DtwConfig) -> Result<(f64, usize)> {
    if reference.is_empty() || query.is_empty() {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }

    let n = reference.len();
    let m = query.len();

    let mut dp = vec![f64::INFINITY; (n + 1) * (m + 1)];
    // Allow starting from any point in reference
    for i in 0..=n {
        dp[i * (m + 1)] = 0.0;
    }

    let idx = |i: usize, j: usize| -> usize { i * (m + 1) + j };

    for i in 1..=n {
        for j in 1..=m {
            let cost = config.distance_fn.compute(reference[i - 1], query[j - 1]);
            let prev = dp[idx(i - 1, j)]
                .min(dp[idx(i, j - 1)])
                .min(dp[idx(i - 1, j - 1)]);
            dp[idx(i, j)] = cost + prev;
        }
    }

    // Find the minimum distance at the last column (query fully matched)
    let mut best_dist = f64::INFINITY;
    let mut best_end = 0;
    for i in 1..=n {
        let d = dp[idx(i, m)];
        if d < best_dist {
            best_dist = d;
            best_end = i - 1; // convert to 0-indexed
        }
    }

    if best_dist.is_infinite() {
        return Err(MetricsError::CalculationError(
            "no valid alignment found".to_string(),
        ));
    }

    Ok((best_dist, best_end))
}

/// Checks whether cell (i, j) is within the constraint window.
fn is_within_constraint(
    i: usize,
    j: usize,
    n: usize,
    m: usize,
    constraint: &DtwConstraint,
) -> bool {
    match constraint {
        DtwConstraint::None => true,
        DtwConstraint::SakoeChibaBand(w) => {
            let ri = i as f64 / n.max(1) as f64;
            let rj = j as f64 / m.max(1) as f64;
            (ri - rj).abs() <= *w
        }
        DtwConstraint::ItakuraParallelogram(s) => {
            // The Itakura parallelogram constrains the warping path to lie
            // within slopes 1/s and s around the diagonal.
            let n_f = n as f64;
            let m_f = m as f64;
            let i_f = i as f64;
            let j_f = j as f64;

            if n_f == 0.0 || m_f == 0.0 {
                return true;
            }

            // Forward constraints from (0,0)
            let j_min_fwd = i_f * (1.0 / s) * (m_f / n_f);
            let j_max_fwd = i_f * (*s) * (m_f / n_f);

            // Backward constraints from (n-1, m-1)
            let remaining_i = n_f - 1.0 - i_f;
            let j_min_bwd = m_f - 1.0 - remaining_i * (*s) * (m_f / n_f);
            let j_max_bwd = m_f - 1.0 - remaining_i * (1.0 / s) * (m_f / n_f);

            let j_min = j_min_fwd.max(j_min_bwd);
            let j_max = j_max_fwd.min(j_max_bwd);

            j_f >= j_min - 0.5 && j_f <= j_max + 0.5
        }
    }
}

/// Traces back through the DP matrix to find the optimal alignment path.
fn traceback(dp: &[f64], n: usize, m: usize) -> Vec<(usize, usize)> {
    let idx = |i: usize, j: usize| -> usize { i * (m + 1) + j };

    let mut path = Vec::with_capacity(n + m);
    let mut i = n;
    let mut j = m;

    path.push((i - 1, j - 1));

    while i > 1 || j > 1 {
        if i == 1 {
            j -= 1;
        } else if j == 1 {
            i -= 1;
        } else {
            let diag = dp[idx(i - 1, j - 1)];
            let left = dp[idx(i, j - 1)];
            let up = dp[idx(i - 1, j)];

            if diag <= left && diag <= up {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i - 1, j - 1));
    }

    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical_sequences() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = DtwConfig::default();
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        assert!(
            result.distance < 1e-10,
            "identical sequences should have DTW=0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_shifted_sequence() {
        // b is a shifted version of a - DTW should handle this well
        let a = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 2.0, 1.0];
        let config = DtwConfig::default();
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        assert!(
            result.distance < 2.0,
            "shifted sequences should have small DTW, got {}",
            result.distance
        );
        assert!(!result.path.is_empty(), "path should not be empty");
    }

    #[test]
    fn test_dtw_known_distance() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let config = DtwConfig::default();
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        // Each step costs |1 - 2| = 1, and 3 diagonal steps, so distance = 3.0
        assert!(
            (result.distance - 3.0).abs() < 1e-10,
            "expected DTW=3.0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_empty_input() {
        let config = DtwConfig::default();
        assert!(dtw_distance(&[], &[1.0], &config).is_err());
        assert!(dtw_distance(&[1.0], &[], &config).is_err());
    }

    #[test]
    fn test_dtw_sakoe_chiba() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = DtwConfig {
            distance_fn: DistanceFunction::Euclidean,
            constraint: DtwConstraint::SakoeChibaBand(0.3),
        };
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        assert!(
            result.distance < 1e-10,
            "identical should be 0 with band constraint, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_itakura() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = DtwConfig {
            distance_fn: DistanceFunction::Euclidean,
            constraint: DtwConstraint::ItakuraParallelogram(2.0),
        };
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        assert!(
            result.distance < 1e-10,
            "identical should be 0 with Itakura constraint, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_squared_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let config = DtwConfig {
            distance_fn: DistanceFunction::SquaredEuclidean,
            constraint: DtwConstraint::None,
        };
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        // Each step costs (0-1)^2 = 1, total = 3
        assert!(
            (result.distance - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_path_validity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 3.0];
        let config = DtwConfig::default();
        let result = dtw_distance(&a, &b, &config).expect("should succeed");
        let path = &result.path;

        // Path should start at (0,0) and end at (n-1, m-1)
        assert_eq!(path[0], (0, 0), "path should start at (0,0)");
        assert_eq!(
            path[path.len() - 1],
            (2, 1),
            "path should end at (n-1, m-1)"
        );

        // Path should be monotonically non-decreasing
        for w in path.windows(2) {
            assert!(w[1].0 >= w[0].0, "i should be non-decreasing");
            assert!(w[1].1 >= w[0].1, "j should be non-decreasing");
        }
    }

    #[test]
    fn test_dtw_nd() {
        let a = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]; // 3 points in 2D
        let b = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]; // identical
        let config = DtwConfig::default();
        let result = dtw_distance_nd(&a, &b, 2, &config).expect("should succeed");
        assert!(
            result.distance < 1e-10,
            "identical ND sequences should have DTW=0, got {}",
            result.distance
        );
    }

    #[test]
    fn test_dtw_nd_different() {
        let a = vec![0.0, 0.0, 1.0, 1.0]; // 2 points in 2D
        let b = vec![1.0, 1.0, 2.0, 2.0]; // shifted
        let config = DtwConfig::default();
        let result = dtw_distance_nd(&a, &b, 2, &config).expect("should succeed");
        assert!(
            result.distance > 0.0,
            "different ND sequences should have positive DTW"
        );
    }

    #[test]
    fn test_dtw_open_end() {
        // Reference contains query as a subsequence
        let reference = vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0];
        let query = vec![1.0, 2.0, 3.0];
        let config = DtwConfig::default();
        let (dist, end_idx) = dtw_open_end(&reference, &query, &config).expect("should succeed");
        assert!(
            dist < 1e-10,
            "exact subsequence should have distance ~0, got {dist}"
        );
        assert_eq!(end_idx, 4, "best match should end at index 4");
    }

    #[test]
    fn test_dtw_open_end_empty() {
        let config = DtwConfig::default();
        assert!(dtw_open_end(&[], &[1.0], &config).is_err());
    }
}
