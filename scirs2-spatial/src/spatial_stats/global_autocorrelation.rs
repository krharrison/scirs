//! Enhanced global spatial autocorrelation with significance testing
//!
//! Provides Moran's I and Geary's C statistics with z-scores, p-values,
//! and spatial weights matrix utilities (distance band, inverse distance,
//! row standardization).

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};

use crate::error::{SpatialError, SpatialResult};

/// Sparse-ish spatial weights stored as a dense matrix with utility methods.
#[derive(Debug, Clone)]
pub struct SpatialWeights {
    /// Dense weights matrix (n x n). Zero entries mean no spatial relationship.
    pub weights: Array2<f64>,
    /// Whether the matrix has been row-standardized.
    pub row_standardized: bool,
}

impl SpatialWeights {
    /// Create from a raw dense weights matrix.
    pub fn from_matrix(weights: Array2<f64>) -> Self {
        Self {
            weights,
            row_standardized: false,
        }
    }

    /// Number of spatial units.
    pub fn n(&self) -> usize {
        self.weights.nrows()
    }

    /// Total sum of all weights (W).
    pub fn total_weight(&self) -> f64 {
        self.weights.sum()
    }

    /// Row-standardize in place: each row sums to 1 (if row sum > 0).
    pub fn row_standardize(&mut self) {
        let n = self.n();
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| self.weights[[i, j]]).sum();
            if row_sum > 0.0 {
                for j in 0..n {
                    self.weights[[i, j]] /= row_sum;
                }
            }
        }
        self.row_standardized = true;
    }

    /// Return the view of the inner matrix.
    pub fn view(&self) -> ArrayView2<f64> {
        self.weights.view()
    }
}

/// Result of a global spatial autocorrelation test.
#[derive(Debug, Clone)]
pub struct GlobalAutocorrelationResult {
    /// The statistic value (Moran's I or Geary's C).
    pub statistic: f64,
    /// Expected value of the statistic under H0 (spatial randomness).
    pub expected: f64,
    /// Variance of the statistic under the chosen assumption.
    pub variance: f64,
    /// Z-score: (statistic - expected) / sqrt(variance).
    pub z_score: f64,
    /// Two-sided p-value from the normal approximation.
    pub p_value: f64,
}

/// Row-standardize a weights matrix (convenience function returning a new SpatialWeights).
pub fn row_standardize_weights(weights: &ArrayView2<f64>) -> SpatialResult<SpatialWeights> {
    if weights.nrows() != weights.ncols() {
        return Err(SpatialError::DimensionError(
            "Weights matrix must be square".to_string(),
        ));
    }
    let mut sw = SpatialWeights::from_matrix(weights.to_owned());
    sw.row_standardize();
    Ok(sw)
}

/// Build a distance-band binary weights matrix.
///
/// Two points are neighbours if their Euclidean distance is at most `threshold`.
/// Diagonal entries are always zero.
pub fn distance_band_weights(
    coordinates: &ArrayView2<f64>,
    threshold: f64,
) -> SpatialResult<SpatialWeights> {
    let n = coordinates.nrows();
    let ndim = coordinates.ncols();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points".to_string(),
        ));
    }
    if threshold <= 0.0 {
        return Err(SpatialError::ValueError(
            "Distance threshold must be positive".to_string(),
        ));
    }

    let mut w = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for d in 0..ndim {
                let diff = coordinates[[i, d]] - coordinates[[j, d]];
                dist_sq += diff * diff;
            }
            if dist_sq.sqrt() <= threshold {
                w[[i, j]] = 1.0;
                w[[j, i]] = 1.0;
            }
        }
    }

    Ok(SpatialWeights::from_matrix(w))
}

/// Build an inverse-distance weights matrix.
///
/// `w_{ij} = 1 / d_{ij}^power` when `d_{ij} <= max_distance`, else 0.
/// Diagonal entries are zero.
pub fn inverse_distance_weights(
    coordinates: &ArrayView2<f64>,
    power: f64,
    max_distance: f64,
) -> SpatialResult<SpatialWeights> {
    let n = coordinates.nrows();
    let ndim = coordinates.ncols();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points".to_string(),
        ));
    }
    if max_distance <= 0.0 {
        return Err(SpatialError::ValueError(
            "max_distance must be positive".to_string(),
        ));
    }
    if power <= 0.0 {
        return Err(SpatialError::ValueError(
            "power must be positive".to_string(),
        ));
    }

    let mut w = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for d in 0..ndim {
                let diff = coordinates[[i, d]] - coordinates[[j, d]];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();
            if dist > 0.0 && dist <= max_distance {
                let wt = 1.0 / dist.powf(power);
                w[[i, j]] = wt;
                w[[j, i]] = wt;
            }
        }
    }

    Ok(SpatialWeights::from_matrix(w))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();
    0.5 * (1.0 + sign * y)
}

/// Two-sided p-value from a z-score using the normal approximation.
fn two_sided_p(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

// ---------------------------------------------------------------------------
// Moran's I test
// ---------------------------------------------------------------------------

/// Compute Moran's I with significance test (normality assumption).
///
/// `I = (N / W) * (sum_i sum_j w_ij (x_i - xbar)(x_j - xbar)) / (sum_i (x_i - xbar)^2)`
///
/// Expected value: `E[I] = -1/(N-1)`
///
/// Variance under normality:
/// `Var[I] = (N^2 * S1 - N * S2 + 3 * W^2) / (W^2 * (N^2 - 1)) - E[I]^2`
///
/// where S1 = sum_i sum_j (w_ij + w_ji)^2 / 2
///       S2 = sum_i (w_i. + w_.i)^2
pub fn moran_test(
    values: &ArrayView1<f64>,
    weights: &ArrayView2<f64>,
) -> SpatialResult<GlobalAutocorrelationResult> {
    let n = values.len();
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 observations for Moran's I test".to_string(),
        ));
    }

    let nf = n as f64;
    let mean = values.sum() / nf;

    // Deviations
    let dev: Vec<f64> = values.iter().map(|&x| x - mean).collect();
    let sum_sq: f64 = dev.iter().map(|d| d * d).sum();
    if sum_sq == 0.0 {
        return Err(SpatialError::ValueError(
            "Variance of values is zero".to_string(),
        ));
    }

    // W
    let w_total: f64 = weights.sum();
    if w_total == 0.0 {
        return Err(SpatialError::ValueError("Total weight is zero".to_string()));
    }

    // Numerator
    let mut numer = 0.0;
    for i in 0..n {
        for j in 0..n {
            numer += weights[[i, j]] * dev[i] * dev[j];
        }
    }

    let i_stat = (nf / w_total) * (numer / sum_sq);

    // Expected
    let e_i = -1.0 / (nf - 1.0);

    // S1 = 0.5 * sum_i sum_j (w_ij + w_ji)^2
    let mut s1 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let s = weights[[i, j]] + weights[[j, i]];
            s1 += s * s;
        }
    }
    s1 *= 0.5;

    // S2 = sum_i (w_i. + w_.i)^2
    let mut s2 = 0.0;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
        let col_sum: f64 = (0..n).map(|j| weights[[j, i]]).sum();
        let s = row_sum + col_sum;
        s2 += s * s;
    }

    // Variance (normality assumption)
    let w2 = w_total * w_total;
    let n2 = nf * nf;
    let var_i = (n2 * s1 - nf * s2 + 3.0 * w2) / (w2 * (n2 - 1.0)) - e_i * e_i;

    let var_i = var_i.max(0.0); // guard against floating-point negative
    let z = if var_i > 0.0 {
        (i_stat - e_i) / var_i.sqrt()
    } else {
        0.0
    };
    let p = two_sided_p(z);

    Ok(GlobalAutocorrelationResult {
        statistic: i_stat,
        expected: e_i,
        variance: var_i,
        z_score: z,
        p_value: p,
    })
}

// ---------------------------------------------------------------------------
// Geary's C test
// ---------------------------------------------------------------------------

/// Compute Geary's C with significance test (normality assumption).
///
/// `C = ((N-1) / (2W)) * (sum_i sum_j w_ij (x_i - x_j)^2) / (sum_i (x_i - xbar)^2)`
///
/// Expected value: `E[C] = 1`
///
/// Variance formula under normality uses moments S1, S2, W analogous to Moran.
pub fn geary_test(
    values: &ArrayView1<f64>,
    weights: &ArrayView2<f64>,
) -> SpatialResult<GlobalAutocorrelationResult> {
    let n = values.len();
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 observations for Geary's C test".to_string(),
        ));
    }

    let nf = n as f64;
    let mean = values.sum() / nf;

    // Sum of squares
    let sum_sq: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum();
    if sum_sq == 0.0 {
        return Err(SpatialError::ValueError(
            "Variance of values is zero".to_string(),
        ));
    }

    let w_total: f64 = weights.sum();
    if w_total == 0.0 {
        return Err(SpatialError::ValueError("Total weight is zero".to_string()));
    }

    // Numerator: sum_i sum_j w_ij (x_i - x_j)^2
    let mut numer = 0.0;
    for i in 0..n {
        for j in 0..n {
            let diff = values[i] - values[j];
            numer += weights[[i, j]] * diff * diff;
        }
    }

    let c_stat = ((nf - 1.0) / (2.0 * w_total)) * (numer / sum_sq);

    // Expected
    let e_c = 1.0;

    // S1, S2 for variance
    let mut s1 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let s = weights[[i, j]] + weights[[j, i]];
            s1 += s * s;
        }
    }
    s1 *= 0.5;

    let mut s2 = 0.0;
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| weights[[i, j]]).sum();
        let col_sum: f64 = (0..n).map(|j| weights[[j, i]]).sum();
        let s = row_sum + col_sum;
        s2 += s * s;
    }

    // Variance under normality (simplified formula)
    let n2m1 = nf * nf - 1.0;
    let var_c = ((2.0 * s1 + s2) * (nf - 1.0) - 4.0 * w_total * w_total)
        / (2.0 * (nf + 1.0) * w_total * w_total);

    let var_c = var_c.max(0.0);
    let z = if var_c > 0.0 {
        (c_stat - e_c) / var_c.sqrt()
    } else {
        0.0
    };
    let p = two_sided_p(z);

    Ok(GlobalAutocorrelationResult {
        statistic: c_stat,
        expected: e_c,
        variance: var_c,
        z_score: z,
        p_value: p,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_moran_test_clustered() {
        // Strongly clustered data on a line graph
        let values = array![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let mut w = Array2::zeros((6, 6));
        // Chain: 0-1-2-3-4-5
        for i in 0..5 {
            w[[i, i + 1]] = 1.0;
            w[[i + 1, i]] = 1.0;
        }

        let result = moran_test(&values.view(), &w.view()).expect("moran_test failed");

        // Clustered data should yield positive Moran's I (> expected)
        assert!(
            result.statistic > result.expected,
            "Moran's I = {} should exceed E[I] = {} for clustered data",
            result.statistic,
            result.expected
        );
        // Expected value
        assert_relative_eq!(result.expected, -1.0 / 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_moran_test_checkerboard() {
        // Checkerboard on a 4-node ring: alternating values
        let values = array![10.0, 0.0, 10.0, 0.0];
        let w = array![
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ];

        let result = moran_test(&values.view(), &w.view()).expect("moran_test failed");

        // Perfect negative autocorrelation => I should be negative
        assert!(
            result.statistic < 0.0,
            "Moran's I should be negative for checkerboard"
        );
    }

    #[test]
    fn test_geary_test_clustered() {
        let values = array![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let mut w = Array2::zeros((6, 6));
        for i in 0..5 {
            w[[i, i + 1]] = 1.0;
            w[[i + 1, i]] = 1.0;
        }

        let result = geary_test(&values.view(), &w.view()).expect("geary_test failed");

        // Clustered => C < 1
        assert!(
            result.statistic < 1.0,
            "Geary's C = {} should be < 1 for clustered data",
            result.statistic
        );
        assert_relative_eq!(result.expected, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geary_test_checkerboard() {
        let values = array![10.0, 0.0, 10.0, 0.0];
        let w = array![
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ];

        let result = geary_test(&values.view(), &w.view()).expect("geary_test failed");

        // Checkerboard => C > 1
        assert!(
            result.statistic > 1.0,
            "Geary's C = {} should be > 1 for checkerboard",
            result.statistic
        );
    }

    #[test]
    fn test_distance_band_weights() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
        let sw = distance_band_weights(&coords.view(), 1.5).expect("distance_band_weights failed");

        // Points 0,1,2 are within 1.5 of each other; 3 is far away
        assert_eq!(sw.weights[[0, 1]], 1.0);
        assert_eq!(sw.weights[[0, 2]], 1.0);
        assert_eq!(sw.weights[[0, 3]], 0.0);
        assert_eq!(sw.weights[[3, 0]], 0.0);
    }

    #[test]
    fn test_inverse_distance_weights() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let sw =
            inverse_distance_weights(&coords.view(), 1.0, 5.0).expect("inverse_distance_weights");

        // w(0,1) = 1/1 = 1.0, w(0,2) = 1/2 = 0.5
        assert_relative_eq!(sw.weights[[0, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sw.weights[[0, 2]], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_row_standardize() {
        let w = array![[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],];
        let sw = row_standardize_weights(&w.view()).expect("row_standardize");
        assert!(sw.row_standardized);
        // Row 0 had sum 2 => each entry becomes 0.5
        assert_relative_eq!(sw.weights[[0, 1]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(sw.weights[[0, 2]], 0.5, epsilon = 1e-10);
        // Row 1 had sum 1 => entry stays 1
        assert_relative_eq!(sw.weights[[1, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_moran_significance() {
        // With enough observations and strong clustering, p-value should be small
        let values = array![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let mut w = Array2::zeros((8, 8));
        for i in 0..7 {
            w[[i, i + 1]] = 1.0;
            w[[i + 1, i]] = 1.0;
        }

        let result = moran_test(&values.view(), &w.view()).expect("moran_test");
        // Strong clustering => large |z|
        assert!(
            result.z_score.abs() > 1.0,
            "z_score = {} should be > 1 for strongly clustered data",
            result.z_score
        );
        assert!(result.p_value < 0.5, "p_value should be moderate-to-small");
    }
}
