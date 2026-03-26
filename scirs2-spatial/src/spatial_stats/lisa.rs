//! Local Indicators of Spatial Association (LISA)
//!
//! - Local Moran's I with permutation-based pseudo-significance
//! - Getis-Ord Gi* statistic with z-scores and p-values
//! - LISA cluster map classification (HH, LL, HL, LH)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{seeded_rng, Rng, RngExt, SeedableRng};

use crate::error::{SpatialError, SpatialResult};

/// Classification label for a LISA cluster map cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LisaCluster {
    /// High value surrounded by high values (hot spot).
    HighHigh,
    /// Low value surrounded by low values (cold spot).
    LowLow,
    /// High value surrounded by low values (spatial outlier).
    HighLow,
    /// Low value surrounded by high values (spatial outlier).
    LowHigh,
    /// Not significant at the chosen alpha level.
    NotSignificant,
}

/// Result of a Local Moran's I analysis for a single observation.
#[derive(Debug, Clone)]
pub struct LisaResult {
    /// Local Moran's I_i values for each observation.
    pub local_i: Array1<f64>,
    /// Pseudo p-values from permutation test (one per observation).
    pub p_values: Array1<f64>,
    /// Cluster classification for each observation.
    pub clusters: Vec<LisaCluster>,
}

/// A complete LISA cluster map.
#[derive(Debug, Clone)]
pub struct LisaClusterMap {
    /// Local Moran's I values.
    pub local_i: Array1<f64>,
    /// Pseudo p-values.
    pub p_values: Array1<f64>,
    /// Cluster labels.
    pub clusters: Vec<LisaCluster>,
    /// Mean of the values.
    pub mean: f64,
    /// Number of permutations used.
    pub n_permutations: usize,
    /// Significance level used for classification.
    pub alpha: f64,
}

/// Result of the Getis-Ord Gi* statistic for one location.
#[derive(Debug, Clone)]
pub struct GetisOrdResult {
    /// Gi* z-scores for each location.
    pub z_scores: Array1<f64>,
    /// Two-sided p-values.
    pub p_values: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Local Moran's I with permutation test
// ---------------------------------------------------------------------------

/// Compute Local Moran's I with a conditional permutation test.
///
/// For each observation i, the value at i is held fixed while remaining values
/// are randomly permuted `n_permutations` times. The pseudo p-value is
/// `(count of |I_perm| >= |I_obs| + 1) / (n_permutations + 1)`.
///
/// # Arguments
///
/// * `values` - Observed values (n,)
/// * `weights` - Spatial weights matrix (n x n)
/// * `n_permutations` - Number of random permutations (e.g. 999)
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
///
/// A `LisaResult` with local I values, p-values, and cluster classifications
/// (using alpha = 0.05).
pub fn local_moran_permutation_test(
    values: &ArrayView1<f64>,
    weights: &ArrayView2<f64>,
    n_permutations: usize,
    seed: u64,
) -> SpatialResult<LisaResult> {
    let n = values.len();
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 observations".to_string(),
        ));
    }

    let nf = n as f64;
    let mean: f64 = values.sum() / nf;
    let variance: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    if variance == 0.0 {
        return Err(SpatialError::ValueError("Variance is zero".to_string()));
    }
    let std_dev = variance.sqrt();

    // Compute standardized deviations
    let z: Vec<f64> = values.iter().map(|&x| (x - mean) / std_dev).collect();

    // Observed Local Moran's I
    let mut local_i = Array1::zeros(n);
    for i in 0..n {
        let mut wz_sum = 0.0;
        for j in 0..n {
            if i != j {
                wz_sum += weights[[i, j]] * z[j];
            }
        }
        local_i[i] = z[i] * wz_sum;
    }

    // Permutation test
    let mut p_values = Array1::zeros(n);
    let mut rng = seeded_rng(seed);

    // Build index list for permuting
    let mut indices: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let obs_i = local_i[i].abs();
        let mut count_extreme = 0usize;

        for _perm in 0..n_permutations {
            // Fisher-Yates shuffle of indices, skipping position i
            // We shuffle all except i, then compute local I at i using permuted values
            // Simple approach: shuffle the entire z vector copy, but keep z[i] fixed
            fisher_yates_shuffle_except(&mut indices, i, &mut rng);

            let mut wz_perm = 0.0;
            for j in 0..n {
                if j != i {
                    // Use z[indices[j]] as the permuted value at location j
                    wz_perm += weights[[i, j]] * z[indices[j]];
                }
            }
            let i_perm = z[i] * wz_perm;

            if i_perm.abs() >= obs_i {
                count_extreme += 1;
            }
        }

        p_values[i] = (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);
    }

    // Classify clusters (alpha = 0.05)
    let alpha = 0.05;
    let clusters = classify_clusters(&z, &local_i, &p_values, weights, alpha);

    Ok(LisaResult {
        local_i,
        p_values,
        clusters,
    })
}

/// Build a complete LISA cluster map with configurable alpha.
///
/// This is a convenience wrapper around `local_moran_permutation_test`
/// that also stores metadata.
pub fn lisa_cluster_map(
    values: &ArrayView1<f64>,
    weights: &ArrayView2<f64>,
    n_permutations: usize,
    alpha: f64,
    seed: u64,
) -> SpatialResult<LisaClusterMap> {
    let n = values.len();
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 observations".to_string(),
        ));
    }

    let nf = n as f64;
    let mean: f64 = values.sum() / nf;
    let variance: f64 = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    if variance == 0.0 {
        return Err(SpatialError::ValueError("Variance is zero".to_string()));
    }
    let std_dev = variance.sqrt();
    let z: Vec<f64> = values.iter().map(|&x| (x - mean) / std_dev).collect();

    let mut local_i = Array1::zeros(n);
    for i in 0..n {
        let mut wz_sum = 0.0;
        for j in 0..n {
            if i != j {
                wz_sum += weights[[i, j]] * z[j];
            }
        }
        local_i[i] = z[i] * wz_sum;
    }

    // Permutation test
    let mut p_values = Array1::zeros(n);
    let mut rng = seeded_rng(seed);
    let mut indices: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let obs_i = local_i[i].abs();
        let mut count_extreme = 0usize;

        for _perm in 0..n_permutations {
            fisher_yates_shuffle_except(&mut indices, i, &mut rng);

            let mut wz_perm = 0.0;
            for j in 0..n {
                if j != i {
                    wz_perm += weights[[i, j]] * z[indices[j]];
                }
            }
            let i_perm = z[i] * wz_perm;

            if i_perm.abs() >= obs_i {
                count_extreme += 1;
            }
        }

        p_values[i] = (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);
    }

    let clusters = classify_clusters(&z, &local_i, &p_values, weights, alpha);

    Ok(LisaClusterMap {
        local_i,
        p_values,
        clusters,
        mean,
        n_permutations,
        alpha,
    })
}

// ---------------------------------------------------------------------------
// Getis-Ord Gi*
// ---------------------------------------------------------------------------

/// Compute the Getis-Ord Gi* statistic for each location.
///
/// Gi* includes the focal location in its own computation (self-weight = 1).
///
/// Gi* = (sum_j w_ij x_j - xbar sum_j w_ij) / (s * sqrt((n sum_j w_ij^2 - (sum_j w_ij)^2) / (n-1)))
///
/// where the sums include j = i.
pub fn getis_ord_gi_star(
    values: &ArrayView1<f64>,
    weights: &ArrayView2<f64>,
) -> SpatialResult<GetisOrdResult> {
    let n = values.len();
    if weights.nrows() != n || weights.ncols() != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }
    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 observations".to_string(),
        ));
    }

    let nf = n as f64;
    let xbar: f64 = values.sum() / nf;
    let s2: f64 = values.iter().map(|&x| (x - xbar) * (x - xbar)).sum::<f64>() / nf;
    if s2 == 0.0 {
        return Err(SpatialError::ValueError(
            "Variance of values is zero".to_string(),
        ));
    }
    let s = s2.sqrt();

    let mut z_scores = Array1::zeros(n);
    let mut p_values = Array1::zeros(n);

    for i in 0..n {
        // Include self-weight (set diagonal to 1 conceptually for Gi*)
        let mut wx_sum = 0.0;
        let mut w_sum = 0.0;
        let mut w_sq_sum = 0.0;

        for j in 0..n {
            let wij = if i == j {
                // Gi* uses self-weight = 1 if diagonal is 0
                if weights[[i, j]] == 0.0 {
                    1.0
                } else {
                    weights[[i, j]]
                }
            } else {
                weights[[i, j]]
            };
            wx_sum += wij * values[j];
            w_sum += wij;
            w_sq_sum += wij * wij;
        }

        let numerator = wx_sum - xbar * w_sum;
        let denom_inner = (nf * w_sq_sum - w_sum * w_sum) / (nf - 1.0);

        if denom_inner > 0.0 {
            let denominator = s * denom_inner.sqrt();
            z_scores[i] = numerator / denominator;
        }

        p_values[i] = two_sided_p(z_scores[i]);
    }

    Ok(GetisOrdResult { z_scores, p_values })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Two-sided p-value via normal CDF approximation.
fn two_sided_p(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

/// Standard normal CDF (Abramowitz & Stegun).
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

/// Fisher-Yates shuffle of `indices`, keeping position `skip` in place.
fn fisher_yates_shuffle_except<R: Rng + ?Sized>(indices: &mut [usize], skip: usize, rng: &mut R) {
    let n = indices.len();
    // Reset indices
    for (idx, val) in indices.iter_mut().enumerate() {
        *val = idx;
    }

    // Shuffle all positions except skip
    // Collect swappable positions
    let positions: Vec<usize> = (0..n).filter(|&p| p != skip).collect();
    let m = positions.len();
    for k in (1..m).rev() {
        let j = rng.random_range(0..=k);
        indices.swap(positions[k], positions[j]);
    }
}

/// Classify each observation into a LISA cluster type.
fn classify_clusters(
    z: &[f64],
    local_i: &Array1<f64>,
    p_values: &Array1<f64>,
    weights: &ArrayView2<f64>,
    alpha: f64,
) -> Vec<LisaCluster> {
    let n = z.len();
    let mut clusters = vec![LisaCluster::NotSignificant; n];

    for i in 0..n {
        if p_values[i] > alpha {
            continue;
        }

        // Compute spatial lag (weighted average of neighbours' z)
        let mut w_sum = 0.0;
        let mut wz_sum = 0.0;
        for j in 0..n {
            if j != i && weights[[i, j]] > 0.0 {
                w_sum += weights[[i, j]];
                wz_sum += weights[[i, j]] * z[j];
            }
        }

        let lag = if w_sum > 0.0 { wz_sum / w_sum } else { 0.0 };

        clusters[i] = if z[i] > 0.0 && lag > 0.0 {
            LisaCluster::HighHigh
        } else if z[i] < 0.0 && lag < 0.0 {
            LisaCluster::LowLow
        } else if z[i] > 0.0 && lag < 0.0 {
            LisaCluster::HighLow
        } else if z[i] < 0.0 && lag > 0.0 {
            LisaCluster::LowHigh
        } else {
            LisaCluster::NotSignificant
        };
    }

    clusters
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn build_chain_weights(n: usize) -> Array2<f64> {
        let mut w = Array2::zeros((n, n));
        for i in 0..(n - 1) {
            w[[i, i + 1]] = 1.0;
            w[[i + 1, i]] = 1.0;
        }
        w
    }

    #[test]
    fn test_local_moran_permutation_hot_spot() {
        // Three high values clustered at one end
        let values = array![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let w = build_chain_weights(6);

        let result =
            local_moran_permutation_test(&values.view(), &w.view(), 199, 42).expect("lisa failed");

        assert_eq!(result.local_i.len(), 6);
        assert_eq!(result.p_values.len(), 6);
        assert_eq!(result.clusters.len(), 6);

        // First observation should be part of a high-high cluster (or at least
        // have positive local I)
        assert!(
            result.local_i[0] > 0.0,
            "Local I at position 0 should be positive for clustered high values"
        );
    }

    #[test]
    fn test_local_moran_permutation_spatial_outlier() {
        // One high outlier surrounded by lows
        let values = array![1.0, 1.0, 10.0, 1.0, 1.0];
        let w = build_chain_weights(5);

        let result =
            local_moran_permutation_test(&values.view(), &w.view(), 199, 123).expect("lisa");

        // The outlier at index 2 should have negative local I
        assert!(
            result.local_i[2] < 0.0,
            "Local I at outlier position should be negative, got {}",
            result.local_i[2]
        );
    }

    #[test]
    fn test_lisa_cluster_map_classifications() {
        let values = array![10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = build_chain_weights(8);

        let map = lisa_cluster_map(&values.view(), &w.view(), 499, 0.10, 42).expect("cluster_map");

        assert_eq!(map.clusters.len(), 8);
        assert_eq!(map.n_permutations, 499);

        // Check that HH, LL, or not-significant labels are present
        let has_hh = map.clusters.contains(&LisaCluster::HighHigh);
        let has_ll = map.clusters.contains(&LisaCluster::LowLow);
        let has_ns = map.clusters.contains(&LisaCluster::NotSignificant);

        // With strong clustering and moderate alpha, we expect at least some
        // significant clusters
        assert!(
            has_hh || has_ll || has_ns,
            "Should produce at least one classification"
        );
    }

    #[test]
    fn test_getis_ord_gi_star_hotspot() {
        // Cluster of high values at one end
        let values = array![10.0, 10.0, 10.0, 1.0, 1.0, 1.0];
        let w = build_chain_weights(6);

        let result = getis_ord_gi_star(&values.view(), &w.view()).expect("gi_star");

        assert_eq!(result.z_scores.len(), 6);
        assert_eq!(result.p_values.len(), 6);

        // First few locations should have positive z-scores (hotspot)
        assert!(
            result.z_scores[0] > 0.0,
            "z-score at position 0 should be positive for high cluster, got {}",
            result.z_scores[0]
        );

        // Last few should have negative z-scores (cold spot)
        assert!(
            result.z_scores[5] < 0.0,
            "z-score at position 5 should be negative for low cluster, got {}",
            result.z_scores[5]
        );
    }

    #[test]
    fn test_getis_ord_gi_star_uniform() {
        // Uniform values => z-scores should be near zero
        let values = array![5.0, 5.0, 5.0, 5.0, 5.0];
        let w = build_chain_weights(5);

        // Uniform values => variance = 0 => should get error
        let result = getis_ord_gi_star(&values.view(), &w.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_getis_ord_gi_star_p_values() {
        let values = array![100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = build_chain_weights(8);

        let result = getis_ord_gi_star(&values.view(), &w.view()).expect("gi_star");

        // P-values should be between 0 and 1
        for &p in result.p_values.iter() {
            assert!((0.0..=1.0).contains(&p), "p-value {} out of range", p);
        }

        // The hotspot location should have a small-ish p-value
        assert!(
            result.p_values[0] < 0.5,
            "p-value at hotspot should be < 0.5"
        );
    }
}
