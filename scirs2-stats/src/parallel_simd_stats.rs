//! Parallel SIMD-optimized statistical operations for v0.2.0
//!
//! This module combines Rayon parallel processing with SIMD acceleration
//! for maximum performance on large datasets.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use scirs2_core::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Compute correlation matrix in parallel using SIMD operations
///
/// This function combines Rayon parallel processing with SIMD acceleration
/// to efficiently compute correlation matrices for large datasets.
/// Expected speedup: 2-3x over serial SIMD, 5-10x over scalar.
///
/// # Arguments
///
/// * `data` - 2D array where each column is a variable
/// * `method` - Correlation method ("pearson", "spearman")
///
/// # Returns
///
/// Correlation matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_stats::parallel_simd_stats::corrcoef_parallel_simd;
///
/// // Use non-constant data to avoid degenerate correlation (zero variance)
/// let data = Array2::from_shape_fn((100, 5), |(i, j)| (i as f64 * 0.1 + j as f64).sin());
/// let corr = corrcoef_parallel_simd(&data.view(), "pearson").expect("Failed");
/// assert_eq!(corr.shape(), &[5, 5]);
/// ```
pub fn corrcoef_parallel_simd<F>(data: &ArrayView2<F>, method: &str) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync,
{
    let n_vars = data.ncols();
    let n_obs = data.nrows();

    if n_obs < 2 {
        return Err(StatsError::invalid_argument(
            "Need at least 2 observations to compute correlation",
        ));
    }

    // Create output matrix
    let mut corr_matrix = Array2::zeros((n_vars, n_vars));

    // Collect indices for upper triangle
    let mut indices = Vec::new();
    for i in 0..n_vars {
        for j in i..n_vars {
            indices.push((i, j));
        }
    }

    // Compute correlations in parallel
    let correlations: Vec<((usize, usize), F)> = indices
        .par_iter()
        .map(|&(i, j)| {
            if i == j {
                ((i, j), F::one())
            } else {
                let var_i = data.slice(s![.., i]);
                let var_j = data.slice(s![.., j]);

                let corr = match method {
                    "pearson" => {
                        use crate::correlation_simd::pearson_r_simd;
                        pearson_r_simd(&var_i, &var_j).unwrap_or_else(|_| F::zero())
                    }
                    "spearman" => {
                        use crate::correlation_simd_enhanced::spearman_r_simd;
                        spearman_r_simd(&var_i, &var_j).unwrap_or_else(|_| F::zero())
                    }
                    _ => F::zero(),
                };

                ((i, j), corr)
            }
        })
        .collect();

    // Fill matrix
    for ((i, j), corr) in correlations {
        corr_matrix[(i, j)] = corr;
        if i != j {
            corr_matrix[(j, i)] = corr;
        }
    }

    Ok(corr_matrix)
}

/// Compute covariance matrix in parallel using SIMD operations
///
/// Combines parallel processing with SIMD for efficient covariance computation.
///
/// # Arguments
///
/// * `data` - 2D array where each column is a variable
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
///
/// Covariance matrix
pub fn covariance_matrix_parallel_simd<F>(
    data: &ArrayView2<F>,
    ddof: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync,
{
    let n_vars = data.ncols();
    let n_obs = data.nrows();

    if n_obs <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough observations for the given degrees of freedom",
        ));
    }

    // Compute means in parallel
    let means: Vec<F> = (0..n_vars)
        .into_par_iter()
        .map(|i| {
            let var = data.slice(s![.., i]);
            use crate::descriptive_simd::mean_simd;
            mean_simd(&var).unwrap_or_else(|_| F::zero())
        })
        .collect();

    // Create output matrix
    let mut cov_matrix = Array2::zeros((n_vars, n_vars));

    // Collect indices for upper triangle
    let mut indices = Vec::new();
    for i in 0..n_vars {
        for j in i..n_vars {
            indices.push((i, j));
        }
    }

    let optimizer = AutoOptimizer::new();

    // Compute covariances in parallel
    let covariances: Vec<((usize, usize), F)> = indices
        .par_iter()
        .map(|&(i, j)| {
            let var_i = data.slice(s![.., i]);
            let var_j = data.slice(s![.., j]);

            let cov = if optimizer.should_use_simd(n_obs) {
                // SIMD path
                let mean_i_array = Array1::from_elem(n_obs, means[i]);
                let mean_j_array = Array1::from_elem(n_obs, means[j]);

                let dev_i = F::simd_sub(&var_i, &mean_i_array.view());
                let dev_j = F::simd_sub(&var_j, &mean_j_array.view());
                let products = F::simd_mul(&dev_i.view(), &dev_j.view());
                let sum_products = F::simd_sum(&products.view());

                sum_products / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for k in 0..n_obs {
                    let dev_i = var_i[k] - means[i];
                    let dev_j = var_j[k] - means[j];
                    sum = sum + dev_i * dev_j;
                }
                sum / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            };

            ((i, j), cov)
        })
        .collect();

    // Fill matrix
    for ((i, j), cov) in covariances {
        cov_matrix[(i, j)] = cov;
        if i != j {
            cov_matrix[(j, i)] = cov;
        }
    }

    Ok(cov_matrix)
}

/// Parallel bootstrap with SIMD acceleration
///
/// Performs bootstrap resampling in parallel with SIMD-accelerated statistics.
///
/// # Arguments
///
/// * `data` - Input data array
/// * `n_bootstrap` - Number of bootstrap samples
/// * `statistic` - Function to compute on each bootstrap sample
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Array of bootstrap statistics
pub fn bootstrap_parallel_simd<F, Stat>(
    data: &ArrayView1<F>,
    n_bootstrap: usize,
    statistic: Stat,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync,
    Stat: Fn(&ArrayView1<F>) -> StatsResult<F> + Send + Sync,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Data array cannot be empty"));
    }

    use scirs2_core::random::SeedableRng;

    // Generate seeds for each bootstrap sample
    let seeds: Vec<u64> = if let Some(s) = seed {
        let mut rng = scirs2_core::random::seeded_rng(s);
        (0..n_bootstrap)
            .map(|_| scirs2_core::random::Rng::random(&mut rng))
            .collect()
    } else {
        let mut rng = scirs2_core::random::thread_rng();
        (0..n_bootstrap)
            .map(|_| scirs2_core::random::Rng::random(&mut rng))
            .collect()
    };

    // Perform bootstrap in parallel
    let results: Vec<F> = seeds
        .par_iter()
        .map(|&s| {
            use crate::sampling_simd::bootstrap_simd;
            let bootstrap_sample =
                bootstrap_simd(data, data.len(), Some(s)).unwrap_or_else(|_| Array1::zeros(0));
            statistic(&bootstrap_sample.view()).unwrap_or_else(|_| F::zero())
        })
        .collect();

    Ok(Array1::from_vec(results))
}

/// Parallel computation of row-wise statistics with SIMD
///
/// Computes statistics for each row in parallel using SIMD acceleration.
///
/// # Arguments
///
/// * `data` - 2D array
/// * `statistic` - Function to compute on each row
///
/// # Returns
///
/// Array of statistics, one per row
pub fn row_statistics_parallel_simd<F, Stat>(
    data: &ArrayView2<F>,
    statistic: Stat,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync,
    Stat: Fn(&ArrayView1<F>) -> StatsResult<F> + Send + Sync,
{
    let n_rows = data.nrows();

    let results: Vec<F> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = data.slice(s![i, ..]);
            statistic(&row).unwrap_or_else(|_| F::zero())
        })
        .collect();

    Ok(Array1::from_vec(results))
}

/// Parallel pairwise distance computation with SIMD
///
/// Computes pairwise distances between all pairs of rows using
/// parallel processing and SIMD acceleration.
///
/// # Arguments
///
/// * `data` - 2D array where each row is a point
/// * `metric` - Distance metric ("euclidean", "manhattan", "cosine")
///
/// # Returns
///
/// Distance matrix
pub fn pairwise_distances_parallel_simd<F>(
    data: &ArrayView2<F>,
    metric: &str,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps + Send + Sync,
{
    let n_points = data.nrows();
    let mut dist_matrix = Array2::zeros((n_points, n_points));

    // Collect indices for upper triangle
    let mut indices = Vec::new();
    for i in 0..n_points {
        for j in i + 1..n_points {
            indices.push((i, j));
        }
    }

    // Compute distances in parallel
    let distances: Vec<((usize, usize), F)> = indices
        .par_iter()
        .map(|&(i, j)| {
            let point_i = data.slice(s![i, ..]);
            let point_j = data.slice(s![j, ..]);

            let dist = match metric {
                "euclidean" => {
                    let diff = F::simd_sub(&point_i, &point_j);
                    let sq = F::simd_mul(&diff.view(), &diff.view());
                    F::simd_sum(&sq.view()).sqrt()
                }
                "manhattan" => {
                    let diff = F::simd_sub(&point_i, &point_j);
                    let abs_diff = F::simd_abs(&diff.view());
                    F::simd_sum(&abs_diff.view())
                }
                "cosine" => {
                    let dot_prod = F::simd_dot(&point_i, &point_j);
                    let norm_i = F::simd_norm(&point_i);
                    let norm_j = F::simd_norm(&point_j);
                    F::one() - dot_prod / (norm_i * norm_j)
                }
                _ => F::zero(),
            };

            ((i, j), dist)
        })
        .collect();

    // Fill matrix (symmetric)
    for ((i, j), dist) in distances {
        dist_matrix[(i, j)] = dist;
        dist_matrix[(j, i)] = dist;
    }

    Ok(dist_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_corrcoef_parallel_simd() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];

        let corr = corrcoef_parallel_simd(&data.view(), "pearson").expect("Failed");

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(corr[(i, j)], corr[(j, i)], epsilon = 1e-10);
            }
        }

        // Check diagonal
        for i in 0..3 {
            assert_abs_diff_eq!(corr[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_covariance_matrix_parallel_simd() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let cov = covariance_matrix_parallel_simd(&data.view(), 1).expect("Failed");

        // Check symmetry
        assert_abs_diff_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-10);

        // Check diagonal (variances are positive)
        assert!(cov[(0, 0)] > 0.0);
        assert!(cov[(1, 1)] > 0.0);
    }

    #[test]
    fn test_pairwise_distances_parallel_simd() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let dist = pairwise_distances_parallel_simd(&data.view(), "euclidean").expect("Failed");

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(dist[(i, j)], dist[(j, i)], epsilon = 1e-10);
            }
        }

        // Check diagonal is zero
        for i in 0..3 {
            assert_abs_diff_eq!(dist[(i, i)], 0.0, epsilon = 1e-10);
        }
    }
}
