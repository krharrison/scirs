//! SIMD-accelerated clustering operations for K-means, GMM, and DBSCAN.
//!
//! Provides cluster assignment, centroid update, GMM E-step,
//! epsilon-neighborhood queries, distortion, and linkage distance computation.

use super::distance::{
    should_use_simd, simd_distance, simd_pairwise_condensed_distances,
    simd_squared_euclidean_distance, SimdClusterConfig, SimdDistanceMetric,
};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// K-means: Cluster Assignment
// ---------------------------------------------------------------------------

/// Assign each data point to its nearest centroid using SIMD-accelerated distances.
///
/// Returns both the label assignments and the distances to the nearest centroid.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `centroids` - Centroid positions (k x n_features)
/// * `metric` - Distance metric to use
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Tuple of `(labels, distances)` where `labels[i]` is the assigned cluster for sample `i`
/// and `distances[i]` is the distance from sample `i` to its assigned centroid.
///
/// # Errors
///
/// Returns error if data and centroids have incompatible feature dimensions.
pub fn simd_assign_clusters<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let n_centroids = centroids.shape()[0];

    if centroids.shape()[1] != n_features {
        return Err(ClusteringError::InvalidInput(format!(
            "Data has {} features but centroids have {} features",
            n_features,
            centroids.shape()[1]
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let use_parallel =
        cfg.enable_parallel && is_parallel_enabled() && n_samples > cfg.parallel_chunk_size;

    if use_parallel {
        simd_assign_clusters_parallel(data, centroids, metric, cfg)
    } else {
        simd_assign_clusters_sequential(data, centroids, metric, cfg)
    }
}

/// Sequential cluster assignment with SIMD distances.
fn simd_assign_clusters_sequential<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];
    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::from_elem(n_samples, F::infinity());

    for i in 0..n_samples {
        let point = data.row(i);
        for j in 0..n_centroids {
            let centroid = centroids.row(j);
            let dist = simd_distance(point, centroid, metric, Some(config))?;
            if dist < distances[i] {
                distances[i] = dist;
                labels[i] = j;
            }
        }
    }

    Ok((labels, distances))
}

/// Parallel cluster assignment with SIMD distances.
fn simd_assign_clusters_parallel<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_centroids = centroids.shape()[0];

    let results: Vec<(usize, F)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let point = data.row(i);
            let mut best_label = 0;
            let mut best_dist = F::infinity();
            for j in 0..n_centroids {
                let centroid = centroids.row(j);
                let dist = simd_distance(point, centroid, metric, Some(config))
                    .unwrap_or_else(|_| F::infinity());
                if dist < best_dist {
                    best_dist = dist;
                    best_label = j;
                }
            }
            (best_label, best_dist)
        })
        .collect();

    let mut labels = Array1::zeros(n_samples);
    let mut distances = Array1::zeros(n_samples);
    for (i, (label, dist)) in results.into_iter().enumerate() {
        labels[i] = label;
        distances[i] = dist;
    }

    Ok((labels, distances))
}

// ---------------------------------------------------------------------------
// K-means: Centroid Update
// ---------------------------------------------------------------------------

/// Recompute centroids from cluster assignments using SIMD accumulation.
///
/// For each cluster k, the new centroid is the element-wise mean of all data
/// points assigned to cluster k. SIMD is used for the accumulation step.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `labels` - Cluster assignment for each sample (length n_samples)
/// * `k` - Number of clusters
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// New centroid matrix (k x n_features). Empty clusters retain all-zero centroids.
///
/// # Errors
///
/// Returns error if labels length does not match the number of samples.
pub fn simd_centroid_update<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    config: Option<&SimdClusterConfig>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Labels length ({}) must match number of samples ({})",
            labels.len(),
            n_samples
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let use_simd = should_use_simd(n_features, cfg);

    let use_parallel = cfg.enable_parallel && is_parallel_enabled() && k > 4;

    if use_parallel {
        simd_centroid_update_parallel(data, labels, k, n_features, use_simd)
    } else {
        simd_centroid_update_sequential(data, labels, k, n_features, use_simd)
    }
}

/// Sequential centroid update with SIMD accumulation.
fn simd_centroid_update_sequential<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    n_features: usize,
    use_simd: bool,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let mut centroids = Array2::zeros((k, n_features));
    let mut counts = vec![0usize; k];

    // Accumulate sums per cluster
    for i in 0..n_samples {
        let cluster = labels[i];
        if cluster >= k {
            continue; // skip invalid labels
        }
        counts[cluster] += 1;

        if use_simd {
            let current = centroids.slice(s![cluster, ..]).to_owned();
            let point = data.row(i);
            let sum = F::simd_add(&current.view(), &point);
            centroids.slice_mut(s![cluster, ..]).assign(&sum);
        } else {
            for j in 0..n_features {
                centroids[[cluster, j]] = centroids[[cluster, j]] + data[[i, j]];
            }
        }
    }

    // Divide by counts to get means
    for cluster in 0..k {
        if counts[cluster] > 0 {
            let count_f = F::from_usize(counts[cluster]).unwrap_or_else(|| F::one());
            if use_simd {
                let centroid_row = centroids.slice(s![cluster, ..]).to_owned();
                let count_arr = Array1::from_elem(n_features, count_f);
                let mean = F::simd_div(&centroid_row.view(), &count_arr.view());
                centroids.slice_mut(s![cluster, ..]).assign(&mean);
            } else {
                for j in 0..n_features {
                    centroids[[cluster, j]] = centroids[[cluster, j]] / count_f;
                }
            }
        }
    }

    Ok(centroids)
}

/// Parallel centroid update with SIMD accumulation.
fn simd_centroid_update_parallel<F>(
    data: ArrayView2<F>,
    labels: &Array1<usize>,
    k: usize,
    n_features: usize,
    use_simd: bool,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    // First, group sample indices by cluster
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &label) in labels.iter().enumerate() {
        if label < k {
            cluster_indices[label].push(i);
        }
    }

    // Compute centroids in parallel across clusters
    let centroid_rows: Vec<Array1<F>> = (0..k)
        .into_par_iter()
        .map(|cluster| {
            let indices = &cluster_indices[cluster];
            if indices.is_empty() {
                return Array1::zeros(n_features);
            }

            let mut sum = Array1::zeros(n_features);
            for &idx in indices {
                if use_simd {
                    let point = data.row(idx);
                    let new_sum = F::simd_add(&sum.view(), &point);
                    sum = new_sum;
                } else {
                    for j in 0..n_features {
                        sum[j] = sum[j] + data[[idx, j]];
                    }
                }
            }

            let count_f = F::from_usize(indices.len()).unwrap_or_else(|| F::one());
            if use_simd {
                let count_arr = Array1::from_elem(n_features, count_f);
                F::simd_div(&sum.view(), &count_arr.view())
            } else {
                sum.mapv(|v| v / count_f)
            }
        })
        .collect();

    // Assemble into matrix
    let mut centroids = Array2::zeros((k, n_features));
    for (cluster, row) in centroid_rows.into_iter().enumerate() {
        centroids.slice_mut(s![cluster, ..]).assign(&row);
    }

    Ok(centroids)
}

// ---------------------------------------------------------------------------
// GMM E-step: Responsibility Computation
// ---------------------------------------------------------------------------

/// Compute log-sum-exp of a 1D array using SIMD for numerical stability.
///
/// Computes `ln(sum(exp(x_i)))` in a numerically stable way by first
/// subtracting the maximum value.
///
/// # Arguments
///
/// * `values` - Input array of log-values
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// The log-sum-exp scalar value.
pub fn simd_logsumexp<F>(values: ArrayView1<F>, config: Option<&SimdClusterConfig>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    if values.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "Cannot compute logsumexp of empty array".to_string(),
        ));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let use_simd = should_use_simd(values.len(), cfg);

    if use_simd {
        // Find max for numerical stability
        let max_val = F::simd_max_element(&values);
        if max_val.is_infinite() && max_val.is_sign_negative() {
            return Ok(F::neg_infinity());
        }

        // Subtract max, exponentiate, sum
        let max_arr = Array1::from_elem(values.len(), max_val);
        let shifted = F::simd_sub(&values, &max_arr.view());
        let exp_vals = F::simd_exp(&shifted.view());
        let sum_exp = F::simd_sum(&exp_vals.view());

        Ok(max_val + sum_exp.ln())
    } else {
        // Scalar fallback
        let mut max_val = F::neg_infinity();
        for &v in values.iter() {
            if v > max_val {
                max_val = v;
            }
        }
        if max_val.is_infinite() && max_val.is_sign_negative() {
            return Ok(F::neg_infinity());
        }
        let mut sum = F::zero();
        for &v in values.iter() {
            sum = sum + (v - max_val).exp();
        }
        Ok(max_val + sum.ln())
    }
}

/// Compute log-sum-exp along rows of a 2D array using SIMD.
///
/// For each row i, computes `ln(sum_j(exp(values[i, j])))`.
///
/// # Arguments
///
/// * `values` - Input 2D array (n_rows x n_cols)
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// 1D array of length n_rows containing the row-wise log-sum-exp.
pub fn simd_logsumexp_rows<F>(
    values: ArrayView2<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_rows = values.shape()[0];
    let mut result = Array1::zeros(n_rows);

    for i in 0..n_rows {
        result[i] = simd_logsumexp(values.row(i), config)?;
    }

    Ok(result)
}

/// Compute GMM log-responsibilities for each sample and component using SIMD.
///
/// Given log-probabilities `log_prob[i, k]` (from the multivariate normal density
/// for sample i under component k) and log-weights `log_weights[k]`, this computes:
///
///   `log_resp[i, k] = log_prob[i, k] + log_weights[k]`
///   `log_norm[i] = logsumexp_k(log_resp[i, k])`
///   `resp[i, k] = exp(log_resp[i, k] - log_norm[i])`
///
/// # Arguments
///
/// * `log_prob` - Log-probability matrix (n_samples x n_components)
/// * `log_weights` - Log of mixture weights (length n_components)
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Tuple of (responsibilities, lower_bound) where:
/// - responsibilities is (n_samples x n_components) with rows summing to 1
/// - lower_bound is the mean log-likelihood across samples
///
/// # Errors
///
/// Returns error if dimensions are inconsistent.
pub fn simd_gmm_log_responsibilities<F>(
    log_prob: ArrayView2<F>,
    log_weights: ArrayView1<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<(Array2<F>, F)>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = log_prob.shape()[0];
    let n_components = log_prob.shape()[1];

    if log_weights.len() != n_components {
        return Err(ClusteringError::InvalidInput(format!(
            "log_weights length ({}) must match number of components ({})",
            log_weights.len(),
            n_components
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let use_simd = should_use_simd(n_components, cfg);

    // Step 1: Add log weights to each row
    let mut log_resp = log_prob.to_owned();
    for i in 0..n_samples {
        if use_simd {
            let row = log_resp.slice(s![i, ..]).to_owned();
            let weighted = F::simd_add(&row.view(), &log_weights);
            log_resp.slice_mut(s![i, ..]).assign(&weighted);
        } else {
            for k in 0..n_components {
                log_resp[[i, k]] = log_resp[[i, k]] + log_weights[k];
            }
        }
    }

    // Step 2: Compute log-normalization per sample (logsumexp across components)
    let log_norm = simd_logsumexp_rows(log_resp.view(), Some(cfg))?;

    // Step 3: Compute responsibilities: exp(log_resp[i,k] - log_norm[i])
    let mut resp = Array2::zeros((n_samples, n_components));
    for i in 0..n_samples {
        if use_simd {
            let row = log_resp.slice(s![i, ..]).to_owned();
            let norm_arr = Array1::from_elem(n_components, log_norm[i]);
            let shifted = F::simd_sub(&row.view(), &norm_arr.view());
            let exp_vals = F::simd_exp(&shifted.view());
            resp.slice_mut(s![i, ..]).assign(&exp_vals);
        } else {
            for k in 0..n_components {
                resp[[i, k]] = (log_resp[[i, k]] - log_norm[i]).exp();
            }
        }
    }

    // Step 4: Compute mean log-likelihood as lower bound
    let lower_bound = if use_simd {
        F::simd_sum(&log_norm.view()) / F::from_usize(n_samples).unwrap_or_else(|| F::one())
    } else {
        let mut sum = F::zero();
        for &v in log_norm.iter() {
            sum = sum + v;
        }
        sum / F::from_usize(n_samples).unwrap_or_else(|| F::one())
    };

    Ok((resp, lower_bound))
}

/// Compute the weighted mean for GMM M-step using SIMD.
///
/// Computes the weighted mean of data points for a single component:
///   `mean_k = sum_i(resp[i,k] * data[i,:]) / sum_i(resp[i,k])`
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `responsibilities` - Responsibility values for component k (length n_samples)
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Weighted mean vector (length n_features).
///
/// # Errors
///
/// Returns error if dimensions are inconsistent or total responsibility is zero.
pub fn simd_gmm_weighted_mean<F>(
    data: ArrayView2<F>,
    responsibilities: ArrayView1<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if responsibilities.len() != n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Responsibilities length ({}) must match number of samples ({})",
            responsibilities.len(),
            n_samples
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let use_simd = should_use_simd(n_features, cfg);

    let total_resp = if use_simd {
        F::simd_sum(&responsibilities)
    } else {
        let mut s = F::zero();
        for &r in responsibilities.iter() {
            s = s + r;
        }
        s
    };

    if total_resp <= F::zero() {
        return Err(ClusteringError::ComputationError(
            "Total responsibility is zero or negative; cannot compute weighted mean".to_string(),
        ));
    }

    let mut weighted_sum = Array1::zeros(n_features);
    for i in 0..n_samples {
        let r = responsibilities[i];
        if use_simd {
            let point = data.row(i);
            let r_arr = Array1::from_elem(n_features, r);
            let weighted_point = F::simd_mul(&point, &r_arr.view());
            let new_sum = F::simd_add(&weighted_sum.view(), &weighted_point.view());
            weighted_sum = new_sum;
        } else {
            for j in 0..n_features {
                weighted_sum[j] = weighted_sum[j] + data[[i, j]] * r;
            }
        }
    }

    // Divide by total responsibility
    if use_simd {
        let total_arr = Array1::from_elem(n_features, total_resp);
        Ok(F::simd_div(&weighted_sum.view(), &total_arr.view()))
    } else {
        Ok(weighted_sum.mapv(|v| v / total_resp))
    }
}

// ---------------------------------------------------------------------------
// DBSCAN: Epsilon-Neighborhood Queries
// ---------------------------------------------------------------------------

/// Find all points within epsilon distance of a query point using SIMD.
///
/// This is the core operation for DBSCAN: for a given query point, find all
/// data points whose distance is at most `eps`.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `query_idx` - Index of the query point in data
/// * `eps` - Epsilon threshold for neighborhood
/// * `metric` - Distance metric to use
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Vector of indices of points within the epsilon neighborhood (excluding the query itself).
///
/// # Errors
///
/// Returns error if query_idx is out of bounds.
pub fn simd_epsilon_neighborhood<F>(
    data: ArrayView2<F>,
    query_idx: usize,
    eps: F,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<Vec<usize>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    if query_idx >= n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Query index {} is out of bounds (data has {} samples)",
            query_idx, n_samples
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let query = data.row(query_idx);
    let mut neighbors = Vec::new();

    for j in 0..n_samples {
        if j == query_idx {
            continue;
        }
        let dist = simd_distance(query, data.row(j), metric, Some(cfg))?;
        if dist <= eps {
            neighbors.push(j);
        }
    }

    Ok(neighbors)
}

/// Compute epsilon neighborhoods for all points in a dataset using SIMD.
///
/// This precomputes the full neighborhood structure needed by DBSCAN.
/// For large datasets, this uses parallel processing.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `eps` - Epsilon threshold for neighborhood
/// * `metric` - Distance metric to use
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Vector of vectors: `result[i]` contains the indices of points within
/// epsilon distance of point i (excluding i itself).
///
/// # Errors
///
/// Returns error if data is empty.
pub fn simd_batch_epsilon_neighborhood<F>(
    data: ArrayView2<F>,
    eps: F,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<Vec<Vec<usize>>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data must have at least one sample".to_string(),
        ));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let use_parallel =
        cfg.enable_parallel && is_parallel_enabled() && n_samples > cfg.parallel_chunk_size;

    if use_parallel {
        simd_batch_neighborhood_parallel(data, eps, metric, cfg)
    } else {
        simd_batch_neighborhood_sequential(data, eps, metric, cfg)
    }
}

/// Sequential batch epsilon neighborhood computation.
fn simd_batch_neighborhood_sequential<F>(
    data: ArrayView2<F>,
    eps: F,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
) -> Result<Vec<Vec<usize>>>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    // Compute using the symmetric property: if dist(i,j) <= eps then j is
    // neighbor of i AND i is neighbor of j.
    let mut scratch: Vec<Vec<usize>> = (0..n_samples).map(|_| Vec::new()).collect();

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = simd_distance(data.row(i), data.row(j), metric, Some(config))?;
            if dist <= eps {
                scratch[i].push(j);
                scratch[j].push(i);
            }
        }
    }

    let neighborhoods = scratch;
    Ok(neighborhoods)
}

/// Parallel batch epsilon neighborhood computation.
fn simd_batch_neighborhood_parallel<F>(
    data: ArrayView2<F>,
    eps: F,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
) -> Result<Vec<Vec<usize>>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];

    // Parallel per-row neighborhood computation
    let neighborhoods: Vec<Vec<usize>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let query = data.row(i);
            let mut neighbors = Vec::new();
            for j in 0..n_samples {
                if j == i {
                    continue;
                }
                let dist = simd_distance(query, data.row(j), metric, Some(config))
                    .unwrap_or_else(|_| F::infinity());
                if dist <= eps {
                    neighbors.push(j);
                }
            }
            neighbors
        })
        .collect();

    Ok(neighborhoods)
}

// ---------------------------------------------------------------------------
// Compute distortion (inertia) for K-means
// ---------------------------------------------------------------------------

/// Compute the total distortion (sum of squared distances to assigned centroids)
/// for k-means using SIMD.
///
/// Distortion = `sum_i || data[i] - centroids[labels[i]] ||^2`
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `centroids` - Centroid positions (k x n_features)
/// * `labels` - Cluster assignment for each sample
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Total distortion (inertia).
///
/// # Errors
///
/// Returns error if dimensions are inconsistent.
pub fn simd_compute_distortion<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
    labels: &Array1<usize>,
    config: Option<&SimdClusterConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Labels length ({}) must match number of samples ({})",
            labels.len(),
            n_samples
        )));
    }

    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);
    let mut total = F::zero();

    for i in 0..n_samples {
        let cluster = labels[i];
        if cluster < centroids.shape()[0] {
            let sq_dist =
                simd_squared_euclidean_distance(data.row(i), centroids.row(cluster), Some(cfg))?;
            total = total + sq_dist;
        }
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// Hierarchical: SIMD-accelerated linkage distance helpers
// ---------------------------------------------------------------------------

/// Compute all pairwise distances needed for hierarchical linkage using SIMD.
///
/// This produces a condensed distance vector suitable for use with the
/// hierarchical clustering linkage algorithms.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `metric` - Distance metric to use
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Condensed distance vector of length n*(n-1)/2.
///
/// # Errors
///
/// Returns error if data is empty.
pub fn simd_linkage_distances<F>(
    data: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    simd_pairwise_condensed_distances(data, metric, config)
}

// ---------------------------------------------------------------------------
// Tests
