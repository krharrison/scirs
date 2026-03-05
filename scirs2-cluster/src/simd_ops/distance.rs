//! SIMD-accelerated distance computation for clustering algorithms.
//!
//! Provides Euclidean, Manhattan, pairwise, and condensed distance computation
//! using SIMD acceleration from `scirs2-core`.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for SIMD-accelerated clustering operations.
///
/// Controls thresholds for enabling SIMD, parallel processing, and
/// cache-friendly memory access patterns.
#[derive(Debug, Clone)]
pub struct SimdClusterConfig {
    /// Minimum vector length to trigger SIMD optimizations (default: 32)
    pub simd_threshold: usize,
    /// Enable parallel processing for large workloads (default: true)
    pub enable_parallel: bool,
    /// Chunk size for parallel processing (default: 512)
    pub parallel_chunk_size: usize,
    /// Block size for cache-friendly pairwise distance computation (default: 128)
    pub block_size: usize,
    /// Force SIMD usage regardless of heuristics (for testing; default: false)
    pub force_simd: bool,
}

impl Default for SimdClusterConfig {
    fn default() -> Self {
        Self {
            simd_threshold: 32,
            enable_parallel: true,
            parallel_chunk_size: 512,
            block_size: 128,
            force_simd: false,
        }
    }
}

/// Determine whether SIMD should be used for a given problem size.
pub(super) fn should_use_simd(n_elements: usize, config: &SimdClusterConfig) -> bool {
    let caps = PlatformCapabilities::detect();
    let optimizer = AutoOptimizer::new();
    config.force_simd
        || (caps.simd_available
            && (optimizer.should_use_simd(n_elements) || n_elements >= config.simd_threshold))
}

// ---------------------------------------------------------------------------
// Distance Computation
// ---------------------------------------------------------------------------

/// Distance metric for SIMD-accelerated operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdDistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Squared Euclidean distance (avoids sqrt for speed)
    SquaredEuclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
}

/// Compute Euclidean distance between two vectors using SIMD.
///
/// Falls back to scalar computation when SIMD is not beneficial.
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if vectors have different lengths.
pub fn simd_euclidean_distance<F>(
    a: ArrayView1<F>,
    b: ArrayView1<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    if a.len() != b.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Vectors must have the same length: got {} and {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(F::zero());
    }
    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    if should_use_simd(a.len(), cfg) {
        let diff = F::simd_sub(&a, &b);
        Ok(F::simd_norm(&diff.view()))
    } else {
        let mut sum = F::zero();
        for i in 0..a.len() {
            let d = a[i] - b[i];
            sum = sum + d * d;
        }
        Ok(sum.sqrt())
    }
}

/// Compute squared Euclidean distance between two vectors using SIMD.
///
/// This avoids the square root and is preferred when only relative ordering
/// of distances matters (e.g., nearest-neighbor search in k-means).
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if vectors have different lengths.
pub fn simd_squared_euclidean_distance<F>(
    a: ArrayView1<F>,
    b: ArrayView1<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    if a.len() != b.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Vectors must have the same length: got {} and {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(F::zero());
    }
    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    if should_use_simd(a.len(), cfg) {
        let diff = F::simd_sub(&a, &b);
        let sq = F::simd_mul(&diff.view(), &diff.view());
        Ok(F::simd_sum(&sq.view()))
    } else {
        let mut sum = F::zero();
        for i in 0..a.len() {
            let d = a[i] - b[i];
            sum = sum + d * d;
        }
        Ok(sum)
    }
}

/// Compute Manhattan distance between two vectors using SIMD.
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if vectors have different lengths.
pub fn simd_manhattan_distance<F>(
    a: ArrayView1<F>,
    b: ArrayView1<F>,
    config: Option<&SimdClusterConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    if a.len() != b.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "Vectors must have the same length: got {} and {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(F::zero());
    }
    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    if should_use_simd(a.len(), cfg) {
        let diff = F::simd_sub(&a, &b);
        let abs_diff = F::simd_abs(&diff.view());
        Ok(F::simd_sum(&abs_diff.view()))
    } else {
        let mut sum = F::zero();
        for i in 0..a.len() {
            sum = sum + (a[i] - b[i]).abs();
        }
        Ok(sum)
    }
}

/// Compute distance between two vectors using a specified metric.
///
/// Dispatches to the appropriate SIMD-accelerated distance function.
///
/// # Errors
///
/// Returns `ClusteringError::InvalidInput` if vectors have different lengths.
pub fn simd_distance<F>(
    a: ArrayView1<F>,
    b: ArrayView1<F>,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    match metric {
        SimdDistanceMetric::Euclidean => simd_euclidean_distance(a, b, config),
        SimdDistanceMetric::SquaredEuclidean => simd_squared_euclidean_distance(a, b, config),
        SimdDistanceMetric::Manhattan => simd_manhattan_distance(a, b, config),
    }
}

// ---------------------------------------------------------------------------
// Pairwise Distance Matrix
// ---------------------------------------------------------------------------

/// Compute the full pairwise distance matrix for a dataset using SIMD.
///
/// Returns an n x n symmetric distance matrix. For large datasets,
/// consider using [`simd_pairwise_condensed_distances`] instead to save memory.
///
/// # Arguments
///
/// * `data` - Input data matrix (n_samples x n_features)
/// * `metric` - Distance metric to use
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Symmetric distance matrix of shape (n_samples x n_samples).
///
/// # Errors
///
/// Returns error if data is empty.
pub fn simd_pairwise_distance_matrix<F>(
    data: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<Array2<F>>
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
    let mut distances = Array2::zeros((n_samples, n_samples));

    let use_parallel =
        cfg.enable_parallel && is_parallel_enabled() && n_samples > cfg.parallel_chunk_size;

    if use_parallel {
        simd_pairwise_matrix_parallel(data, metric, cfg, &mut distances);
    } else {
        simd_pairwise_matrix_sequential(data, metric, cfg, &mut distances)?;
    }

    Ok(distances)
}

/// Sequential pairwise distance matrix computation with SIMD.
fn simd_pairwise_matrix_sequential<F>(
    data: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
    distances: &mut Array2<F>,
) -> Result<()>
where
    F: Float + FromPrimitive + Debug + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    let block_size = config.block_size;

    for block_i in (0..n_samples).step_by(block_size) {
        let end_i = (block_i + block_size).min(n_samples);
        for block_j in (block_i..n_samples).step_by(block_size) {
            let end_j = (block_j + block_size).min(n_samples);

            for i in block_i..end_i {
                let start_j = if block_i == block_j { i + 1 } else { block_j };
                for j in start_j..end_j {
                    let dist = simd_distance(data.row(i), data.row(j), metric, Some(config))?;
                    distances[[i, j]] = dist;
                    distances[[j, i]] = dist;
                }
            }
        }
    }
    Ok(())
}

/// Parallel pairwise distance matrix computation with SIMD.
fn simd_pairwise_matrix_parallel<F>(
    data: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: &SimdClusterConfig,
    distances: &mut Array2<F>,
) where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];

    // Compute upper triangle in parallel by row
    let row_results: Vec<Vec<(usize, F)>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut row_dists = Vec::with_capacity(n_samples - i - 1);
            for j in (i + 1)..n_samples {
                let dist = simd_distance(data.row(i), data.row(j), metric, Some(config))
                    .unwrap_or_else(|_| F::zero());
                row_dists.push((j, dist));
            }
            row_dists
        })
        .collect();

    // Fill symmetric matrix from parallel results
    for (i, row_dists) in row_results.into_iter().enumerate() {
        for (j, dist) in row_dists {
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }
}

/// Compute condensed pairwise distances (upper triangular, no diagonal).
///
/// This is the memory-efficient form: stores n*(n-1)/2 distances instead of n*n.
/// The index mapping is: `condensed_idx = n * i + j - (i + 1) * (i + 2) / 2`
/// for row i < j.
///
/// # Arguments
///
/// * `data` - Input data matrix (n_samples x n_features)
/// * `metric` - Distance metric
/// * `config` - Optional SIMD configuration
///
/// # Returns
///
/// Condensed distance vector of length n*(n-1)/2.
///
/// # Errors
///
/// Returns error if data is empty.
pub fn simd_pairwise_condensed_distances<F>(
    data: ArrayView2<F>,
    metric: SimdDistanceMetric,
    config: Option<&SimdClusterConfig>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + Send + Sync + SimdUnifiedOps,
{
    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data must have at least one sample".to_string(),
        ));
    }

    let n_distances = n_samples * (n_samples - 1) / 2;
    let mut distances = Array1::zeros(n_distances);
    let default_config = SimdClusterConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = simd_distance(data.row(i), data.row(j), metric, Some(cfg))?;
            distances[idx] = dist;
            idx += 1;
        }
    }

    Ok(distances)
}
