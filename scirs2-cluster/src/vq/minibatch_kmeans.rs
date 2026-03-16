//! Mini-Batch K-means clustering implementation
//!
//! This module provides an implementation of the Mini-Batch K-means algorithm
//! (Sculley 2010), a variant of k-means that uses mini-batches to reduce
//! computation time while still attempting to optimize the same objective function.
//!
//! # Advantages over standard K-means
//!
//! - **Much faster** for large datasets (sublinear per-iteration cost)
//! - **Similar quality** to standard k-means in practice
//! - **Streaming compatible**: can process data in chunks
//!
//! # Algorithm
//!
//! 1. Initialize centroids (k-means++ or random)
//! 2. For each iteration:
//!    a. Sample a mini-batch of size `batch_size` from the data
//!    b. Assign each sample in the batch to its nearest centroid
//!    c. Update centroids using per-center learning rate: eta = 1 / count(center)
//! 3. Monitor convergence using exponentially weighted average (EWA) of inertia
//! 4. Detect and reassign near-empty clusters
//!
//! # References
//!
//! Sculley, D. (2010). "Web-Scale K-Means Clustering." WWW, pp. 1177-1178.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::{Rng, RngExt, SeedableRng};
use std::fmt::Debug;

use super::{euclidean_distance, kmeans_plus_plus};
use crate::error::{ClusteringError, Result};

/// Initialization method for Mini-Batch K-means
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniBatchInit {
    /// K-means++ initialization (default, recommended)
    KMeansPlusPlus,
    /// Random sampling from data points
    Random,
}

impl Default for MiniBatchInit {
    fn default() -> Self {
        MiniBatchInit::KMeansPlusPlus
    }
}

/// Options for Mini-Batch K-means clustering
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansOptions<F: Float> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Size of mini-batches
    pub batch_size: usize,
    /// Convergence threshold for centroid movement
    pub tol: F,
    /// Random seed for initialization and batch sampling
    pub random_seed: Option<u64>,
    /// Number of iterations without improvement before stopping
    pub max_no_improvement: usize,
    /// Number of samples to use for initialization
    pub init_size: Option<usize>,
    /// Ratio of samples that should be reassigned to prevent empty clusters
    pub reassignment_ratio: F,
    /// Initialization method
    pub init: MiniBatchInit,
    /// EWA smoothing factor for inertia tracking (0 to 1)
    pub ewa_smoothing: F,
}

impl<F: Float + FromPrimitive> Default for MiniBatchKMeansOptions<F> {
    fn default() -> Self {
        Self {
            max_iter: 100,
            batch_size: 1024,
            tol: F::from(1e-4).unwrap_or(F::epsilon()),
            random_seed: None,
            max_no_improvement: 10,
            init_size: None,
            reassignment_ratio: F::from(0.01).unwrap_or(F::epsilon()),
            init: MiniBatchInit::KMeansPlusPlus,
            ewa_smoothing: F::from(0.7).unwrap_or(F::one()),
        }
    }
}

/// Result of Mini-Batch K-means with convergence diagnostics
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansResult<F: Float> {
    /// Final cluster centroids (k x n_features)
    pub centroids: Array2<F>,
    /// Cluster assignments for each data point
    pub labels: Array1<usize>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Final inertia (sum of squared distances to nearest centroid)
    pub inertia: F,
    /// Whether the algorithm converged
    pub converged: bool,
    /// History of EWA inertia values per iteration
    pub inertia_history: Vec<F>,
    /// Per-cluster count of assigned samples
    pub cluster_counts: Array1<usize>,
    /// Number of reassignments performed during training
    pub n_reassignments: usize,
}

/// Mini-Batch K-means clustering algorithm
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features)
/// * `k` - Number of clusters
/// * `options` - Optional parameters
///
/// # Returns
///
/// * Tuple of (centroids, labels)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::vq::minibatch_kmeans;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     3.7, 4.2,
///     3.9, 3.9,
///     4.2, 4.1,
/// ]).expect("Operation failed");
///
/// let (centroids, labels) = minibatch_kmeans(ArrayView2::from(&data), 2, None)
///     .expect("Operation failed");
/// ```
pub fn minibatch_kmeans<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<MiniBatchKMeansOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    let result = minibatch_kmeans_full(data, k, options)?;
    Ok((result.centroids, result.labels))
}

/// Mini-Batch K-means with full diagnostic output
pub fn minibatch_kmeans_full<F>(
    data: ArrayView2<F>,
    k: usize,
    options: Option<MiniBatchKMeansOptions<F>>,
) -> Result<MiniBatchKMeansResult<F>>
where
    F: Float + FromPrimitive + Debug + std::iter::Sum,
{
    // Input validation
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters must be greater than 0".to_string(),
        ));
    }

    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput(
            "Input data is empty".to_string(),
        ));
    }

    if k > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters ({}) cannot be greater than number of data points ({})",
            k, n_samples
        )));
    }

    let opts = options.unwrap_or_default();

    // Setup RNG
    let mut rng = match opts.random_seed {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => {
            scirs2_core::random::rngs::StdRng::seed_from_u64(scirs2_core::random::rng().random())
        }
    };

    // Determine initialization size
    let init_size = opts
        .init_size
        .unwrap_or_else(|| {
            let default_size = 3 * opts.batch_size;
            if default_size < 3 * k {
                3 * k
            } else {
                default_size
            }
        })
        .min(n_samples);

    // Initialize centroids
    let centroids = match opts.init {
        MiniBatchInit::KMeansPlusPlus => {
            if init_size < n_samples {
                let mut indices = Vec::with_capacity(init_size);
                for _ in 0..init_size {
                    indices.push(rng.random_range(0..n_samples));
                }
                let init_data =
                    Array2::from_shape_fn((init_size, n_features), |(i, j)| data[[indices[i], j]]);
                kmeans_plus_plus(init_data.view(), k, opts.random_seed)?
            } else {
                kmeans_plus_plus(data, k, opts.random_seed)?
            }
        }
        MiniBatchInit::Random => {
            let mut centers = Array2::zeros((k, n_features));
            for i in 0..k {
                let idx = rng.random_range(0..n_samples);
                centers.row_mut(i).assign(&data.row(idx));
            }
            centers
        }
    };

    // Initialize variables
    let mut centroids = centroids;
    let mut counts = Array1::<F>::from_elem(k, F::one());

    // Convergence tracking
    let mut ewa_inertia: Option<F> = None;
    let mut no_improvement_count = 0;
    let mut best_inertia = F::infinity();
    let mut prev_centers: Option<Array2<F>> = None;
    let mut inertia_history = Vec::with_capacity(opts.max_iter);
    let mut total_reassignments = 0;
    let mut converged = false;
    let mut n_iter = 0;

    // Mini-batch optimization loop
    for iter in 0..opts.max_iter {
        n_iter = iter + 1;

        // Sample a mini-batch
        let batch_size = opts.batch_size.min(n_samples);
        let mut batch_indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            batch_indices.push(rng.random_range(0..n_samples));
        }

        // Perform mini-batch step
        let step_result =
            mini_batch_step(&data, &batch_indices, &mut centroids, &mut counts, &opts)?;

        total_reassignments += step_result.n_reassignments;

        // Update EWA of inertia
        let current_ewa = match ewa_inertia {
            Some(prev_ewa) => {
                prev_ewa * opts.ewa_smoothing
                    + step_result.batch_inertia * (F::one() - opts.ewa_smoothing)
            }
            None => step_result.batch_inertia,
        };
        ewa_inertia = Some(current_ewa);
        inertia_history.push(current_ewa);

        // Check inertia improvement
        if current_ewa < best_inertia {
            best_inertia = current_ewa;
            no_improvement_count = 0;
        } else {
            no_improvement_count += 1;
        }

        // Check centroid movement convergence
        if let Some(ref prev) = prev_centers {
            let mut center_shift = F::zero();
            for i in 0..k {
                let dist = euclidean_distance(centroids.slice(s![i, ..]), prev.slice(s![i, ..]));
                center_shift = center_shift + dist;
            }
            let k_f = F::from(k).unwrap_or(F::one());
            center_shift = center_shift / k_f;

            if center_shift < opts.tol {
                converged = true;
                break;
            }
        }

        prev_centers = Some(centroids.clone());

        // Early stopping
        if no_improvement_count >= opts.max_no_improvement {
            converged = true;
            break;
        }
    }

    // Final label assignment
    let (final_labels, final_distances) = assign_labels(data, centroids.view())?;

    // Compute final inertia
    let final_inertia = final_distances
        .iter()
        .fold(F::zero(), |acc, &d| acc + d * d);

    // Compute per-cluster counts
    let mut cluster_counts = Array1::<usize>::zeros(k);
    for &label in final_labels.iter() {
        if label < k {
            cluster_counts[label] += 1;
        }
    }

    Ok(MiniBatchKMeansResult {
        centroids,
        labels: final_labels,
        n_iter,
        inertia: final_inertia,
        converged,
        inertia_history,
        cluster_counts,
        n_reassignments: total_reassignments,
    })
}

/// Result of a single mini-batch step
struct MiniBatchStepResult<F: Float> {
    /// Inertia of the mini-batch (average squared distance)
    batch_inertia: F,
    /// Number of reassignments performed
    n_reassignments: usize,
}

/// Performs a single Mini-Batch K-means step
fn mini_batch_step<F>(
    data: &ArrayView2<F>,
    batch_indices: &[usize],
    centroids: &mut Array2<F>,
    counts: &mut Array1<F>,
    opts: &MiniBatchKMeansOptions<F>,
) -> Result<MiniBatchStepResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let k = centroids.shape()[0];
    let n_features = centroids.shape()[1];
    let batch_size = batch_indices.len();

    let mut closest_distances = Array1::from_elem(batch_size, F::infinity());
    let mut closest_centers = Array1::<usize>::zeros(batch_size);
    let mut inertia = F::zero();

    // Assignment: find nearest centroid for each batch sample
    for (i, &sample_idx) in batch_indices.iter().enumerate() {
        let sample = data.slice(s![sample_idx, ..]);

        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for j in 0..k {
            let dist = euclidean_distance(sample, centroids.slice(s![j, ..]));
            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }

        closest_centers[i] = min_idx;
        closest_distances[i] = min_dist;
        inertia = inertia + min_dist * min_dist;
    }

    // Update centroids using per-center learning rate
    for i in 0..batch_size {
        let center_idx = closest_centers[i];
        let sample_idx = batch_indices[i];
        let sample = data.slice(s![sample_idx, ..]);

        let count = counts[center_idx];
        // Learning rate decreases as count increases: eta = 1 / (count + 1)
        let learning_rate = F::one() / (count + F::one());

        for j in 0..n_features {
            centroids[[center_idx, j]] =
                centroids[[center_idx, j]] * (F::one() - learning_rate) + sample[j] * learning_rate;
        }

        counts[center_idx] = count + F::one();
    }

    // Handle near-empty clusters via reassignment
    let mut n_reassignments = 0;
    let max_count = counts.fold(F::zero(), |a, &b| a.max(b));
    let reassign_threshold = max_count * opts.reassignment_ratio;

    for c in 0..k {
        if counts[c] < reassign_threshold {
            // Find the batch point furthest from its assigned centroid
            let mut max_dist = F::zero();
            let mut max_idx = 0;

            for j in 0..batch_size {
                if closest_distances[j] > max_dist {
                    max_dist = closest_distances[j];
                    max_idx = j;
                }
            }

            if max_dist > F::zero() {
                let sample_idx = batch_indices[max_idx];
                let sample = data.slice(s![sample_idx, ..]);

                for j in 0..n_features {
                    centroids[[c, j]] = sample[j];
                }

                counts[c] = counts[c].max(F::one());
                closest_centers[max_idx] = c;
                closest_distances[max_idx] = F::zero();
                n_reassignments += 1;
            }
        }
    }

    // Normalize inertia by batch size
    let batch_f = F::from(batch_size).unwrap_or(F::one());
    inertia = inertia / batch_f;

    Ok(MiniBatchStepResult {
        batch_inertia: inertia,
        n_reassignments,
    })
}

/// Assigns each sample in the dataset to its closest centroid
fn assign_labels<F>(
    data: ArrayView2<F>,
    centroids: ArrayView2<F>,
) -> Result<(Array1<usize>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_clusters = centroids.shape()[0];

    let mut labels = Array1::<usize>::zeros(n_samples);
    let mut distances = Array1::<F>::zeros(n_samples);

    for i in 0..n_samples {
        let sample = data.slice(s![i, ..]);
        let mut min_dist = F::infinity();
        let mut min_idx = 0;

        for j in 0..n_clusters {
            let centroid = centroids.slice(s![j, ..]);
            let dist = euclidean_distance(sample, centroid);

            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }

        labels[i] = min_idx;
        distances[i] = min_dist;
    }

    Ok((labels, distances))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 4.0, 5.0, 4.2, 4.8, 3.9, 5.1],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_minibatch_kmeans_simple() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            max_iter: 10,
            batch_size: 3,
            random_seed: Some(42),
            ..Default::default()
        };

        let (centroids, labels) =
            minibatch_kmeans(data.view(), 2, Some(options)).expect("Should succeed");

        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.shape(), &[6]);

        let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique_labels.len(), 2);

        // First 3 points should share a label, last 3 should share another
        let first_label = labels[0];
        assert_eq!(labels[1], first_label);
        assert_eq!(labels[2], first_label);

        let second_label = labels[3];
        assert_eq!(labels[4], second_label);
        assert_eq!(labels[5], second_label);
    }

    #[test]
    fn test_minibatch_kmeans_full_diagnostics() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            max_iter: 50,
            batch_size: 4,
            random_seed: Some(42),
            ..Default::default()
        };

        let result = minibatch_kmeans_full(data.view(), 2, Some(options)).expect("Should succeed");

        assert_eq!(result.centroids.shape(), &[2, 2]);
        assert_eq!(result.labels.shape(), &[6]);
        assert!(result.n_iter > 0);
        assert!(result.inertia >= 0.0);
        assert!(!result.inertia_history.is_empty());

        // Every cluster should have at least one point
        for &count in result.cluster_counts.iter() {
            assert!(count > 0, "Each cluster should have assigned points");
        }
    }

    #[test]
    fn test_minibatch_kmeans_convergence() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            max_iter: 1000,
            batch_size: 6, // Full batch
            random_seed: Some(42),
            tol: 1e-6,
            max_no_improvement: 20,
            ..Default::default()
        };

        let result = minibatch_kmeans_full(data.view(), 2, Some(options)).expect("Should succeed");

        // Should converge before max_iter
        assert!(
            result.n_iter < 1000,
            "Should converge before max_iter, took {} iters",
            result.n_iter
        );
    }

    #[test]
    fn test_minibatch_kmeans_empty_clusters() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.2, 1.0, 1.0, 1.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .expect("Failed to create data");

        let options = MiniBatchKMeansOptions {
            max_iter: 20,
            batch_size: 4,
            random_seed: Some(42),
            reassignment_ratio: 0.1,
            ..Default::default()
        };

        let (centroids, labels) =
            minibatch_kmeans(data.view(), 3, Some(options)).expect("Should succeed");

        assert_eq!(centroids.shape(), &[3, 2]);
        assert_eq!(labels.shape(), &[8]);

        let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert!(unique_labels.len() <= 3);
    }

    #[test]
    fn test_minibatch_kmeans_random_init() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            init: MiniBatchInit::Random,
            random_seed: Some(42),
            max_iter: 50,
            batch_size: 4,
            ..Default::default()
        };

        let (centroids, labels) =
            minibatch_kmeans(data.view(), 2, Some(options)).expect("Should succeed");

        assert_eq!(centroids.shape(), &[2, 2]);
        assert_eq!(labels.shape(), &[6]);
    }

    #[test]
    fn test_minibatch_kmeans_inertia_decreases() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            max_iter: 50,
            batch_size: 6,
            random_seed: Some(42),
            ewa_smoothing: 0.5,
            ..Default::default()
        };

        let result = minibatch_kmeans_full(data.view(), 2, Some(options)).expect("Should succeed");

        // Overall trend of inertia should be decreasing
        if result.inertia_history.len() >= 3 {
            let first_few: f64 = result.inertia_history[..3].iter().copied().sum::<f64>() / 3.0;
            let last_few: f64 = result.inertia_history[result.inertia_history.len() - 3..]
                .iter()
                .copied()
                .sum::<f64>()
                / 3.0;

            assert!(
                last_few <= first_few + 1.0,
                "Inertia should generally decrease: first_avg={}, last_avg={}",
                first_few,
                last_few
            );
        }
    }

    #[test]
    fn test_minibatch_kmeans_invalid_inputs() {
        let data = make_two_cluster_data();

        // k = 0
        let result = minibatch_kmeans(data.view(), 0, None);
        assert!(result.is_err());

        // k > n_samples
        let result = minibatch_kmeans(data.view(), 100, None);
        assert!(result.is_err());

        // Empty data
        let empty = Array2::<f64>::zeros((0, 2));
        let result = minibatch_kmeans(empty.view(), 2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_minibatch_kmeans_k_equals_n() {
        let data = make_two_cluster_data();

        let options = MiniBatchKMeansOptions {
            random_seed: Some(42),
            max_iter: 10,
            ..Default::default()
        };

        let (centroids, labels) =
            minibatch_kmeans(data.view(), 6, Some(options)).expect("Should succeed");

        assert_eq!(centroids.shape(), &[6, 2]);
        assert_eq!(labels.shape(), &[6]);

        // Each point should be in its own cluster
        let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();
        assert_eq!(unique_labels.len(), 6);
    }
}
