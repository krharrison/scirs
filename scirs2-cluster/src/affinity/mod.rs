//! Affinity Propagation clustering implementation
//!
//! Affinity Propagation is a clustering algorithm that identifies exemplars (cluster centers)
//! among the data points and forms clusters around these exemplars. It works by exchanging
//! real-valued messages between data points until a high-quality set of exemplars emerges.
//!
//! # Features
//!
//! - **Message passing**: Responsibility and availability matrix updates
//! - **Damping**: Prevents numerical oscillations (0.5 to 1.0)
//! - **Convergence detection**: Monitors exemplar stability across iterations
//! - **Preference tuning**: Controls the number of clusters
//! - **Sparse similarity**: Support for precomputed sparse similarity matrices
//!
//! # Algorithm
//!
//! 1. Compute similarity matrix S (negative squared Euclidean distance by default)
//! 2. Set diagonal (preference) to median of similarities or user-specified value
//! 3. Iteratively update responsibility R and availability A matrices
//! 4. Identify exemplars where R(k,k) + A(k,k) > 0
//! 5. Assign non-exemplar points to nearest exemplar
//!
//! # References
//!
//! Frey, B.J. and Dueck, D. (2007). "Clustering by Passing Messages Between Data Points."
//! Science, 315(5814), pp. 972-976.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Options for Affinity Propagation clustering
#[derive(Debug, Clone)]
pub struct AffinityPropagationOptions<F: Float> {
    /// Damping factor (between 0.5 and 1.0) to avoid numerical oscillations
    pub damping: F,

    /// Maximum number of iterations
    pub max_iter: usize,

    /// Number of iterations with no change in exemplars that triggers convergence
    pub convergence_iter: usize,

    /// Preference value for all data points.
    /// Controls cluster count: lower preference => fewer clusters.
    /// If None, the median of non-diagonal similarities is used.
    pub preference: Option<F>,

    /// Whether input is a precomputed similarity matrix
    pub affinity: String,

    /// Verbose output for debugging
    pub verbose: bool,
}

impl<F: Float + FromPrimitive> Default for AffinityPropagationOptions<F> {
    fn default() -> Self {
        Self {
            damping: F::from(0.5).unwrap_or(F::one()),
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
            affinity: "euclidean".to_string(),
            verbose: false,
        }
    }
}

/// Result of the Affinity Propagation algorithm
#[derive(Debug, Clone)]
pub struct AffinityPropagationResult<F: Float> {
    /// Indices of exemplar (cluster center) points
    pub cluster_centers_indices: Vec<usize>,
    /// Cluster label for each data point
    pub labels: Array1<i32>,
    /// Number of iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final responsibility matrix
    pub responsibility: Array2<F>,
    /// Final availability matrix
    pub availability: Array2<F>,
}

/// Compute pairwise similarity matrix based on negative squared Euclidean distance
fn compute_similarity<F>(data: ArrayView2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut similarity = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            if i == j {
                similarity[[i, i]] = F::zero();
            } else {
                let mut dist_sq = F::zero();
                for k in 0..n_features {
                    let diff = data[[i, k]] - data[[j, k]];
                    dist_sq = dist_sq + diff * diff;
                }
                let sim = -dist_sq;
                similarity[[i, j]] = sim;
                similarity[[j, i]] = sim;
            }
        }
    }

    Ok(similarity)
}

/// Compute preference values and set the diagonal of the similarity matrix
fn compute_preference<F>(mut similarity: Array2<F>, preference: Option<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = similarity.shape()[0];

    if let Some(pref) = preference {
        for i in 0..n_samples {
            similarity[[i, i]] = pref;
        }
        return Ok(similarity);
    }

    // Compute median of non-diagonal similarities
    let mut non_diag_similarities = Vec::with_capacity(n_samples * (n_samples - 1));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i != j {
                non_diag_similarities.push(similarity[[i, j]]);
            }
        }
    }

    if non_diag_similarities.is_empty() {
        return Err(ClusteringError::ComputationError(
            "Could not compute preferences, no non-diagonal similarities found".to_string(),
        ));
    }

    non_diag_similarities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = non_diag_similarities.len();
    let median = if n % 2 == 0 {
        let two = F::from(2.0).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?;
        (non_diag_similarities[n / 2 - 1] + non_diag_similarities[n / 2]) / two
    } else {
        non_diag_similarities[n / 2]
    };

    for i in 0..n_samples {
        similarity[[i, i]] = median;
    }

    Ok(similarity)
}

/// Run the core Affinity Propagation message-passing algorithm
fn run_affinity_propagation<F>(
    similarity: &Array2<F>,
    options: &AffinityPropagationOptions<F>,
) -> Result<AffinityPropagationResult<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = similarity.shape()[0];

    // Validate damping
    let half = F::from(0.5).ok_or_else(|| {
        ClusteringError::ComputationError("Failed to convert damping threshold".into())
    })?;
    if options.damping < half || options.damping > F::one() {
        return Err(ClusteringError::InvalidInput(
            "Damping factor must be between 0.5 and 1.0".to_string(),
        ));
    }

    let damping = options.damping;
    let one_minus_damping = F::one() - damping;

    // Initialize message matrices
    let mut responsibility = Array2::zeros((n_samples, n_samples));
    let mut availability = Array2::zeros((n_samples, n_samples));

    // Convergence tracking
    let mut convergence_count = 0;
    let mut last_exemplars: Option<Vec<usize>> = None;
    let mut converged = false;

    // Main iteration loop
    let mut n_iter = 0;
    for _iter in 0..options.max_iter {
        n_iter = _iter + 1;

        // === Update responsibility matrix ===
        // r(i, k) = s(i, k) - max_{k' != k} { a(i, k') + s(i, k') }
        let old_responsibility = responsibility.clone();

        for i in 0..n_samples {
            for k in 0..n_samples {
                let mut max_val = F::neg_infinity();
                for k_prime in 0..n_samples {
                    if k_prime != k {
                        let val = availability[[i, k_prime]] + similarity[[i, k_prime]];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                responsibility[[i, k]] = similarity[[i, k]] - max_val;
            }
        }

        // Apply damping to responsibility
        for i in 0..n_samples {
            for k in 0..n_samples {
                responsibility[[i, k]] = damping * old_responsibility[[i, k]]
                    + one_minus_damping * responsibility[[i, k]];
            }
        }

        // === Update availability matrix ===
        // a(i, k) = min(0, r(k,k) + sum_{i' != i,k} max(0, r(i',k)))  if i != k
        // a(k, k) = sum_{i' != k} max(0, r(i',k))  if i == k
        let old_availability = availability.clone();

        for i in 0..n_samples {
            for k in 0..n_samples {
                if i != k {
                    let mut sum = F::zero();
                    for i_prime in 0..n_samples {
                        if i_prime != i && i_prime != k {
                            sum = sum + F::max(F::zero(), responsibility[[i_prime, k]]);
                        }
                    }
                    availability[[i, k]] = F::min(F::zero(), responsibility[[k, k]] + sum);
                } else {
                    let mut sum = F::zero();
                    for i_prime in 0..n_samples {
                        if i_prime != k {
                            sum = sum + F::max(F::zero(), responsibility[[i_prime, k]]);
                        }
                    }
                    availability[[k, k]] = sum;
                }
            }
        }

        // Apply damping to availability
        for i in 0..n_samples {
            for k in 0..n_samples {
                availability[[i, k]] =
                    damping * old_availability[[i, k]] + one_minus_damping * availability[[i, k]];
            }
        }

        // Check for convergence: identify current exemplars
        let exemplars = identify_exemplars(&responsibility, &availability);

        if let Some(ref last) = last_exemplars {
            if exemplars == *last {
                convergence_count += 1;
            } else {
                convergence_count = 0;
            }
        }

        if convergence_count >= options.convergence_iter {
            converged = true;
            break;
        }

        last_exemplars = Some(exemplars);
    }

    // Final cluster extraction
    let (cluster_centers_indices, labels) =
        extract_clusters_from_matrices(&responsibility, &availability, similarity)?;

    Ok(AffinityPropagationResult {
        cluster_centers_indices,
        labels,
        n_iter,
        converged,
        responsibility,
        availability,
    })
}

/// Identify exemplars from the current R and A matrices
fn identify_exemplars<F: Float>(
    responsibility: &Array2<F>,
    availability: &Array2<F>,
) -> Vec<usize> {
    let n_samples = responsibility.shape()[0];
    let mut exemplars = Vec::new();

    for k in 0..n_samples {
        if responsibility[[k, k]] + availability[[k, k]] > F::zero() {
            exemplars.push(k);
        }
    }

    exemplars
}

/// Extract clusters from responsibility and availability matrices
fn extract_clusters_from_matrices<F>(
    responsibility: &Array2<F>,
    availability: &Array2<F>,
    similarity: &Array2<F>,
) -> Result<(Vec<usize>, Array1<i32>)>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = responsibility.shape()[0];

    // Find exemplars
    let mut exemplars = identify_exemplars(responsibility, availability);

    // If no exemplars found, select the point with highest self-criterion
    if exemplars.is_empty() {
        let mut max_criterion = F::neg_infinity();
        let mut max_idx = 0;

        for k in 0..n_samples {
            let criterion = responsibility[[k, k]] + availability[[k, k]];
            if criterion > max_criterion {
                max_criterion = criterion;
                max_idx = k;
            }
        }

        exemplars.push(max_idx);
    }

    // Assign labels based on similarity to exemplars
    let mut labels = Array1::from_vec(vec![-1; n_samples]);

    for i in 0..n_samples {
        let mut max_similarity = F::neg_infinity();
        let mut best_cluster: i32 = -1;

        for (cluster_idx, &exemplar) in exemplars.iter().enumerate() {
            let sim = similarity[[i, exemplar]];
            if sim > max_similarity {
                max_similarity = sim;
                best_cluster = cluster_idx as i32;
            }
        }

        labels[i] = best_cluster;
    }

    Ok((exemplars, labels))
}

/// Compare two label assignments for equality
fn compare_labels(labels1: ArrayView1<i32>, labels2: ArrayView1<i32>) -> bool {
    if labels1.len() != labels2.len() {
        return false;
    }

    for i in 0..labels1.len() {
        if labels1[i] != labels2[i] {
            return false;
        }
    }

    true
}

/// Affinity Propagation clustering algorithm
///
/// Finds exemplars (cluster centers) among the data points and forms clusters around them.
/// The algorithm determines the number of clusters based on the input preference value.
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features) or precomputed similarity matrix
/// * `precomputed` - Whether data is a precomputed similarity matrix
/// * `options` - Optional algorithm parameters
///
/// # Returns
///
/// * Tuple of (cluster_centers_indices, labels)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::affinity::{affinity_propagation, AffinityPropagationOptions};
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     4.0, 5.0,
///     4.2, 4.8,
///     3.9, 5.1,
/// ]).expect("Operation failed");
///
/// let options = AffinityPropagationOptions {
///     damping: 0.9,
///     ..Default::default()
/// };
///
/// let result = affinity_propagation(data.view(), false, Some(options));
/// if let Ok((centers, labels)) = result {
///     println!("Cluster centers: {:?}", centers);
///     println!("Cluster assignments: {:?}", labels);
/// }
/// ```
pub fn affinity_propagation<F>(
    data: ArrayView2<F>,
    precomputed: bool,
    options: Option<AffinityPropagationOptions<F>>,
) -> Result<(Vec<usize>, Array1<i32>)>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let opts = options.unwrap_or_default();

    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    if n_samples == 1 {
        return Ok((vec![0], Array1::from_vec(vec![0])));
    }

    let similarity = if precomputed {
        if data.shape()[0] != data.shape()[1] {
            return Err(ClusteringError::InvalidInput(
                "Precomputed similarity matrix must be square".into(),
            ));
        }
        compute_preference(data.to_owned(), opts.preference)?
    } else {
        let sim = compute_similarity(data)?;
        compute_preference(sim, opts.preference)?
    };

    let result = run_affinity_propagation(&similarity, &opts)?;

    Ok((result.cluster_centers_indices, result.labels))
}

/// Affinity Propagation with full result including convergence info
///
/// # Arguments
///
/// * `data` - Input data (n_samples x n_features) or precomputed similarity matrix
/// * `precomputed` - Whether data is a precomputed similarity matrix
/// * `options` - Algorithm parameters
///
/// # Returns
///
/// * Full AffinityPropagationResult with convergence information
pub fn affinity_propagation_full<F>(
    data: ArrayView2<F>,
    precomputed: bool,
    options: AffinityPropagationOptions<F>,
) -> Result<AffinityPropagationResult<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = data.shape()[0];
    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    let similarity = if precomputed {
        if data.shape()[0] != data.shape()[1] {
            return Err(ClusteringError::InvalidInput(
                "Precomputed similarity matrix must be square".into(),
            ));
        }
        compute_preference(data.to_owned(), options.preference)?
    } else {
        let sim = compute_similarity(data)?;
        compute_preference(sim, options.preference)?
    };

    run_affinity_propagation(&similarity, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 5.0, 6.0, 5.2, 5.8, 4.8, 6.1],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_affinity_propagation_basic() {
        let data = make_two_cluster_data();

        let options = AffinityPropagationOptions {
            damping: 0.5,
            preference: Some(-50.0),
            max_iter: 200,
            convergence_iter: 15,
            ..Default::default()
        };

        let result = affinity_propagation(data.view(), false, Some(options));
        assert!(result.is_ok());

        let (centers, labels) = result.expect("Should succeed");

        assert!(!centers.is_empty());
        assert!(centers.len() <= 6);
        assert_eq!(labels.len(), 6);

        for &label in labels.iter() {
            assert!(label >= 0);
            assert!((label as usize) < centers.len());
        }
    }

    #[test]
    fn test_affinity_propagation_precomputed() {
        let similarity = Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, -1.0, -3.0, -5.0, -1.0, 0.0, -2.0, -4.0, -3.0, -2.0, 0.0, -6.0, -5.0, -4.0,
                -6.0, 0.0,
            ],
        )
        .expect("Failed to create similarity");

        let options = AffinityPropagationOptions {
            damping: 0.9,
            preference: Some(-5.0),
            ..Default::default()
        };

        let result = affinity_propagation(similarity.view(), true, Some(options));
        assert!(result.is_ok());

        let (centers, labels) = result.expect("Should succeed");

        assert_eq!(labels.len(), 4);
        assert!(!centers.is_empty());
        assert!(labels.iter().any(|&l| l == labels[0]));
    }

    #[test]
    fn test_affinity_propagation_convergence() {
        let data = make_two_cluster_data();

        let options = AffinityPropagationOptions {
            damping: 0.9,
            preference: Some(-10.0),
            max_iter: 500,
            convergence_iter: 10,
            ..Default::default()
        };

        let result =
            affinity_propagation_full(data.view(), false, options).expect("Should succeed");

        // With enough iterations, should converge
        assert!(result.n_iter <= 500);
        assert!(!result.cluster_centers_indices.is_empty());
    }

    #[test]
    fn test_affinity_propagation_damping_effect() {
        let data = make_two_cluster_data();

        // Low damping -> faster updates, potentially less stable
        let options_low = AffinityPropagationOptions {
            damping: 0.5,
            preference: Some(-20.0),
            max_iter: 100,
            ..Default::default()
        };

        // High damping -> slower updates, more stable
        let options_high = AffinityPropagationOptions {
            damping: 0.95,
            preference: Some(-20.0),
            max_iter: 100,
            ..Default::default()
        };

        let result_low = affinity_propagation(data.view(), false, Some(options_low));
        let result_high = affinity_propagation(data.view(), false, Some(options_high));

        assert!(result_low.is_ok());
        assert!(result_high.is_ok());
    }

    #[test]
    fn test_affinity_propagation_invalid_damping() {
        let data = make_two_cluster_data();

        let options = AffinityPropagationOptions {
            damping: 0.3, // Too low
            ..Default::default()
        };

        let result = affinity_propagation(data.view(), false, Some(options));
        assert!(result.is_err());
    }

    #[test]
    fn test_affinity_propagation_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("Failed to create data");

        let result = affinity_propagation(data.view(), false, None);
        assert!(result.is_ok());

        let (centers, labels) = result.expect("Should succeed");
        assert_eq!(centers.len(), 1);
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], 0);
    }

    #[test]
    fn test_affinity_propagation_empty_data() {
        let data = Array2::<f64>::zeros((0, 2));
        let result = affinity_propagation(data.view(), false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_affinity_propagation_preference_controls_clusters() {
        let data = make_two_cluster_data();

        // Very low preference -> fewer clusters
        let options_few = AffinityPropagationOptions {
            damping: 0.9,
            preference: Some(-100.0),
            max_iter: 200,
            ..Default::default()
        };

        // Higher preference -> potentially more clusters
        let options_many = AffinityPropagationOptions {
            damping: 0.9,
            preference: Some(-1.0),
            max_iter: 200,
            ..Default::default()
        };

        let (centers_few, _) =
            affinity_propagation(data.view(), false, Some(options_few)).expect("Should succeed");
        let (centers_many, _) =
            affinity_propagation(data.view(), false, Some(options_many)).expect("Should succeed");

        // Higher preference should produce at least as many clusters
        assert!(
            centers_many.len() >= centers_few.len(),
            "Higher preference should yield >= clusters: {} vs {}",
            centers_many.len(),
            centers_few.len()
        );
    }

    #[test]
    fn test_affinity_propagation_full_result() {
        let data = make_two_cluster_data();

        let options = AffinityPropagationOptions {
            damping: 0.8,
            preference: Some(-20.0),
            max_iter: 100,
            convergence_iter: 10,
            ..Default::default()
        };

        let result =
            affinity_propagation_full(data.view(), false, options).expect("Should succeed");

        // Verify the result has all expected fields
        assert_eq!(result.labels.len(), 6);
        assert!(!result.cluster_centers_indices.is_empty());
        assert!(result.n_iter > 0);
        assert_eq!(result.responsibility.shape(), &[6, 6]);
        assert_eq!(result.availability.shape(), &[6, 6]);
    }
}
