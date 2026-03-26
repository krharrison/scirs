//! Spatial statistics module for analyzing spatial patterns and relationships
//!
//! This module provides statistical measures commonly used in spatial analysis,
//! including measures of spatial autocorrelation, clustering, and pattern analysis.
//!
//! # Features
//!
//! * **Spatial Autocorrelation**: Moran's I, Geary's C
//! * **Local Indicators**: Local Moran's I (LISA)
//! * **Distance-based Statistics**: Getis-Ord statistics
//! * **Pattern Analysis**: Nearest neighbor analysis
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_spatial::spatial_stats::{morans_i, gearys_c};
//!
//! // Create spatial data (values at different locations)
//! let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
//!
//! // Define spatial weights matrix (adjacency-based)
//! let weights = array![
//!     [0.0, 1.0, 0.0, 0.0, 1.0],
//!     [1.0, 0.0, 1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0, 0.0, 1.0],
//!     [1.0, 0.0, 0.0, 1.0, 0.0],
//! ];
//!
//! // Calculate spatial autocorrelation
//! let moran = morans_i(&values.view(), &weights.view()).expect("Operation failed");
//! let geary = gearys_c(&values.view(), &weights.view()).expect("Operation failed");
//!
//! println!("Moran's I: {:.3}", moran);
//! println!("Geary's C: {:.3}", geary);
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;

use crate::error::{SpatialError, SpatialResult};

/// Calculate Moran's I statistic for spatial autocorrelation
///
/// Moran's I measures the degree of spatial autocorrelation in a dataset.
/// Values range from -1 (perfect negative autocorrelation) to +1 (perfect positive autocorrelation).
/// A value of 0 indicates no spatial autocorrelation (random spatial pattern).
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix (typically binary adjacency or distance-based)
///
/// # Returns
///
/// * Moran's I statistic
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::morans_i;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let moran = morans_i(&values.view(), &weights.view()).expect("Operation failed");
/// println!("Moran's I: {:.3}", moran);
/// ```
#[allow(dead_code)]
pub fn morans_i<T: Float>(values: &ArrayView1<T>, weights: &ArrayView2<T>) -> SpatialResult<T> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate mean
    let mean = values.sum() / T::from(n).expect("Operation failed");

    // Calculate deviations from mean
    let deviations: Array1<T> = values.map(|&x| x - mean);

    // Calculate sum of weights
    let w_sum = weights.sum();

    if w_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Sum of weights cannot be zero".to_string(),
        ));
    }

    // Calculate numerator: sum of (w_ij * (x_i - mean) * (x_j - mean))
    let mut numerator = T::zero();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                numerator = numerator + weights[[i, j]] * deviations[i] * deviations[j];
            }
        }
    }

    // Calculate denominator: sum of (x_i - mean)^2
    let denominator: T = deviations.map(|&x| x * x).sum();

    if denominator.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    // Moran's I = (n / W) * (numerator / denominator)
    let morans_i = (T::from(n).expect("Operation failed") / w_sum) * (numerator / denominator);

    Ok(morans_i)
}

/// Calculate Geary's C statistic for spatial autocorrelation
///
/// Geary's C is another measure of spatial autocorrelation that ranges from 0 to 2.
/// Values close to 1 indicate no spatial autocorrelation, values < 1 indicate positive
/// autocorrelation, and values > 1 indicate negative autocorrelation.
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
///
/// # Returns
///
/// * Geary's C statistic
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::gearys_c;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let geary = gearys_c(&values.view(), &weights.view()).expect("Operation failed");
/// println!("Geary's C: {:.3}", geary);
/// ```
#[allow(dead_code)]
pub fn gearys_c<T: Float>(values: &ArrayView1<T>, weights: &ArrayView2<T>) -> SpatialResult<T> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate mean
    let mean = values.sum() / T::from(n).expect("Operation failed");

    // Calculate sum of weights
    let w_sum = weights.sum();

    if w_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Sum of weights cannot be zero".to_string(),
        ));
    }

    // Calculate numerator: sum of (w_ij * (x_i - x_j)^2)
    let mut numerator = T::zero();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = values[i] - values[j];
                numerator = numerator + weights[[i, j]] * diff * diff;
            }
        }
    }

    // Calculate denominator: 2 * W * sum of (x_i - mean)^2
    let variance_sum: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum();

    if variance_sum.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let denominator = (T::one() + T::one()) * w_sum * variance_sum;

    // Geary's C = ((n-1) / 2W) * (numerator / variance_sum)
    let gearys_c = (T::from((n - 1) as i32).expect("Operation failed") / denominator) * numerator;

    Ok(gearys_c)
}

/// Calculate Local Indicators of Spatial Association (LISA) using Local Moran's I
///
/// Local Moran's I identifies clusters and outliers for each location individually.
/// Positive values indicate that a location is part of a cluster of similar values,
/// while negative values indicate spatial outliers.
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
///
/// # Returns
///
/// * Array of Local Moran's I values, one for each location
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::local_morans_i;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let local_i = local_morans_i(&values.view(), &weights.view()).expect("Operation failed");
/// println!("Local Moran's I values: {:?}", local_i);
/// ```
#[allow(dead_code)]
pub fn local_morans_i<T: Float>(
    values: &ArrayView1<T>,
    weights: &ArrayView2<T>,
) -> SpatialResult<Array1<T>> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate global mean
    let mean = values.sum() / T::from(n).expect("Operation failed");

    // Calculate global variance
    let variance: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum()
        / T::from(n).expect("Operation failed");

    if variance.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let mut local_i = Array1::zeros(n);

    for i in 0..n {
        let zi = (values[i] - mean) / variance.sqrt();

        // Calculate weighted sum of neighboring deviations
        let mut weighted_sum = T::zero();
        for j in 0..n {
            if i != j && weights[[i, j]] > T::zero() {
                let zj = (values[j] - mean) / variance.sqrt();
                weighted_sum = weighted_sum + weights[[i, j]] * zj;
            }
        }

        local_i[i] = zi * weighted_sum;
    }

    Ok(local_i)
}

/// Calculate Getis-Ord Gi statistic for hotspot analysis
///
/// The Getis-Ord Gi statistic identifies statistically significant spatial
/// clusters of high values (hotspots) and low values (coldspots).
///
/// # Arguments
///
/// * `values` - The observed values at each location
/// * `weights` - Spatial weights matrix
/// * `include_self` - Whether to include the focal location in the calculation
///
/// # Returns
///
/// * Array of Gi statistics, one for each location
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::getis_ord_gi;
///
/// let values = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let weights = array![
///     [0.0, 1.0, 0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 1.0],
///     [1.0, 0.0, 0.0, 1.0, 0.0],
/// ];
///
/// let gi_stats = getis_ord_gi(&values.view(), &weights.view(), false).expect("Operation failed");
/// println!("Gi statistics: {:?}", gi_stats);
/// ```
#[allow(dead_code)]
pub fn getis_ord_gi<T: Float>(
    values: &ArrayView1<T>,
    weights: &ArrayView2<T>,
    include_self: bool,
) -> SpatialResult<Array1<T>> {
    let n = values.len();

    if weights.shape()[0] != n || weights.shape()[1] != n {
        return Err(SpatialError::DimensionError(
            "Weights matrix dimensions must match number of values".to_string(),
        ));
    }

    // Calculate global mean and variance
    let mean = values.sum() / T::from(n).expect("Operation failed");
    let variance: T = values
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum()
        / T::from(n).expect("Operation failed");

    if variance.is_zero() {
        return Err(SpatialError::ValueError(
            "Variance cannot be zero".to_string(),
        ));
    }

    let _std_dev = variance.sqrt();
    let mut gi_stats = Array1::zeros(n);

    for i in 0..n {
        let mut weighted_sum = T::zero();
        let mut weight_sum = T::zero();
        let mut weight_sum_squared = T::zero();

        for j in 0..n {
            let use_weight = if include_self {
                weights[[i, j]]
            } else if i == j {
                T::zero()
            } else {
                weights[[i, j]]
            };

            if use_weight > T::zero() {
                weighted_sum = weighted_sum + use_weight * values[j];
                weight_sum = weight_sum + use_weight;
                weight_sum_squared = weight_sum_squared + use_weight * use_weight;
            }
        }

        if weight_sum > T::zero() {
            let n_f = T::from(n).expect("Operation failed");
            let expected = weight_sum * mean;

            // Calculate standard deviation of the sum
            let variance_of_sum =
                (n_f * weight_sum_squared - weight_sum * weight_sum) * variance / (n_f - T::one());

            if variance_of_sum > T::zero() {
                gi_stats[i] = (weighted_sum - expected) / variance_of_sum.sqrt();
            }
        }
    }

    Ok(gi_stats)
}

/// Calculate spatial weights matrix based on distance decay
///
/// Creates a spatial weights matrix where weights decay with distance according
/// to a specified function (inverse distance, exponential decay, etc.).
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each location
/// * `max_distance` - Maximum distance for neighbors (beyond this, weight = 0)
/// * `decay_function` - Function to apply distance decay ("inverse", "exponential", "gaussian")
/// * `bandwidth` - Parameter controlling the rate of decay
///
/// # Returns
///
/// * Spatial weights matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::distance_weights_matrix;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let weights = distance_weights_matrix(
///     &coords.view(),
///     2.0,
///     "inverse",
///     1.0
/// ).expect("Operation failed");
///
/// println!("Distance-based weights matrix: {:?}", weights);
/// ```
#[allow(dead_code)]
pub fn distance_weights_matrix<T: Float>(
    coordinates: &ArrayView2<T>,
    max_distance: T,
    decay_function: &str,
    bandwidth: T,
) -> SpatialResult<Array2<T>> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D (x, y)".to_string(),
        ));
    }

    let mut weights = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Calculate Euclidean _distance
                let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
                let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
                let _distance = (dx * dx + dy * dy).sqrt();

                if _distance <= max_distance {
                    let weight = match decay_function {
                        "inverse" => {
                            if _distance > T::zero() {
                                T::one() / (T::one() + _distance / bandwidth)
                            } else {
                                T::zero()
                            }
                        }
                        "exponential" => (-_distance / bandwidth).exp(),
                        "gaussian" => {
                            let exponent = -(_distance * _distance) / (bandwidth * bandwidth);
                            exponent.exp()
                        }
                        _ => {
                            return Err(SpatialError::ValueError(
                                "Unknown decay function. Use 'inverse', 'exponential', or 'gaussian'".to_string(),
                            ));
                        }
                    };

                    weights[[i, j]] = weight;
                }
            }
        }
    }

    Ok(weights)
}

/// Calculate Clark-Evans nearest neighbor index
///
/// The Clark-Evans index compares the average nearest neighbor distance
/// to the expected distance in a random point pattern. Values < 1 indicate
/// clustering, values > 1 indicate regularity, and values ≈ 1 indicate randomness.
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each point
/// * `study_area` - Area of the study region
///
/// # Returns
///
/// * Clark-Evans index (R)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::clark_evans_index;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let ce_index = clark_evans_index(&coords.view(), 4.0).expect("Operation failed");
/// println!("Clark-Evans index: {:.3}", ce_index);
/// ```
#[allow(dead_code)]
pub fn clark_evans_index<T: Float>(coordinates: &ArrayView2<T>, study_area: T) -> SpatialResult<T> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D (x, y)".to_string(),
        ));
    }

    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points to calculate nearest neighbor distances".to_string(),
        ));
    }

    // Calculate nearest neighbor distances
    let mut nn_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_distance = T::infinity();

        for j in 0..n {
            if i != j {
                let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
                let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
                let distance = (dx * dx + dy * dy).sqrt();

                if distance < min_distance {
                    min_distance = distance;
                }
            }
        }

        nn_distances.push(min_distance);
    }

    // Calculate observed mean nearest neighbor distance
    let observed_mean = nn_distances.iter().fold(T::zero(), |acc, &d| acc + d)
        / T::from(n).expect("Operation failed");

    // Calculate expected mean nearest neighbor distance for random pattern
    let density = T::from(n).expect("Operation failed") / study_area;
    let expected_mean = T::one() / (T::from(2.0).expect("Operation failed") * density.sqrt());

    // Clark-Evans index
    let clark_evans = observed_mean / expected_mean;

    Ok(clark_evans)
}

/// Build a contiguity-based spatial weights matrix from polygon adjacency
///
/// Given a list of polygon boundaries (each polygon is a list of vertex indices),
/// two polygons are considered neighbors if they share at least `min_shared_vertices`
/// vertices (rook contiguity uses edges = 2 shared vertices in 2D,
/// queen contiguity uses any shared vertex = 1).
///
/// # Arguments
///
/// * `polygons` - A slice of polygons, each represented as a vector of vertex indices
/// * `n` - Total number of spatial units
/// * `min_shared_vertices` - Minimum number of shared vertices to be considered neighbors
///   (1 for queen contiguity, 2 for rook contiguity)
///
/// # Returns
///
/// * Binary spatial weights matrix (n x n)
pub fn contiguity_weights_matrix(
    polygons: &[Vec<usize>],
    n: usize,
    min_shared_vertices: usize,
) -> SpatialResult<Array2<f64>> {
    if polygons.len() != n {
        return Err(SpatialError::DimensionError(format!(
            "Number of polygons ({}) must match n ({})",
            polygons.len(),
            n
        )));
    }

    if min_shared_vertices == 0 {
        return Err(SpatialError::ValueError(
            "min_shared_vertices must be at least 1".to_string(),
        ));
    }

    let mut weights = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            // Count shared vertices
            let shared = polygons[i]
                .iter()
                .filter(|v| polygons[j].contains(v))
                .count();

            if shared >= min_shared_vertices {
                weights[[i, j]] = 1.0;
                weights[[j, i]] = 1.0;
            }
        }
    }

    Ok(weights)
}

/// Build a k-nearest neighbors spatial weights matrix
///
/// For each location, the k nearest neighbors (by Euclidean distance) receive
/// a weight of 1, and all other locations receive a weight of 0. The resulting
/// matrix is typically asymmetric (i may be a neighbor of j, but j may not be
/// a neighbor of i).
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs for each location (n x d)
/// * `k` - Number of nearest neighbors
///
/// # Returns
///
/// * Binary spatial weights matrix (n x n), where `w[i,j]` = 1 if j is among
///   the k nearest neighbors of i
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::knn_weights_matrix;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let weights = knn_weights_matrix(&coords.view(), 2).expect("Operation failed");
/// // Each point has exactly 2 neighbors
/// for i in 0..4 {
///     let row_sum: f64 = (0..4).map(|j| weights[[i, j]]).sum();
///     assert!((row_sum - 2.0).abs() < 1e-10);
/// }
/// ```
pub fn knn_weights_matrix(coordinates: &ArrayView2<f64>, k: usize) -> SpatialResult<Array2<f64>> {
    let n = coordinates.shape()[0];
    let ndim = coordinates.shape()[1];

    if k == 0 {
        return Err(SpatialError::ValueError("k must be at least 1".to_string()));
    }

    if k >= n {
        return Err(SpatialError::ValueError(format!(
            "k ({}) must be less than number of points ({})",
            k, n
        )));
    }

    let mut weights = Array2::zeros((n, n));

    for i in 0..n {
        // Compute distances from point i to all other points
        let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i != j {
                let mut dist_sq = 0.0;
                for d in 0..ndim {
                    let diff = coordinates[[i, d]] - coordinates[[j, d]];
                    dist_sq += diff * diff;
                }
                distances.push((j, dist_sq));
            }
        }

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign weight 1 to the k nearest neighbors
        for &(j, _) in distances.iter().take(k) {
            weights[[i, j]] = 1.0;
        }
    }

    Ok(weights)
}

/// Compute Ripley's K-function for point pattern analysis
///
/// Ripley's K-function estimates the expected number of points within distance t
/// of a typical point, divided by the intensity (point density). It is used to
/// detect clustering or regularity at different spatial scales.
///
/// K(t) = (area / n^2) * sum_{i != j} I(d_{ij} <= t)
///
/// Where I is the indicator function. Under complete spatial randomness (CSR),
/// K(t) = pi * t^2 in 2D.
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each point (n x 2)
/// * `study_area` - Area of the study region
/// * `distances` - Array of distance thresholds at which to evaluate K
///
/// # Returns
///
/// * Array of K(t) values, one for each distance threshold
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::ripleys_k;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
///     [0.5, 0.5],
/// ];
///
/// let distances = array![0.5, 1.0, 1.5, 2.0];
/// let k_values = ripleys_k(&coords.view(), 4.0, &distances.view()).expect("Operation failed");
/// assert_eq!(k_values.len(), 4);
/// // K should be monotonically non-decreasing
/// for i in 1..k_values.len() {
///     assert!(k_values[i] >= k_values[i - 1]);
/// }
/// ```
pub fn ripleys_k(
    coordinates: &ArrayView2<f64>,
    study_area: f64,
    distances: &ArrayView1<f64>,
) -> SpatialResult<Array1<f64>> {
    let n = coordinates.shape()[0];

    if coordinates.shape()[1] != 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D (x, y)".to_string(),
        ));
    }

    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for Ripley's K".to_string(),
        ));
    }

    if study_area <= 0.0 {
        return Err(SpatialError::ValueError(
            "Study area must be positive".to_string(),
        ));
    }

    let n_dists = distances.len();
    let mut k_values = Array1::zeros(n_dists);
    let n_f = n as f64;
    let intensity = n_f / study_area;

    // Precompute all pairwise distances
    let mut pairwise_dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
            let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
            pairwise_dists.push((dx * dx + dy * dy).sqrt());
        }
    }

    // For each distance threshold, count pairs within that distance
    for (t_idx, &t) in distances.iter().enumerate() {
        let count: usize = pairwise_dists.iter().filter(|&&d| d <= t).count();
        // Each pair (i,j) counted once, but K sums over ordered pairs i != j
        // so multiply by 2
        k_values[t_idx] = (study_area / (n_f * n_f)) * (2.0 * count as f64);
    }

    Ok(k_values)
}

/// Compute Ripley's L-function (variance-stabilized K-function)
///
/// L(t) = sqrt(K(t) / pi), which linearizes K under CSR so that L(t) = t
/// for a complete spatial random process.
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each point (n x 2)
/// * `study_area` - Area of the study region
/// * `distances` - Array of distance thresholds at which to evaluate L
///
/// # Returns
///
/// * Array of L(t) values, one for each distance threshold
pub fn ripleys_l(
    coordinates: &ArrayView2<f64>,
    study_area: f64,
    distances: &ArrayView1<f64>,
) -> SpatialResult<Array1<f64>> {
    let k_values = ripleys_k(coordinates, study_area, distances)?;

    let l_values = k_values.mapv(|k| {
        if k >= 0.0 {
            (k / std::f64::consts::PI).sqrt()
        } else {
            0.0
        }
    });

    Ok(l_values)
}

/// Calculate the average nearest neighbor distance statistic
///
/// Computes the mean distance from each point to its nearest neighbor,
/// and returns the observed mean, the expected mean under CSR, the z-score,
/// and the nearest-neighbor index (R = observed / expected).
///
/// # Arguments
///
/// * `coordinates` - Array of coordinate pairs [x, y] for each point (n x 2)
/// * `study_area` - Area of the study region
///
/// # Returns
///
/// * `AnnResult` containing observed mean, expected mean, z-score, and R index
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_spatial::spatial_stats::average_nearest_neighbor;
///
/// let coords = array![
///     [0.0, 0.0],
///     [1.0, 0.0],
///     [0.0, 1.0],
///     [1.0, 1.0],
/// ];
///
/// let result = average_nearest_neighbor(&coords.view(), 4.0).expect("Operation failed");
/// assert!(result.observed_mean > 0.0);
/// assert!(result.expected_mean > 0.0);
/// // Regular grid should have R > 1
/// assert!(result.r_index > 1.0);
/// ```
pub fn average_nearest_neighbor(
    coordinates: &ArrayView2<f64>,
    study_area: f64,
) -> SpatialResult<AnnResult> {
    let n = coordinates.shape()[0];
    let ndim = coordinates.shape()[1];

    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for average nearest neighbor".to_string(),
        ));
    }

    if study_area <= 0.0 {
        return Err(SpatialError::ValueError(
            "Study area must be positive".to_string(),
        ));
    }

    // Calculate nearest neighbor distances
    let mut nn_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_distance = f64::INFINITY;

        for j in 0..n {
            if i != j {
                let mut dist_sq = 0.0;
                for d in 0..ndim {
                    let diff = coordinates[[i, d]] - coordinates[[j, d]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                if dist < min_distance {
                    min_distance = dist;
                }
            }
        }

        nn_distances.push(min_distance);
    }

    // Observed mean nearest neighbor distance
    let observed_mean: f64 = nn_distances.iter().sum::<f64>() / n as f64;

    // Expected mean under CSR: E(d) = 1 / (2 * sqrt(density))
    let density = n as f64 / study_area;
    let expected_mean = 1.0 / (2.0 * density.sqrt());

    // Standard error of the mean under CSR
    let se = 0.26136 / (n as f64 * density).sqrt();

    // Z-score
    let z_score = if se > 0.0 {
        (observed_mean - expected_mean) / se
    } else {
        0.0
    };

    // Nearest neighbor index R
    let r_index = if expected_mean > 0.0 {
        observed_mean / expected_mean
    } else {
        0.0
    };

    Ok(AnnResult {
        observed_mean,
        expected_mean,
        z_score,
        r_index,
        nn_distances,
    })
}

/// Result of the average nearest neighbor analysis
#[derive(Debug, Clone)]
pub struct AnnResult {
    /// Observed mean nearest neighbor distance
    pub observed_mean: f64,
    /// Expected mean nearest neighbor distance under complete spatial randomness
    pub expected_mean: f64,
    /// Z-score for significance testing
    pub z_score: f64,
    /// Nearest neighbor index (R = observed / expected)
    /// R < 1 indicates clustering, R > 1 indicates dispersion, R ≈ 1 indicates randomness
    pub r_index: f64,
    /// Individual nearest neighbor distances for each point
    pub nn_distances: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_morans_i() {
        // Test with a simple case where adjacent values are similar
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let moran = morans_i(&values.view(), &weights.view()).expect("Operation failed");

        // Should be positive due to spatial clustering
        assert!(moran > 0.0);
    }

    #[test]
    fn test_gearys_c() {
        // Test with clustered data
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let geary = gearys_c(&values.view(), &weights.view()).expect("Operation failed");

        // Should be less than 1 due to positive spatial autocorrelation
        assert!(geary < 1.0);
    }

    #[test]
    fn test_local_morans_i() {
        let values = array![1.0, 1.0, 3.0, 3.0, 3.0];
        let weights = array![
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        let local_i = local_morans_i(&values.view(), &weights.view()).expect("Operation failed");

        // Should have 5 values (one for each location)
        assert_eq!(local_i.len(), 5);
    }

    #[test]
    fn test_distance_weights_matrix() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0],];

        let weights =
            distance_weights_matrix(&coords.view(), 1.5, "inverse", 1.0).expect("Operation failed");

        // Check dimensions
        assert_eq!(weights.shape(), &[4, 4]);

        // Diagonal should be zero
        for i in 0..4 {
            assert_relative_eq!(weights[[i, i]], 0.0, epsilon = 1e-10);
        }

        // Points (0,0) and (1,0) should have positive weight (distance = 1)
        assert!(weights[[0, 1]] > 0.0);
        assert!(weights[[1, 0]] > 0.0);

        // Points (0,0) and (2,2) should have zero weight (distance > max_distance)
        assert_relative_eq!(weights[[0, 3]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clark_evans_index() {
        // Perfect grid pattern should have R > 1 (regular)
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let ce_index = clark_evans_index(&coords.view(), 4.0).expect("Operation failed");

        // Grid pattern should be regular (R > 1)
        assert!(ce_index > 1.0);
    }

    #[test]
    fn test_contiguity_weights_matrix_queen() {
        // 4 polygons sharing vertices (queen contiguity)
        // Polygon 0: vertices {0, 1, 2, 3}
        // Polygon 1: vertices {2, 3, 4, 5}
        // Polygon 2: vertices {3, 5, 6, 7}
        // Polygon 3: vertices {8, 9, 10, 11}
        let polygons = vec![
            vec![0, 1, 2, 3],
            vec![2, 3, 4, 5],
            vec![3, 5, 6, 7],
            vec![8, 9, 10, 11],
        ];

        let weights = contiguity_weights_matrix(&polygons, 4, 1).expect("Operation failed");

        assert_eq!(weights.shape(), &[4, 4]);

        // Polygon 0 and 1 share vertices 2, 3
        assert_relative_eq!(weights[[0, 1]], 1.0);
        assert_relative_eq!(weights[[1, 0]], 1.0);

        // Polygon 0 and 2 share vertex 3
        assert_relative_eq!(weights[[0, 2]], 1.0);

        // Polygon 1 and 2 share vertices 3, 5
        assert_relative_eq!(weights[[1, 2]], 1.0);

        // Polygon 3 is isolated
        assert_relative_eq!(weights[[0, 3]], 0.0);
        assert_relative_eq!(weights[[1, 3]], 0.0);
        assert_relative_eq!(weights[[2, 3]], 0.0);
    }

    #[test]
    fn test_contiguity_weights_matrix_rook() {
        // With rook contiguity (min_shared = 2), only edges count
        let polygons = vec![
            vec![0, 1, 2, 3],
            vec![2, 3, 4, 5],
            vec![3, 5, 6, 7], // shares only vertex 3 with polygon 0
            vec![8, 9, 10, 11],
        ];

        let weights = contiguity_weights_matrix(&polygons, 4, 2).expect("Operation failed");

        // Polygon 0 and 1 share 2 vertices => neighbors
        assert_relative_eq!(weights[[0, 1]], 1.0);

        // Polygon 0 and 2 share only vertex 3 => NOT neighbors with rook
        assert_relative_eq!(weights[[0, 2]], 0.0);

        // Polygon 1 and 2 share vertices 3, 5 => neighbors
        assert_relative_eq!(weights[[1, 2]], 1.0);
    }

    #[test]
    fn test_knn_weights_matrix() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0],];

        let weights = knn_weights_matrix(&coords.view(), 2).expect("Operation failed");

        assert_eq!(weights.shape(), &[5, 5]);

        // Each point should have exactly 2 neighbors
        for i in 0..5 {
            let row_sum: f64 = (0..5).map(|j| weights[[i, j]]).sum();
            assert_relative_eq!(row_sum, 2.0, epsilon = 1e-10);
        }

        // Diagonal should be zero
        for i in 0..5 {
            assert_relative_eq!(weights[[i, i]], 0.0);
        }

        // Point (0,0) should have (1,0) and (0,1) as nearest neighbors
        assert_relative_eq!(weights[[0, 1]], 1.0);
        assert_relative_eq!(weights[[0, 2]], 1.0);
    }

    #[test]
    fn test_knn_weights_errors() {
        let coords = array![[0.0, 0.0], [1.0, 0.0]];

        // k = 0 should fail
        assert!(knn_weights_matrix(&coords.view(), 0).is_err());

        // k >= n should fail
        assert!(knn_weights_matrix(&coords.view(), 2).is_err());
    }

    #[test]
    fn test_ripleys_k() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],];

        let distances = array![0.5, 1.0, 1.5, 2.0];
        let k_values = ripleys_k(&coords.view(), 4.0, &distances.view()).expect("Operation failed");

        assert_eq!(k_values.len(), 4);

        // K should be monotonically non-decreasing
        for i in 1..k_values.len() {
            assert!(
                k_values[i] >= k_values[i - 1],
                "K should be non-decreasing: K[{}] = {} < K[{}] = {}",
                i,
                k_values[i],
                i - 1,
                k_values[i - 1]
            );
        }

        // At distance 0.5, only (0.5,0.5) is within 0.5 of some points
        // K(0) should be 0 (no points at distance 0)
        // K at larger distances should be larger
        assert!(k_values[3] > k_values[0]);
    }

    #[test]
    fn test_ripleys_l() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],];

        let distances = array![0.5, 1.0, 1.5];
        let l_values = ripleys_l(&coords.view(), 4.0, &distances.view()).expect("Operation failed");

        assert_eq!(l_values.len(), 3);

        // L should be non-negative
        for &l in l_values.iter() {
            assert!(l >= 0.0);
        }
    }

    #[test]
    fn test_ripleys_k_errors() {
        let coords = array![[0.0, 0.0]]; // Only 1 point
        let distances = array![1.0];

        assert!(ripleys_k(&coords.view(), 1.0, &distances.view()).is_err());

        // Negative study area
        let coords2 = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(ripleys_k(&coords2.view(), -1.0, &distances.view()).is_err());
    }

    #[test]
    fn test_average_nearest_neighbor() {
        // Regular grid pattern
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let result = average_nearest_neighbor(&coords.view(), 4.0).expect("Operation failed");

        assert!(result.observed_mean > 0.0);
        assert!(result.expected_mean > 0.0);
        assert_eq!(result.nn_distances.len(), 4);

        // All nearest neighbor distances should be 1.0 for a unit grid
        for &d in &result.nn_distances {
            assert_relative_eq!(d, 1.0, epsilon = 1e-10);
        }

        // Regular pattern should have R > 1
        assert!(result.r_index > 1.0);
    }

    #[test]
    fn test_average_nearest_neighbor_clustered() {
        // Clustered pattern: points very close together
        let coords = array![[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01],];

        let result = average_nearest_neighbor(&coords.view(), 100.0).expect("Operation failed");

        // Clustered pattern should have R < 1
        assert!(
            result.r_index < 1.0,
            "Expected R < 1 for clustered pattern, got {}",
            result.r_index
        );
    }

    #[test]
    fn test_average_nearest_neighbor_errors() {
        let coords = array![[0.0, 0.0]]; // Only 1 point

        assert!(average_nearest_neighbor(&coords.view(), 1.0).is_err());

        // Negative study area
        let coords2 = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(average_nearest_neighbor(&coords2.view(), -1.0).is_err());
    }
}
