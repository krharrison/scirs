//! Mean Shift clustering implementation
//!
//! Mean Shift is a non-parametric clustering technique that does not require
//! specifying the number of clusters in advance. It works by iteratively
//! shifting each data point towards the mode of the local density.
//!
//! # Features
//!
//! - **Flat kernel** and **Gaussian kernel** support
//! - **Bandwidth estimation**: Silverman's rule, Scott's rule, k-NN quantile
//! - **Bin seeding** for acceleration on large datasets
//! - **Cluster-all** mode and noise detection mode
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_cluster::meanshift::{mean_shift, MeanShiftOptions, KernelType};
//!
//! let data = array![
//!     [1.0, 1.0], [2.0, 1.0], [1.0, 0.0],
//!     [4.0, 7.0], [3.0, 5.0], [3.0, 6.0]
//! ];
//!
//! let options = MeanShiftOptions {
//!     bandwidth: Some(2.0),
//!     kernel: KernelType::Gaussian,
//!     ..Default::default()
//! };
//!
//! let (centers, labels) = mean_shift(&data.view(), options).expect("Operation failed");
//! println!("Number of clusters: {}", centers.nrows());
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::marker::{Send, Sync};

use crate::error::ClusteringError;
use scirs2_core::validation::{
    check_positive, checkarray_finite, clustering::validate_clustering_data,
    parameters::check_unit_interval,
};
use scirs2_spatial::distance::EuclideanDistance;
use scirs2_spatial::kdtree::KDTree;

/// Kernel type for Mean Shift
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Flat (uniform) kernel: all points within bandwidth contribute equally
    Flat,
    /// Gaussian kernel: points are weighted by exp(-||x - xi||^2 / (2 * bandwidth^2))
    Gaussian,
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::Flat
    }
}

/// Bandwidth estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthEstimator {
    /// k-NN quantile method (default): uses quantile of nearest neighbor distances
    KNNQuantile,
    /// Silverman's rule of thumb: h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
    Silverman,
    /// Scott's rule: h = n^(-1/(d+4)) * std
    Scott,
}

impl Default for BandwidthEstimator {
    fn default() -> Self {
        BandwidthEstimator::KNNQuantile
    }
}

/// Configuration options for Mean Shift algorithm
pub struct MeanShiftOptions<T: Float> {
    /// Bandwidth parameter.
    /// If not provided, it will be estimated from the data.
    pub bandwidth: Option<T>,

    /// Points used as initial kernel locations.
    /// If not provided, either all points or discretized bins will be used.
    pub seeds: Option<Array2<T>>,

    /// If true, initial kernels are located on a grid with bin_size = bandwidth.
    pub bin_seeding: bool,

    /// Only bins with at least min_bin_freq points will be selected as seeds.
    pub min_bin_freq: usize,

    /// If true, all points are assigned to clusters, even outliers.
    pub cluster_all: bool,

    /// Maximum number of iterations for a single seed.
    pub max_iter: usize,

    /// Kernel type to use
    pub kernel: KernelType,

    /// Bandwidth estimation method (used when bandwidth is None)
    pub bandwidth_estimator: BandwidthEstimator,
}

impl<T: Float> Default for MeanShiftOptions<T> {
    fn default() -> Self {
        Self {
            bandwidth: None,
            seeds: None,
            bin_seeding: false,
            min_bin_freq: 1,
            cluster_all: true,
            max_iter: 300,
            kernel: KernelType::Flat,
            bandwidth_estimator: BandwidthEstimator::KNNQuantile,
        }
    }
}

/// FloatPoint wrapper to make f32/f64 arrays comparable and hashable
#[derive(Debug, Clone)]
struct FloatPoint<T: Float>(Vec<T>);

impl<T: Float> PartialEq for FloatPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if !a.is_finite() || !b.is_finite() || (*a - *b).abs() > T::epsilon() {
                return false;
            }
        }
        true
    }
}

impl<T: Float> Eq for FloatPoint<T> {}

impl<T: Float> Hash for FloatPoint<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for value in &self.0 {
            let bits = if let Some(bits) = value.to_f64() {
                (bits * 1e10).round() as i64
            } else {
                0
            };
            bits.hash(state);
        }
    }
}

/// Estimate bandwidth using Silverman's rule of thumb
///
/// h = 0.9 * min(std, IQR/1.34) * n^(-1/5)
///
/// This works well for normally distributed data.
pub fn estimate_bandwidth_silverman<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
) -> Result<T, ClusteringError> {
    checkarray_finite(data, "data")?;

    let n = data.nrows();
    if n < 2 {
        return Ok(T::from(1.0).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?);
    }

    let n_features = data.ncols();
    let n_f = T::from(n)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to convert n".into()))?;

    // Compute bandwidth per dimension and take the average
    let mut bandwidth_sum = T::zero();

    for col_idx in 0..n_features {
        // Gather column values
        let mut values: Vec<T> = (0..n).map(|i| data[[i, col_idx]]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Standard deviation
        let mean = values.iter().fold(T::zero(), |a, &b| a + b) / n_f;
        let var = values
            .iter()
            .fold(T::zero(), |acc, &v| acc + (v - mean) * (v - mean))
            / n_f;
        let std_dev = var.sqrt();

        // IQR
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let iqr = values[q3_idx.min(n - 1)] - values[q1_idx];
        let one_point_three_four = T::from(1.34).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?;
        let iqr_scaled = iqr / one_point_three_four;

        // min(std, IQR/1.34), but skip IQR if it's zero
        let spread = if iqr_scaled > T::zero() && iqr_scaled < std_dev {
            iqr_scaled
        } else {
            std_dev
        };

        // Silverman factor: 0.9 * spread * n^(-1/5)
        let zero_nine = T::from(0.9).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?;
        let exponent = T::from(-0.2).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?;
        let n_factor = n_f.powf(exponent);

        let h = zero_nine * spread * n_factor;
        bandwidth_sum = bandwidth_sum + h;
    }

    let n_feat_f = T::from(n_features)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to convert n_features".into()))?;
    let bandwidth = bandwidth_sum / n_feat_f;

    // Ensure positive bandwidth
    if bandwidth <= T::zero() {
        return Ok(T::from(1.0).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?);
    }

    Ok(bandwidth)
}

/// Estimate bandwidth using Scott's rule
///
/// h = n^(-1/(d+4)) * std
///
/// Good general-purpose estimator for multivariate data.
pub fn estimate_bandwidth_scott<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
) -> Result<T, ClusteringError> {
    checkarray_finite(data, "data")?;

    let n = data.nrows();
    if n < 2 {
        return Ok(T::from(1.0).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?);
    }

    let n_features = data.ncols();
    let n_f = T::from(n)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to convert n".into()))?;

    // Scott's exponent: -1/(d+4)
    let d_plus_4 = T::from(n_features as f64 + 4.0)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to convert dimension".into()))?;
    let exponent = T::from(-1.0)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to convert constant".into()))?
        / d_plus_4;
    let n_factor = n_f.powf(exponent);

    // Average standard deviation across dimensions
    let mut std_sum = T::zero();
    for col_idx in 0..n_features {
        let mean = (0..n)
            .map(|i| data[[i, col_idx]])
            .fold(T::zero(), |a, b| a + b)
            / n_f;
        let var = (0..n)
            .map(|i| {
                let diff = data[[i, col_idx]] - mean;
                diff * diff
            })
            .fold(T::zero(), |a, b| a + b)
            / n_f;
        std_sum = std_sum + var.sqrt();
    }

    let avg_std = std_sum
        / T::from(n_features).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert n_features".into())
        })?;

    let bandwidth = n_factor * avg_std;

    if bandwidth <= T::zero() {
        return Ok(T::from(1.0).ok_or_else(|| {
            ClusteringError::ComputationError("Failed to convert constant".into())
        })?);
    }

    Ok(bandwidth)
}

/// Estimate the bandwidth using k-NN quantile method.
///
/// Computes the average distance to the k-th nearest neighbor across all points,
/// where k = quantile * n_samples.
pub fn estimate_bandwidth<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
    quantile: Option<T>,
    n_samples: Option<usize>,
    _random_state: Option<u64>,
) -> Result<T, ClusteringError> {
    checkarray_finite(data, "data")?;

    let quantile = quantile
        .unwrap_or_else(|| T::from(0.3).unwrap_or_else(|| T::from(0.3f64).unwrap_or(T::one())));
    let _quantile = check_unit_interval(quantile, "quantile", "estimate_bandwidth")?;

    // Select a subset of samples if specified
    let data = if let Some(n) = n_samples {
        if n >= data.nrows() {
            data.to_owned()
        } else {
            let mut rng = scirs2_core::random::rng();
            use scirs2_core::random::seq::SliceRandom;
            let mut indices: Vec<usize> = (0..data.nrows()).collect();
            indices.shuffle(&mut rng);

            let indices = &indices[0..n];
            let mut sampled_data = Array2::zeros((n, data.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                sampled_data.row_mut(i).assign(&data.row(idx));
            }
            sampled_data
        }
    } else {
        data.to_owned()
    };

    let n_neighbors = (T::from(data.nrows()).unwrap_or(T::one()) * quantile)
        .to_usize()
        .unwrap_or(1)
        .max(1)
        .min(data.nrows().saturating_sub(1));

    // Build KDTree for nearest neighbor search
    let kdtree = KDTree::<_, EuclideanDistance<T>>::new(&data)
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e)))?;

    let mut bandwidth_sum = T::zero();

    let batch_size = 500;
    for i in (0..data.nrows()).step_by(batch_size) {
        let end = (i + batch_size).min(data.nrows());
        let batch = data.slice(scirs2_core::ndarray::s![i..end, ..]);

        for row in batch.rows() {
            let (_, distances) = kdtree.query(&row.to_vec(), n_neighbors + 1).map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
            })?;

            if distances.len() > 1 {
                let kth_dist = distances
                    .last()
                    .copied()
                    .unwrap_or_else(|| T::from(1.0).unwrap_or(T::one()));
                bandwidth_sum = bandwidth_sum + kth_dist;
            } else if !distances.is_empty() {
                bandwidth_sum = bandwidth_sum + T::from(1.0).unwrap_or(T::one());
            }
        }
    }

    Ok(bandwidth_sum / T::from(data.nrows()).unwrap_or(T::one()))
}

/// Find seeds for mean_shift by binning data onto a grid.
pub fn get_bin_seeds<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
    bin_size: T,
    min_bin_freq: usize,
) -> Array2<T> {
    if bin_size <= T::zero() {
        return data.to_owned();
    }

    let mut bin_sizes: HashMap<FloatPoint<T>, usize> = HashMap::new();

    for row in data.rows() {
        let mut binned_point = Vec::with_capacity(row.len());
        for &val in row.iter() {
            binned_point.push((val / bin_size).round() * bin_size);
        }
        let point = FloatPoint::<T>(binned_point);
        *bin_sizes.entry(point).or_insert(0) += 1;
    }

    let seeds: Vec<Vec<T>> = bin_sizes
        .into_iter()
        .filter(|(_, freq)| *freq >= min_bin_freq)
        .map(|(point, _)| point.0)
        .collect();

    if seeds.len() == data.nrows() {
        return data.to_owned();
    }

    if seeds.is_empty() {
        Array2::zeros((0, data.ncols()))
    } else {
        let mut result = Array2::zeros((seeds.len(), data.ncols()));
        for (i, seed) in seeds.into_iter().enumerate() {
            for (j, val) in seed.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }
        result
    }
}

/// Mean Shift single seed update with flat kernel
fn mean_shift_single_seed_flat<
    T: Float
        + Display
        + std::iter::Sum
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + scirs2_core::ndarray::ScalarOperand,
>(
    seed: ArrayView1<T>,
    data: &ArrayView2<T>,
    bandwidth: T,
    max_iter: usize,
) -> (Vec<T>, usize, usize) {
    let stop_thresh = bandwidth * T::from(1e-3).unwrap_or(T::epsilon());
    let mut my_mean = seed.to_owned();
    let mut completed_iterations = 0;

    let owned_data = data.to_owned();
    let kdtree = match KDTree::<_, EuclideanDistance<T>>::new(&owned_data) {
        Ok(tree) => tree,
        Err(_) => return (seed.to_vec(), 0, 0),
    };

    loop {
        let (indices, _distances) = match kdtree.query_radius(&my_mean.to_vec(), bandwidth) {
            Ok((idx, distances)) => (idx, distances),
            Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
        };

        if indices.is_empty() {
            break;
        }
        let my_old_mean = my_mean.clone();

        // Flat kernel: equal weights for all neighbors
        my_mean.fill(T::zero());
        let mut sum = Array1::zeros(my_mean.dim());
        for &point_idx in &indices {
            let row_clone = data.row(point_idx).to_owned();
            for (s, v) in sum.iter_mut().zip(row_clone.iter()) {
                *s = *s + *v;
            }
        }
        my_mean = sum / T::from(indices.len()).unwrap_or(T::one());

        let mut dist_squared = T::zero();
        for (a, b) in my_mean.iter().zip(my_old_mean.iter()) {
            dist_squared = dist_squared + (*a - *b) * (*a - *b);
        }
        let dist = dist_squared.sqrt();

        if dist <= stop_thresh || completed_iterations == max_iter {
            break;
        }

        completed_iterations += 1;
    }

    let (final_indices, _) = match kdtree.query_radius(&my_mean.to_vec(), bandwidth) {
        Ok((idx, distances)) => (idx, distances),
        Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
    };

    (my_mean.to_vec(), final_indices.len(), completed_iterations)
}

/// Mean Shift single seed update with Gaussian kernel
fn mean_shift_single_seed_gaussian<
    T: Float
        + Display
        + std::iter::Sum
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + scirs2_core::ndarray::ScalarOperand,
>(
    seed: ArrayView1<T>,
    data: &ArrayView2<T>,
    bandwidth: T,
    max_iter: usize,
) -> (Vec<T>, usize, usize) {
    let stop_thresh = bandwidth * T::from(1e-3).unwrap_or(T::epsilon());
    let mut my_mean = seed.to_owned();
    let mut completed_iterations = 0;
    let bw_sq = bandwidth * bandwidth;

    // Use 3*bandwidth as the search radius for Gaussian kernel
    let search_radius = bandwidth * T::from(3.0).unwrap_or(T::one() + T::one() + T::one());

    let owned_data = data.to_owned();
    let kdtree = match KDTree::<_, EuclideanDistance<T>>::new(&owned_data) {
        Ok(tree) => tree,
        Err(_) => return (seed.to_vec(), 0, 0),
    };

    loop {
        let (indices, distances) = match kdtree.query_radius(&my_mean.to_vec(), search_radius) {
            Ok((idx, distances)) => (idx, distances),
            Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
        };

        if indices.is_empty() {
            break;
        }
        let my_old_mean = my_mean.clone();

        // Gaussian kernel: weight = exp(-dist^2 / (2 * bw^2))
        let two = T::from(2.0).unwrap_or(T::one() + T::one());
        let n_features = my_mean.dim();
        let mut weighted_sum = Array1::zeros(n_features);
        let mut weight_total = T::zero();

        for (local_idx, &point_idx) in indices.iter().enumerate() {
            let dist = distances[local_idx];
            let dist_sq = dist * dist;
            let weight = (-dist_sq / (two * bw_sq)).exp();

            let row = data.row(point_idx);
            for (ws, &v) in weighted_sum.iter_mut().zip(row.iter()) {
                *ws = *ws + v * weight;
            }
            weight_total = weight_total + weight;
        }

        if weight_total > T::zero() {
            my_mean = weighted_sum / weight_total;
        }

        let mut dist_squared = T::zero();
        for (a, b) in my_mean.iter().zip(my_old_mean.iter()) {
            dist_squared = dist_squared + (*a - *b) * (*a - *b);
        }
        let dist = dist_squared.sqrt();

        if dist <= stop_thresh || completed_iterations == max_iter {
            break;
        }

        completed_iterations += 1;
    }

    let (final_indices, _) = match kdtree.query_radius(&my_mean.to_vec(), bandwidth) {
        Ok((idx, distances)) => (idx, distances),
        Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
    };

    (my_mean.to_vec(), final_indices.len(), completed_iterations)
}

/// Perform Mean Shift clustering.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array.
/// * `options` - The Mean Shift algorithm options.
///
/// # Returns
///
/// * `Result<(Array2<T>, Array1<i32>), ClusteringError>` - Tuple of (cluster centers, labels).
pub fn mean_shift<
    T: Float
        + Display
        + std::iter::Sum
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + Debug,
>(
    data: &ArrayView2<T>,
    options: MeanShiftOptions<T>,
) -> Result<(Array2<T>, Array1<i32>), ClusteringError> {
    let mut model = MeanShift::new(options);
    let model = model.fit(data)?;
    Ok((
        model.cluster_centers().to_owned(),
        model.labels().to_owned(),
    ))
}

/// Mean Shift clustering model.
pub struct MeanShift<T: Float> {
    options: MeanShiftOptions<T>,
    cluster_centers_: Option<Array2<T>>,
    labels_: Option<Array1<i32>>,
    n_iter_: usize,
    bandwidth_used_: Option<T>,
}

impl<
        T: Float
            + Display
            + std::iter::Sum
            + FromPrimitive
            + Send
            + Sync
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + Debug,
    > MeanShift<T>
{
    /// Create a new Mean Shift instance.
    pub fn new(options: MeanShiftOptions<T>) -> Self {
        Self {
            options,
            cluster_centers_: None,
            labels_: None,
            n_iter_: 0,
            bandwidth_used_: None,
        }
    }

    /// Fit the Mean Shift model to data.
    pub fn fit(&mut self, data: &ArrayView2<T>) -> Result<&mut Self, ClusteringError> {
        let config = crate::input_validation::ValidationConfig::default();
        crate::input_validation::validate_clustering_data(data.view(), &config)?;

        let (n_samples, n_features) = data.dim();

        // Determine bandwidth
        let bandwidth = match self.options.bandwidth {
            Some(bw) => check_positive(bw, "bandwidth")?,
            None => match self.options.bandwidth_estimator {
                BandwidthEstimator::Silverman => estimate_bandwidth_silverman(data)?,
                BandwidthEstimator::Scott => estimate_bandwidth_scott(data)?,
                BandwidthEstimator::KNNQuantile => {
                    estimate_bandwidth(data, Some(T::from(0.3).unwrap_or(T::one())), None, None)?
                }
            },
        };
        self.bandwidth_used_ = Some(bandwidth);

        // Get seeds
        let seeds = match &self.options.seeds {
            Some(s) => s.clone(),
            None => {
                if self.options.bin_seeding {
                    get_bin_seeds(data, bandwidth, self.options.min_bin_freq)
                } else {
                    data.to_owned()
                }
            }
        };

        if seeds.is_empty() {
            return Err(ClusteringError::ComputationError(
                "No seeds provided and bin seeding produced no seeds".to_string(),
            ));
        }

        // Run mean shift on each seed with the appropriate kernel
        let kernel = self.options.kernel;
        let max_iter = self.options.max_iter;

        let seed_results: Vec<_> = seeds
            .axis_iter(Axis(0))
            .map(|seed| match kernel {
                KernelType::Flat => mean_shift_single_seed_flat(seed, data, bandwidth, max_iter),
                KernelType::Gaussian => {
                    mean_shift_single_seed_gaussian(seed, data, bandwidth, max_iter)
                }
            })
            .collect();

        // Process results
        let mut center_intensity_dict: HashMap<FloatPoint<T>, usize> = HashMap::new();
        for (center, size, iterations) in seed_results {
            if size > 0 {
                center_intensity_dict.insert(FloatPoint(center), size);
            }
            self.n_iter_ = self.n_iter_.max(iterations);
        }

        if center_intensity_dict.is_empty() {
            return Err(ClusteringError::ComputationError(format!(
                "No point was within bandwidth={} of any seed. \
                 Try a different seeding strategy or increase the bandwidth.",
                bandwidth
            )));
        }

        // Sort centers by intensity
        let mut sorted_by_intensity: Vec<_> = center_intensity_dict.into_iter().collect();
        sorted_by_intensity.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                a.0 .0
                    .iter()
                    .zip(b.0 .0.iter())
                    .find_map(|(a_val, b_val)| a_val.partial_cmp(b_val))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        if !self.options.cluster_all {
            let min_density_threshold = 2;
            sorted_by_intensity.retain(|(_, intensity)| *intensity >= min_density_threshold);

            if sorted_by_intensity.is_empty() {
                return Err(ClusteringError::ComputationError(
                    "No clusters found with sufficient density.".to_string(),
                ));
            }
        }

        // Convert to Array2
        let mut sorted_centers = Array2::zeros((sorted_by_intensity.len(), n_features));
        for (i, center_) in sorted_by_intensity.iter().enumerate() {
            for (j, &val) in center_.0 .0.iter().enumerate() {
                sorted_centers[[i, j]] = val;
            }
        }

        // Remove near-duplicate centers
        let mut unique = vec![true; sorted_centers.nrows()];

        let kdtree = KDTree::<_, EuclideanDistance<T>>::new(&sorted_centers).map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
        })?;

        let merge_threshold = bandwidth * T::from(0.1).unwrap_or(T::epsilon());

        for i in 0..sorted_centers.nrows() {
            if unique[i] {
                let (indices_, _) = kdtree
                    .query_radius(&sorted_centers.row(i).to_vec(), merge_threshold)
                    .map_err(|e| {
                        ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                    })?;

                for &idx in indices_.iter() {
                    if idx != i {
                        unique[idx] = false;
                    }
                }
            }
        }

        let unique_indices: Vec<_> = unique
            .iter()
            .enumerate()
            .filter(|&(_, &is_unique)| is_unique)
            .map(|(i_, _)| i_)
            .collect();

        let mut cluster_centers = Array2::zeros((unique_indices.len(), n_features));
        for (i, &idx) in unique_indices.iter().enumerate() {
            cluster_centers.row_mut(i).assign(&sorted_centers.row(idx));
        }

        // Assign labels
        let centers_kdtree =
            KDTree::<_, EuclideanDistance<T>>::new(&cluster_centers).map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
            })?;

        let mut labels = Array1::zeros(n_samples);

        let batch_size = 1000;
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let batch = data.slice(scirs2_core::ndarray::s![i..end, ..]);

            for (row_idx, row) in batch.rows().into_iter().enumerate() {
                let point_idx = i + row_idx;

                let (indices, distances) = centers_kdtree.query(&row.to_vec(), 1).map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                })?;

                if !indices.is_empty() {
                    let idx = indices[0];
                    let distance = T::from(distances[0]).unwrap_or(T::zero());

                    if self.options.cluster_all || (distance <= bandwidth) {
                        labels[point_idx] =
                            T::to_i32(&T::from(idx).unwrap_or(T::zero())).unwrap_or(0);
                    } else {
                        labels[point_idx] = -1;
                    }
                } else {
                    labels[point_idx] = -1;
                }
            }
        }

        self.cluster_centers_ = Some(cluster_centers);
        self.labels_ = Some(labels);

        Ok(self)
    }

    /// Get cluster centers found by the algorithm.
    pub fn cluster_centers(&self) -> &Array2<T> {
        self.cluster_centers_
            .as_ref()
            .expect("Model has not been fitted yet")
    }

    /// Get labels assigned to each data point.
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_
            .as_ref()
            .expect("Model has not been fitted yet")
    }

    /// Get the number of iterations performed for the most complex seed.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Get the bandwidth that was actually used (useful when auto-estimated).
    pub fn bandwidth_used(&self) -> Option<T> {
        self.bandwidth_used_
    }

    /// Predict the closest cluster each sample in data belongs to.
    pub fn predict(&self, data: &ArrayView2<T>) -> Result<Array1<i32>, ClusteringError> {
        let centers = self.cluster_centers_.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Model has not been fitted yet".to_string())
        })?;

        checkarray_finite(data, "prediction data")?;

        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        let kdtree = KDTree::<_, EuclideanDistance<T>>::new(centers).map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
        })?;

        let batch_size = 1000;
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let batch = data.slice(scirs2_core::ndarray::s![i..end, ..]);

            for (row_idx, row) in batch.rows().into_iter().enumerate() {
                let (indices_, _distances) = kdtree.query(&row.to_vec(), 1).map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                })?;

                if !indices_.is_empty() {
                    labels[i + row_idx] =
                        T::to_i32(&T::from(indices_[0]).unwrap_or(T::zero())).unwrap_or(0);
                } else {
                    labels[i + row_idx] = -1;
                }
            }
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};
    use std::collections::HashSet;

    fn make_test_data() -> Array2<f64> {
        array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ]
    }

    #[test]
    fn test_estimate_bandwidth() {
        let data = make_test_data();
        let bandwidth = estimate_bandwidth(&data.view(), Some(0.4), None, None)
            .expect("Bandwidth estimation should succeed");

        assert!(
            bandwidth > 0.0,
            "Bandwidth should be positive, got: {}",
            bandwidth
        );
        assert!(
            bandwidth < 20.0,
            "Bandwidth should be reasonable, got: {}",
            bandwidth
        );
    }

    #[test]
    fn test_estimate_bandwidth_silverman() {
        let data = make_test_data();
        let bandwidth = estimate_bandwidth_silverman(&data.view())
            .expect("Silverman estimation should succeed");

        assert!(bandwidth > 0.0, "Silverman bandwidth should be positive");
        assert!(bandwidth < 20.0, "Silverman bandwidth should be reasonable");
    }

    #[test]
    fn test_estimate_bandwidth_scott() {
        let data = make_test_data();
        let bandwidth =
            estimate_bandwidth_scott(&data.view()).expect("Scott estimation should succeed");

        assert!(bandwidth > 0.0, "Scott bandwidth should be positive");
        assert!(bandwidth < 20.0, "Scott bandwidth should be reasonable");
    }

    #[test]
    fn test_estimate_bandwidth_small_sample() {
        let data = array![[1.0, 1.0]];
        let bandwidth = estimate_bandwidth(&data.view(), Some(0.3), None, None)
            .expect("Should work for single sample");
        assert!(bandwidth > 0.0);
        assert_eq!(bandwidth, 1.0);
    }

    #[test]
    fn test_get_bin_seeds() {
        let data = array![
            [1.0, 1.0],
            [1.4, 1.4],
            [1.8, 1.2],
            [2.0, 1.0],
            [2.1, 1.1],
            [0.0, 0.0]
        ];

        let bin_seeds = get_bin_seeds(&data.view(), 1.0, 1);
        assert_eq!(bin_seeds.nrows(), 3);

        let bin_seeds = get_bin_seeds(&data.view(), 1.0, 2);
        assert_eq!(bin_seeds.nrows(), 2);

        let bin_seeds = get_bin_seeds(&data.view(), 0.01, 1);
        assert_eq!(bin_seeds.nrows(), data.nrows());
    }

    #[test]
    fn test_mean_shift_flat_kernel() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            kernel: KernelType::Flat,
            ..Default::default()
        };

        let (centers, labels) =
            mean_shift(&data.view(), options).expect("Mean shift with flat kernel should succeed");

        assert!(centers.nrows() >= 1, "Should find at least 1 cluster");
        assert!(centers.nrows() <= 3, "Should find at most 3 clusters");
        assert!(
            labels.iter().all(|&l| l >= 0),
            "All labels should be non-negative"
        );
    }

    #[test]
    fn test_mean_shift_gaussian_kernel() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            kernel: KernelType::Gaussian,
            ..Default::default()
        };

        let (centers, labels) = mean_shift(&data.view(), options)
            .expect("Mean shift with Gaussian kernel should succeed");

        assert!(centers.nrows() >= 1, "Should find at least 1 cluster");
        assert!(
            labels.iter().all(|&l| l >= 0),
            "All labels should be non-negative"
        );
    }

    #[test]
    fn test_mean_shift_bin_seeding() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            bin_seeding: true,
            ..Default::default()
        };

        let (centers, labels) =
            mean_shift(&data.view(), options).expect("Mean shift with bin seeding should succeed");

        assert!(centers.nrows() >= 1);
        assert!(centers.nrows() <= 3);
        assert!(labels.iter().all(|&l| l >= 0));
    }

    #[test]
    fn test_mean_shift_no_cluster_all() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0],
            [10.0, 10.0]
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            cluster_all: false,
            ..Default::default()
        };

        let (_centers, labels) =
            mean_shift(&data.view(), options).expect("Mean shift should succeed");

        assert!(labels.iter().any(|&l| l == -1));
    }

    #[test]
    fn test_mean_shift_max_iter() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            max_iter: 1,
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model.fit(&data.view()).expect("Should fit");

        assert_eq!(model.n_iter(), 1);
    }

    #[test]
    fn test_mean_shift_predict() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model.fit(&data.view()).expect("Should fit");

        let predicted_labels = model.predict(&data.view()).expect("Predict should succeed");
        assert_eq!(predicted_labels, model.labels().clone());
    }

    #[test]
    fn test_mean_shift_silverman_bandwidth() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: None,
            bandwidth_estimator: BandwidthEstimator::Silverman,
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model
            .fit(&data.view())
            .expect("Should fit with Silverman bandwidth");

        assert!(model.bandwidth_used().is_some());
        assert!(
            model.bandwidth_used().unwrap_or(0.0) > 0.0,
            "Silverman bandwidth should be positive"
        );
    }

    #[test]
    fn test_mean_shift_scott_bandwidth() {
        let data = make_test_data();

        let options = MeanShiftOptions {
            bandwidth: None,
            bandwidth_estimator: BandwidthEstimator::Scott,
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model
            .fit(&data.view())
            .expect("Should fit with Scott bandwidth");

        assert!(model.bandwidth_used().is_some());
        assert!(
            model.bandwidth_used().unwrap_or(0.0) > 0.0,
            "Scott bandwidth should be positive"
        );
    }

    #[test]
    fn test_mean_shift_large_dataset() {
        let mut data = Array2::zeros((20, 2));

        for i in 0..10 {
            data[[i, 0]] = 1.0 + 0.05 * (i as f64);
            data[[i, 1]] = 1.0 + 0.05 * (i as f64);
        }

        for i in 10..20 {
            data[[i, 0]] = 8.0 + 0.05 * ((i - 10) as f64);
            data[[i, 1]] = 8.0 + 0.05 * ((i - 10) as f64);
        }

        let options = MeanShiftOptions {
            bandwidth: Some(1.5),
            bin_seeding: true,
            ..Default::default()
        };

        let (centers, labels) =
            mean_shift(&data.view(), options).expect("Should handle larger dataset");

        assert!(centers.nrows() >= 1);
        assert!(centers.nrows() <= 3);

        let unique_labels: HashSet<_> = labels.iter().cloned().collect();
        assert!(!unique_labels.is_empty());
        assert!(unique_labels.len() <= centers.nrows());
    }
}
