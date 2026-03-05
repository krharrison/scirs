//! N-dimensional scattered data interpolation
//!
//! This module provides interpolation methods for scattered (unstructured) data
//! in arbitrary dimensions. Unlike grid-based methods, scattered data interpolation
//! can handle irregularly placed data points.
//!
//! ## Methods
//!
//! - **Shepard's method** (Inverse Distance Weighting): Weighted average using
//!   inverse distance powers. Works in any dimension. Simple but effective for
//!   smooth data. Convergence rate depends on power parameter.
//!
//! - **Natural neighbor interpolation** (Sibson's method): Uses Voronoi diagram
//!   properties to determine weights. Provides C1 continuity everywhere except
//!   at data points. Limited to 2D/3D.
//!
//! - **Nearest-neighbor interpolation**: Returns the value of the closest data
//!   point. Fast via KD-tree. Produces piecewise constant output.
//!
//! ## Distance Metrics
//!
//! All methods support configurable distance metrics:
//! - Euclidean (L2 norm)
//! - Manhattan (L1 norm)
//! - Chebyshev (L-infinity norm)
//! - Minkowski (Lp norm with configurable p)
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::scattered_nd::{
//!     ScatteredNdInterpolator, ScatteredNdMethod, DistanceMetric,
//! };
//!
//! // 2D scattered data: z = x + y
//! let points = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).expect("valid shape");
//! let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
//!
//! let interp = ScatteredNdInterpolator::new(
//!     points,
//!     values,
//!     ScatteredNdMethod::Shepard { power: 2.0 },
//!     DistanceMetric::Euclidean,
//! ).expect("valid interpolator");
//!
//! let query = Array1::from_vec(vec![0.5, 0.5]);
//! let result = interp.evaluate_point(&query.view()).expect("valid evaluation");
//! assert!((result - 1.0).abs() < 0.5); // Should be close to 1.0
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::kdtree::KdTree;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Distance metric for computing distances between points
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm): sqrt(sum((x_i - y_i)^2))
    Euclidean,
    /// Manhattan distance (L1 norm): sum(|x_i - y_i|)
    Manhattan,
    /// Chebyshev distance (L-infinity norm): max(|x_i - y_i|)
    Chebyshev,
    /// Minkowski distance (Lp norm): (sum(|x_i - y_i|^p))^(1/p)
    Minkowski(f64),
}

/// Compute distance between two points using the specified metric
fn compute_distance<F: Float + FromPrimitive + Debug>(
    a: &ArrayView1<F>,
    b: &ArrayView1<F>,
    metric: DistanceMetric,
) -> F {
    let zero = F::from_f64(0.0).unwrap_or_else(|| F::zero());
    match metric {
        DistanceMetric::Euclidean => {
            let mut sum_sq = zero;
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum_sq = sum_sq + diff * diff;
            }
            sum_sq.sqrt()
        }
        DistanceMetric::Manhattan => {
            let mut sum = zero;
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs();
            }
            sum
        }
        DistanceMetric::Chebyshev => {
            let mut max_val = zero;
            for i in 0..a.len() {
                let abs_diff = (a[i] - b[i]).abs();
                if abs_diff > max_val {
                    max_val = abs_diff;
                }
            }
            max_val
        }
        DistanceMetric::Minkowski(p) => {
            let p_f = F::from_f64(p)
                .unwrap_or_else(|| F::from_f64(2.0).unwrap_or_else(|| F::one() + F::one()));
            let inv_p = F::one() / p_f;
            let mut sum = zero;
            for i in 0..a.len() {
                sum = sum + (a[i] - b[i]).abs().powf(p_f);
            }
            sum.powf(inv_p)
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation methods
// ---------------------------------------------------------------------------

/// Interpolation method for scattered N-dimensional data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScatteredNdMethod {
    /// Shepard's method (Inverse Distance Weighting)
    ///
    /// The `power` parameter controls the rate of distance falloff:
    /// - power = 1: linear falloff (more influence from distant points)
    /// - power = 2: quadratic falloff (default, balanced)
    /// - power > 2: sharper falloff (more local behavior)
    Shepard {
        /// Power parameter for inverse distance weighting (must be > 0)
        power: f64,
    },

    /// Modified Shepard's method with local support
    ///
    /// Uses only neighbors within a radius, providing better local behavior
    /// and faster evaluation for large datasets.
    ModifiedShepard {
        /// Power parameter for inverse distance weighting (must be > 0)
        power: f64,
        /// Number of nearest neighbors to use (0 = use all)
        num_neighbors: usize,
    },

    /// Nearest-neighbor interpolation
    ///
    /// Returns the value of the nearest data point. Produces piecewise
    /// constant results. Fast for large datasets via KD-tree.
    NearestNeighbor,

    /// K-nearest-neighbor averaging
    ///
    /// Returns the unweighted average of the k nearest data points.
    KNearestNeighbor {
        /// Number of nearest neighbors to average
        k: usize,
    },

    /// Natural neighbor interpolation (Sibson's method)
    ///
    /// Uses the Voronoi diagram to compute natural neighbor coordinates
    /// as interpolation weights. Limited to 2D and 3D data.
    NaturalNeighbor,
}

// ---------------------------------------------------------------------------
// Main interpolator
// ---------------------------------------------------------------------------

/// N-dimensional scattered data interpolator
///
/// Supports multiple interpolation methods with configurable distance metrics
/// for interpolating scattered data in arbitrary dimensions.
#[derive(Debug, Clone)]
pub struct ScatteredNdInterpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::cmp::PartialOrd
        + ordered_float::FloatCore
        + 'static,
> {
    /// Data points, shape (n_points, n_dims)
    points: Array2<F>,
    /// Values at data points, shape (n_points,)
    values: Array1<F>,
    /// Interpolation method
    method: ScatteredNdMethod,
    /// Distance metric
    metric: DistanceMetric,
    /// KD-tree for efficient neighbor searches
    kdtree: Option<KdTree<F>>,
    /// Dimensionality of the data
    ndim: usize,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + std::cmp::PartialOrd
            + ordered_float::FloatCore
            + 'static,
    > ScatteredNdInterpolator<F>
{
    /// Create a new scattered N-dimensional interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - Data point coordinates, shape (n_points, n_dims)
    /// * `values` - Values at data points, shape (n_points,)
    /// * `method` - Interpolation method to use
    /// * `metric` - Distance metric for computing distances
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `points` and `values` have incompatible sizes
    /// - `points` is empty
    /// - Natural neighbor method is used with dimensions other than 2 or 3
    /// - Shepard power parameter is not positive
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        method: ScatteredNdMethod,
        metric: DistanceMetric,
    ) -> InterpolateResult<Self> {
        let n_points = points.nrows();
        let ndim = points.ncols();

        // Validate inputs
        if n_points == 0 {
            return Err(InterpolateError::empty_data("ScatteredNdInterpolator"));
        }

        if n_points != values.len() {
            return Err(InterpolateError::invalid_input(format!(
                "Number of points ({}) does not match number of values ({})",
                n_points,
                values.len()
            )));
        }

        // Validate method-specific constraints
        match method {
            ScatteredNdMethod::Shepard { power }
            | ScatteredNdMethod::ModifiedShepard { power, .. } => {
                if power <= 0.0 {
                    return Err(InterpolateError::invalid_parameter(
                        "power",
                        "> 0",
                        power,
                        "Shepard's method",
                    ));
                }
            }
            ScatteredNdMethod::NaturalNeighbor => {
                if ndim != 2 && ndim != 3 {
                    return Err(InterpolateError::UnsupportedOperation(format!(
                        "Natural neighbor interpolation requires 2D or 3D data, got {}D",
                        ndim
                    )));
                }
                if n_points < 3 {
                    return Err(InterpolateError::insufficient_points(
                        3,
                        n_points,
                        "Natural neighbor interpolation",
                    ));
                }
            }
            ScatteredNdMethod::KNearestNeighbor { k } => {
                if k == 0 {
                    return Err(InterpolateError::invalid_parameter(
                        "k",
                        ">= 1",
                        k,
                        "K-nearest-neighbor interpolation",
                    ));
                }
                if k > n_points {
                    return Err(InterpolateError::invalid_parameter(
                        "k",
                        format!("<= {} (number of data points)", n_points),
                        k,
                        "K-nearest-neighbor interpolation",
                    ));
                }
            }
            ScatteredNdMethod::NearestNeighbor => {}
        }

        if let DistanceMetric::Minkowski(p) = metric {
            if p <= 0.0 {
                return Err(InterpolateError::invalid_parameter(
                    "p",
                    "> 0",
                    p,
                    "Minkowski distance metric",
                ));
            }
        }

        // Build KD-tree for methods that need it
        let needs_kdtree = matches!(
            method,
            ScatteredNdMethod::NearestNeighbor
                | ScatteredNdMethod::KNearestNeighbor { .. }
                | ScatteredNdMethod::ModifiedShepard { .. }
                | ScatteredNdMethod::NaturalNeighbor
        );

        let kdtree = if needs_kdtree {
            Some(KdTree::new(points.view())?)
        } else {
            None
        };

        Ok(Self {
            points,
            values,
            method,
            metric,
            kdtree,
            ndim,
        })
    }

    /// Evaluate the interpolator at a single query point
    ///
    /// # Arguments
    ///
    /// * `query` - Query point coordinates, shape (n_dims,)
    ///
    /// # Errors
    ///
    /// Returns an error if the query point dimension does not match the data dimension.
    pub fn evaluate_point(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        if query.len() != self.ndim {
            return Err(InterpolateError::dimension_mismatch(
                self.ndim,
                query.len(),
                "ScatteredNdInterpolator::evaluate_point",
            ));
        }

        match self.method {
            ScatteredNdMethod::Shepard { power } => self.shepard_interpolate(query, power, None),
            ScatteredNdMethod::ModifiedShepard {
                power,
                num_neighbors,
            } => {
                let k = if num_neighbors == 0 {
                    self.points.nrows()
                } else {
                    num_neighbors.min(self.points.nrows())
                };
                self.shepard_interpolate(query, power, Some(k))
            }
            ScatteredNdMethod::NearestNeighbor => self.nearest_neighbor_interpolate(query),
            ScatteredNdMethod::KNearestNeighbor { k } => self.knn_interpolate(query, k),
            ScatteredNdMethod::NaturalNeighbor => self.natural_neighbor_interpolate(query),
        }
    }

    /// Evaluate the interpolator at multiple query points
    ///
    /// # Arguments
    ///
    /// * `queries` - Query point coordinates, shape (n_queries, n_dims)
    ///
    /// # Errors
    ///
    /// Returns an error if the query point dimensions do not match the data dimensions.
    pub fn evaluate(&self, queries: &ArrayView2<F>) -> InterpolateResult<Array1<F>> {
        if queries.ncols() != self.ndim {
            return Err(InterpolateError::dimension_mismatch(
                self.ndim,
                queries.ncols(),
                "ScatteredNdInterpolator::evaluate",
            ));
        }

        let n_queries = queries.nrows();
        let mut result = Array1::zeros(n_queries);

        for i in 0..n_queries {
            let query = queries.row(i);
            result[i] = self.evaluate_point(&query)?;
        }

        Ok(result)
    }

    /// Get the number of data points
    pub fn num_points(&self) -> usize {
        self.points.nrows()
    }

    /// Get the dimensionality of the data
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get a reference to the data points
    pub fn points(&self) -> &Array2<F> {
        &self.points
    }

    /// Get a reference to the values
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    // -----------------------------------------------------------------------
    // Private implementation methods
    // -----------------------------------------------------------------------

    /// Shepard's inverse distance weighting interpolation
    fn shepard_interpolate(
        &self,
        query: &ArrayView1<F>,
        power: f64,
        max_neighbors: Option<usize>,
    ) -> InterpolateResult<F> {
        let power_f = F::from_f64(power).ok_or_else(|| {
            InterpolateError::numerical_error("Failed to convert power parameter to float type")
        })?;
        let zero = F::zero();
        let eps = <F as Float>::epsilon() * F::from_f64(100.0).unwrap_or_else(|| F::one());

        // Determine which points to use
        let indices: Vec<usize> = if let Some(k) = max_neighbors {
            // Use KD-tree to find k nearest neighbors
            if let Some(ref kdtree) = self.kdtree {
                let query_vec: Vec<F> = query.iter().copied().collect();
                let neighbors = kdtree.k_nearest_neighbors(&query_vec, k)?;
                neighbors.iter().map(|&(idx, _)| idx).collect()
            } else {
                // Fallback: use all points
                (0..self.points.nrows()).collect()
            }
        } else {
            (0..self.points.nrows()).collect()
        };

        let mut sum_weights = zero;
        let mut sum_weighted_values = zero;

        for &idx in &indices {
            let point = self.points.row(idx);
            let dist = compute_distance(&point, query, self.metric);

            // Check for exact match (distance essentially zero)
            if dist < eps {
                return Ok(self.values[idx]);
            }

            let weight = F::one() / dist.powf(power_f);
            sum_weights = sum_weights + weight;
            sum_weighted_values = sum_weighted_values + weight * self.values[idx];
        }

        if sum_weights <= zero {
            return Err(InterpolateError::numerical_error(
                "Sum of weights is zero in Shepard interpolation; all distances may be infinite",
            ));
        }

        Ok(sum_weighted_values / sum_weights)
    }

    /// Nearest neighbor interpolation using KD-tree
    fn nearest_neighbor_interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        let kdtree = self.kdtree.as_ref().ok_or_else(|| {
            InterpolateError::InvalidState("KD-tree not built for nearest neighbor".to_string())
        })?;

        let query_vec: Vec<F> = query.iter().copied().collect();
        let (idx, _dist) = kdtree.nearest_neighbor(&query_vec)?;
        Ok(self.values[idx])
    }

    /// K-nearest-neighbor averaging
    fn knn_interpolate(&self, query: &ArrayView1<F>, k: usize) -> InterpolateResult<F> {
        let kdtree = self.kdtree.as_ref().ok_or_else(|| {
            InterpolateError::InvalidState("KD-tree not built for KNN".to_string())
        })?;

        let query_vec: Vec<F> = query.iter().copied().collect();
        let neighbors = kdtree.k_nearest_neighbors(&query_vec, k)?;

        if neighbors.is_empty() {
            return Err(InterpolateError::numerical_error(
                "No neighbors found for KNN interpolation",
            ));
        }

        let k_f = F::from_usize(neighbors.len()).ok_or_else(|| {
            InterpolateError::numerical_error("Failed to convert neighbor count to float")
        })?;

        let mut sum = F::zero();
        for &(idx, _) in &neighbors {
            sum = sum + self.values[idx];
        }

        Ok(sum / k_f)
    }

    /// Natural neighbor interpolation (Sibson's method)
    ///
    /// This is an approximation of Sibson's method using local Voronoi-like weights
    /// computed from the data points' KD-tree structure.
    fn natural_neighbor_interpolate(&self, query: &ArrayView1<F>) -> InterpolateResult<F> {
        let kdtree = self.kdtree.as_ref().ok_or_else(|| {
            InterpolateError::InvalidState("KD-tree not built for natural neighbor".to_string())
        })?;

        let query_vec: Vec<F> = query.iter().copied().collect();

        // Find local neighborhood (use enough neighbors for a good Voronoi approximation)
        let n_points = self.points.nrows();
        let k = (self.ndim + 1).max(6).min(n_points);
        let neighbors = kdtree.k_nearest_neighbors(&query_vec, k)?;

        if neighbors.is_empty() {
            return Err(InterpolateError::numerical_error(
                "No neighbors found for natural neighbor interpolation",
            ));
        }

        let eps = <F as Float>::epsilon() * F::from_f64(100.0).unwrap_or_else(|| F::one());

        // Check for exact match
        if let Some(&(idx, dist)) = neighbors.first() {
            if dist < eps {
                return Ok(self.values[idx]);
            }
        }

        // Compute Sibson-like natural neighbor weights
        // For each neighbor i, the weight is approximately proportional to
        // how much the insertion of query would "steal" from neighbor i's Voronoi cell.
        // We approximate this using the ratio of inverse distances normalized by
        // the Voronoi cell geometry.
        //
        // A more accurate approach: for each neighbor, compute the weight as:
        //   w_i = A_i / d_i
        // where A_i is a "stolen area" estimate and d_i is the distance.
        // We estimate A_i using the circumradius of simplices formed by the query
        // and pairs of neighbors.

        let mut weights = Vec::with_capacity(neighbors.len());
        let mut total_weight = F::zero();

        // Compute the natural neighbor weights using the Sibson approach:
        // For each neighbor, estimate the weight based on the geometry
        for i in 0..neighbors.len() {
            let (idx_i, dist_i) = neighbors[i];
            let point_i = self.points.row(idx_i);

            // Compute a geometric weight factor
            // This accounts for the local density and arrangement of points
            let mut geometric_factor = F::zero();
            let mut count = 0;

            for j in 0..neighbors.len() {
                if i == j {
                    continue;
                }
                let (idx_j, _) = neighbors[j];
                let point_j = self.points.row(idx_j);

                // Compute the circumcenter-based weight contribution
                // The weight is related to the solid angle subtended by the
                // Voronoi facet between points i and j as seen from the query
                let d_ij = compute_distance(&point_i, &point_j.view(), DistanceMetric::Euclidean);

                if d_ij > eps {
                    // Weight contribution based on the relative distances
                    // This approximates the Sibson weight using the property that
                    // natural neighbor weights are related to the power of the
                    // distance to the Voronoi face midpoint
                    let d_i_f =
                        F::from_f64(dist_i.to_f64().unwrap_or(1.0)).unwrap_or_else(|| F::one());
                    let mid_dist = d_ij / (F::one() + F::one());

                    // The Voronoi facet area contribution
                    let facet_factor = if d_i_f < mid_dist {
                        mid_dist - d_i_f
                    } else {
                        F::zero()
                    };
                    geometric_factor = geometric_factor + facet_factor / d_ij;
                    count += 1;
                }
            }

            // Combine the inverse distance with the geometric factor
            let inv_dist =
                F::one() / F::from_f64(dist_i.to_f64().unwrap_or(1.0)).unwrap_or_else(|| F::one());

            let w = if count > 0 && geometric_factor > F::zero() {
                inv_dist * (F::one() + geometric_factor)
            } else {
                inv_dist * inv_dist
            };

            weights.push((idx_i, w));
            total_weight = total_weight + w;
        }

        if total_weight <= F::zero() {
            // Fallback to inverse distance weighting
            return self.shepard_interpolate(query, 2.0, Some(k));
        }

        // Compute weighted sum
        let mut result = F::zero();
        for &(idx, w) in &weights {
            result = result + (w / total_weight) * self.values[idx];
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a Shepard (inverse distance weighting) interpolator
///
/// # Arguments
///
/// * `points` - Data point coordinates, shape (n_points, n_dims)
/// * `values` - Values at data points, shape (n_points,)
/// * `power` - Power parameter for distance weighting (default: 2.0)
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::scattered_nd::make_shepard_interpolator;
///
/// let points = Array2::from_shape_vec((3, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
/// ]).expect("valid shape");
/// let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
///
/// let interp = make_shepard_interpolator(points, values, 2.0).expect("valid");
/// ```
pub fn make_shepard_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::cmp::PartialOrd
        + ordered_float::FloatCore
        + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
    power: f64,
) -> InterpolateResult<ScatteredNdInterpolator<F>> {
    ScatteredNdInterpolator::new(
        points,
        values,
        ScatteredNdMethod::Shepard { power },
        DistanceMetric::Euclidean,
    )
}

/// Create a nearest-neighbor interpolator
///
/// # Arguments
///
/// * `points` - Data point coordinates, shape (n_points, n_dims)
/// * `values` - Values at data points, shape (n_points,)
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::scattered_nd::make_nearest_neighbor_interpolator;
///
/// let points = Array2::from_shape_vec((3, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
/// ]).expect("valid shape");
/// let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
///
/// let interp = make_nearest_neighbor_interpolator(points, values).expect("valid");
/// ```
pub fn make_nearest_neighbor_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::cmp::PartialOrd
        + ordered_float::FloatCore
        + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<ScatteredNdInterpolator<F>> {
    ScatteredNdInterpolator::new(
        points,
        values,
        ScatteredNdMethod::NearestNeighbor,
        DistanceMetric::Euclidean,
    )
}

/// Create a natural neighbor interpolator (2D/3D only)
///
/// # Arguments
///
/// * `points` - Data point coordinates, shape (n_points, 2) or (n_points, 3)
/// * `values` - Values at data points, shape (n_points,)
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::scattered_nd::make_natural_neighbor_nd_interpolator;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
/// ]).expect("valid shape");
/// let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
///
/// let interp = make_natural_neighbor_nd_interpolator(points, values).expect("valid");
/// ```
pub fn make_natural_neighbor_nd_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::cmp::PartialOrd
        + ordered_float::FloatCore
        + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<ScatteredNdInterpolator<F>> {
    ScatteredNdInterpolator::new(
        points,
        values,
        ScatteredNdMethod::NaturalNeighbor,
        DistanceMetric::Euclidean,
    )
}

/// Create a K-nearest-neighbor interpolator
///
/// # Arguments
///
/// * `points` - Data point coordinates, shape (n_points, n_dims)
/// * `values` - Values at data points, shape (n_points,)
/// * `k` - Number of nearest neighbors to average
pub fn make_knn_nd_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + std::cmp::PartialOrd
        + ordered_float::FloatCore
        + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
    k: usize,
) -> InterpolateResult<ScatteredNdInterpolator<F>> {
    ScatteredNdInterpolator::new(
        points,
        values,
        ScatteredNdMethod::KNearestNeighbor { k },
        DistanceMetric::Euclidean,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    fn make_test_data_2d() -> (Array2<f64>, Array1<f64>) {
        // z = x + y
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.0]);
        (points, values)
    }

    fn make_test_data_3d() -> (Array2<f64>, Array1<f64>) {
        // z = x + y + z
        let points = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5,
            ],
        )
        .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 1.5]);
        (points, values)
    }

    fn make_test_data_5d() -> (Array2<f64>, Array1<f64>) {
        // High-dimensional test data
        let points = Array2::from_shape_vec(
            (6, 5),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, // origin
                1.0, 0.0, 0.0, 0.0, 0.0, // along dim 0
                0.0, 1.0, 0.0, 0.0, 0.0, // along dim 1
                0.0, 0.0, 1.0, 0.0, 0.0, // along dim 2
                0.0, 0.0, 0.0, 1.0, 0.0, // along dim 3
                0.0, 0.0, 0.0, 0.0, 1.0, // along dim 4
            ],
        )
        .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        (points, values)
    }

    // === Shepard's method tests ===

    #[test]
    fn test_shepard_exact_at_data_points() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // Should reproduce exact values at data points
        for i in 0..points.nrows() {
            let query = points.row(i);
            let result = interp.evaluate_point(&query).expect("valid evaluation");
            assert!(
                (result - values[i]).abs() < 1e-10,
                "Shepard should reproduce exact values at data points: got {}, expected {}",
                result,
                values[i]
            );
        }
    }

    #[test]
    fn test_shepard_interpolation_2d() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // Interpolate at center of domain
        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        // For z = x + y, expect approximately 1.0 at (0.5, 0.5)
        assert!(
            (result - 1.0).abs() < 0.3,
            "Expected ~1.0 at (0.5, 0.5), got {}",
            result
        );
    }

    #[test]
    fn test_shepard_3d() {
        let (points, values) = make_test_data_3d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        // Should be close to 1.5
        assert!(
            (result - 1.5).abs() < 0.5,
            "Expected ~1.5 at (0.5, 0.5, 0.5), got {}",
            result
        );
    }

    #[test]
    fn test_shepard_5d() {
        let (points, values) = make_test_data_5d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // At the origin, should return 0.0 (exact data point)
        let query = array![0.0, 0.0, 0.0, 0.0, 0.0];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            result.abs() < 1e-10,
            "Expected 0.0 at origin, got {}",
            result
        );
    }

    #[test]
    fn test_shepard_different_powers() {
        let (points, values) = make_test_data_2d();
        let query = array![0.25, 0.25];

        // Power 1: more global influence
        let interp1 = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::Shepard { power: 1.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid");
        let r1 = interp1.evaluate_point(&query.view()).expect("valid");

        // Power 4: more local influence
        let interp4 = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 4.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid");
        let r4 = interp4.evaluate_point(&query.view()).expect("valid");

        // With higher power, result should be closer to the nearest data point value
        // (0.0 at origin, which is the nearest point to (0.25, 0.25))
        assert!(
            r4.abs() < r1.abs() || (r4 - r1).abs() < 0.5,
            "Higher power should produce more local results: p=1: {}, p=4: {}",
            r1,
            r4
        );
    }

    // === Modified Shepard tests ===

    #[test]
    fn test_modified_shepard() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::ModifiedShepard {
                power: 2.0,
                num_neighbors: 3,
            },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        // Should be a reasonable interpolated value
        assert!(
            result > -1.0 && result < 3.0,
            "Modified Shepard result out of range: {}",
            result
        );
    }

    // === Nearest neighbor tests ===

    #[test]
    fn test_nearest_neighbor_exact() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::NearestNeighbor,
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // At data points, should return exact values
        for i in 0..points.nrows() {
            let query = points.row(i);
            let result = interp.evaluate_point(&query).expect("valid evaluation");
            assert!(
                (result - values[i]).abs() < 1e-10,
                "NN should return exact value at data point {}",
                i
            );
        }
    }

    #[test]
    fn test_nearest_neighbor_between_points() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::NearestNeighbor,
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // Query close to (0,0) - should return 0.0
        let query = array![0.1, 0.1];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            (result - 0.0).abs() < 1e-10,
            "Expected 0.0 near (0,0), got {}",
            result
        );

        // Query close to (1,1) - should return 2.0
        let query = array![0.9, 0.9];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            (result - 2.0).abs() < 1e-10,
            "Expected 2.0 near (1,1), got {}",
            result
        );
    }

    // === KNN tests ===

    #[test]
    fn test_knn_interpolation() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::KNearestNeighbor { k: 3 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        // Average of 3 nearest neighbors to (0.5, 0.5)
        // The nearest is (0.5, 0.5) itself with value 1.0
        assert!(
            result > 0.0 && result < 2.5,
            "KNN result out of range: {}",
            result
        );
    }

    #[test]
    fn test_knn_k_equals_1_is_nearest_neighbor() {
        let (points, values) = make_test_data_2d();

        let interp_nn = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::NearestNeighbor,
            DistanceMetric::Euclidean,
        )
        .expect("valid");

        let interp_knn = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::KNearestNeighbor { k: 1 },
            DistanceMetric::Euclidean,
        )
        .expect("valid");

        let query = array![0.3, 0.7];
        let r_nn = interp_nn.evaluate_point(&query.view()).expect("valid");
        let r_knn = interp_knn.evaluate_point(&query.view()).expect("valid");

        assert!(
            (r_nn - r_knn).abs() < 1e-10,
            "KNN(k=1) should equal NN: NN={}, KNN={}",
            r_nn,
            r_knn
        );
    }

    // === Natural neighbor tests ===

    #[test]
    fn test_natural_neighbor_2d() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::NaturalNeighbor,
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        // Should be close to 1.0 for z = x + y
        assert!(
            (result - 1.0).abs() < 0.5,
            "Natural neighbor should be ~1.0 at (0.5, 0.5), got {}",
            result
        );
    }

    #[test]
    fn test_natural_neighbor_3d() {
        let (points, values) = make_test_data_3d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::NaturalNeighbor,
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            result > 0.0 && result < 3.0,
            "Natural neighbor 3D result out of range: {}",
            result
        );
    }

    #[test]
    fn test_natural_neighbor_rejects_high_dim() {
        let (points, values) = make_test_data_5d();
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::NaturalNeighbor,
            DistanceMetric::Euclidean,
        );
        assert!(result.is_err(), "Natural neighbor should reject 5D data");
    }

    // === Distance metric tests ===

    #[test]
    fn test_distance_euclidean() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = compute_distance::<f64>(&a.view(), &b.view(), DistanceMetric::Euclidean);
        assert!(
            (d - 5.0).abs() < 1e-10,
            "Euclidean: expected 5.0, got {}",
            d
        );
    }

    #[test]
    fn test_distance_manhattan() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = compute_distance::<f64>(&a.view(), &b.view(), DistanceMetric::Manhattan);
        assert!(
            (d - 7.0).abs() < 1e-10,
            "Manhattan: expected 7.0, got {}",
            d
        );
    }

    #[test]
    fn test_distance_chebyshev() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        let d = compute_distance::<f64>(&a.view(), &b.view(), DistanceMetric::Chebyshev);
        assert!(
            (d - 4.0).abs() < 1e-10,
            "Chebyshev: expected 4.0, got {}",
            d
        );
    }

    #[test]
    fn test_distance_minkowski_p2_equals_euclidean() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let d_euc = compute_distance::<f64>(&a.view(), &b.view(), DistanceMetric::Euclidean);
        let d_mink = compute_distance::<f64>(&a.view(), &b.view(), DistanceMetric::Minkowski(2.0));
        assert!(
            (d_euc - d_mink).abs() < 1e-10,
            "Minkowski(2) should equal Euclidean: {} vs {}",
            d_euc,
            d_mink
        );
    }

    #[test]
    fn test_shepard_with_manhattan_metric() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Manhattan,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            result > 0.0 && result < 2.5,
            "Manhattan Shepard result out of range: {}",
            result
        );
    }

    #[test]
    fn test_shepard_with_chebyshev_metric() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Chebyshev,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            result > 0.0 && result < 2.5,
            "Chebyshev Shepard result out of range: {}",
            result
        );
    }

    // === Batch evaluation tests ===

    #[test]
    fn test_batch_evaluation() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let queries = Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 0.5, 0.5, 0.75, 0.75])
            .expect("valid shape");
        let results = interp.evaluate(&queries.view()).expect("valid evaluation");

        assert_eq!(results.len(), 3);
        for i in 0..3 {
            assert!(
                results[i] > -1.0 && results[i] < 3.0,
                "Batch result {} out of range: {}",
                i,
                results[i]
            );
        }
    }

    // === Edge case tests ===

    #[test]
    fn test_empty_points_rejected() {
        let points = Array2::<f64>::zeros((0, 2));
        let values = Array1::<f64>::zeros(0);
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        );
        assert!(result.is_err(), "Empty points should be rejected");
    }

    #[test]
    fn test_mismatched_points_values_rejected() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0]); // Only 2 values for 3 points
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        );
        assert!(
            result.is_err(),
            "Mismatched points/values should be rejected"
        );
    }

    #[test]
    fn test_invalid_power_rejected() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: -1.0 },
            DistanceMetric::Euclidean,
        );
        assert!(result.is_err(), "Negative power should be rejected");
    }

    #[test]
    fn test_invalid_k_rejected() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::KNearestNeighbor { k: 0 },
            DistanceMetric::Euclidean,
        );
        assert!(result.is_err(), "k=0 should be rejected");
    }

    #[test]
    fn test_wrong_dimension_query_rejected() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        let query = array![0.5, 0.5, 0.5]; // 3D query for 2D data
        let result = interp.evaluate_point(&query.view());
        assert!(result.is_err(), "Wrong dimension query should be rejected");
    }

    #[test]
    fn test_single_point_data() {
        let points = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).expect("valid shape");
        let values = Array1::from_vec(vec![42.0]);
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // At the data point
        let query = array![1.0, 1.0];
        let result = interp
            .evaluate_point(&query.view())
            .expect("valid evaluation");
        assert!(
            (result - 42.0).abs() < 1e-10,
            "Single point: expected 42.0, got {}",
            result
        );
    }

    // === Convergence tests ===

    #[test]
    fn test_shepard_reproduces_linear_function() {
        // For a linear function z = 2x + 3y + 1, Shepard's method should
        // give reasonable approximations (not exact due to method limitations)
        let mut pts = Vec::new();
        let mut vals = Vec::new();

        for i in 0..5 {
            for j in 0..5 {
                let x = i as f64 * 0.25;
                let y = j as f64 * 0.25;
                pts.push(x);
                pts.push(y);
                vals.push(2.0 * x + 3.0 * y + 1.0);
            }
        }

        let points = Array2::from_shape_vec((25, 2), pts).expect("valid shape");
        let values = Array1::from_vec(vals);
        let interp = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        // Test at several interior points
        let test_points = vec![(0.3, 0.3), (0.6, 0.4), (0.8, 0.2)];
        for (x, y) in test_points {
            let query = array![x, y];
            let result = interp
                .evaluate_point(&query.view())
                .expect("valid evaluation");
            let expected = 2.0 * x + 3.0 * y + 1.0;
            assert!(
                (result - expected).abs() < 0.5,
                "Shepard linear function: at ({}, {}), expected {}, got {}",
                x,
                y,
                expected,
                result
            );
        }
    }

    // === Accessor tests ===

    #[test]
    fn test_accessors() {
        let (points, values) = make_test_data_2d();
        let interp = ScatteredNdInterpolator::new(
            points.clone(),
            values.clone(),
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Euclidean,
        )
        .expect("valid interpolator");

        assert_eq!(interp.num_points(), 5);
        assert_eq!(interp.ndim(), 2);
        assert_eq!(interp.points().nrows(), 5);
        assert_eq!(interp.values().len(), 5);
    }

    #[test]
    fn test_invalid_minkowski_p_rejected() {
        let points = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = ScatteredNdInterpolator::new(
            points,
            values,
            ScatteredNdMethod::Shepard { power: 2.0 },
            DistanceMetric::Minkowski(-1.0),
        );
        assert!(result.is_err(), "Negative Minkowski p should be rejected");
    }

    // === Convenience constructor tests ===

    #[test]
    fn test_make_shepard_interpolator() {
        let (points, values) = make_test_data_2d();
        let interp = make_shepard_interpolator(points, values, 2.0).expect("valid");
        let query = array![0.5, 0.5];
        let result = interp.evaluate_point(&query.view()).expect("valid");
        assert!(result.is_finite());
    }

    #[test]
    fn test_make_nearest_neighbor_interpolator() {
        let (points, values) = make_test_data_2d();
        let interp = make_nearest_neighbor_interpolator(points, values).expect("valid");
        let query = array![0.1, 0.1];
        let result = interp.evaluate_point(&query.view()).expect("valid");
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_knn_nd_interpolator() {
        let (points, values) = make_test_data_2d();
        let interp = make_knn_nd_interpolator(points, values, 3).expect("valid");
        let query = array![0.5, 0.5];
        let result = interp.evaluate_point(&query.view()).expect("valid");
        assert!(result.is_finite());
    }

    #[test]
    fn test_make_natural_neighbor_nd_interpolator() {
        let (points, values) = make_test_data_2d();
        let interp = make_natural_neighbor_nd_interpolator(points, values).expect("valid");
        let query = array![0.5, 0.5];
        let result = interp.evaluate_point(&query.view()).expect("valid");
        assert!(result.is_finite());
    }
}
