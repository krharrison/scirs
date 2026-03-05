//! Shepard's Method (Inverse Distance Weighting) and variants
//!
//! This module provides a comprehensive implementation of Shepard's interpolation
//! family for N-dimensional scattered data, including:
//!
//! - **Global IDW**: classic inverse-distance weighting with power parameter `p`.
//! - **Modified (local) Shepard**: uses a compact-support weight function so that
//!   only points within radius `R` contribute, giving O(k) per query.
//! - **Franke-Little Shepard**: a higher-order compact-support weight that tapers
//!   smoothly to zero at the boundary, virtually eliminating the "bulls-eye"
//!   artefact of global IDW.
//! - **Adaptive radius selection**: set `R` automatically from the `k`-nearest-
//!   neighbour distance to the query point.
//!
//! ## References
//!
//! - Shepard, D. (1968). "A two-dimensional interpolation function for
//!   irregularly-spaced data." *Proc. 23rd ACM National Conference*, 517-524.
//! - Franke, R. and Nielson, G. (1980). "Smooth interpolation of large sets of
//!   scattered data." *Int. J. Numer. Meth. Eng.* 15, 1691-1704.
//! - Renka, R. J. (1988). "Multivariate interpolation of large sets of scattered
//!   data." *ACM TOMS* 14(2), 139-148.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

// ---------------------------------------------------------------------------
// Distance utilities
// ---------------------------------------------------------------------------

/// Compute Euclidean distance between two equal-length slices.
#[inline]
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Free-function API
// ---------------------------------------------------------------------------

/// Classic inverse-distance weighting (global Shepard).
///
/// ```text
/// f(q) = Σ_i  w_i · f_i  /  Σ_i  w_i,     w_i = 1 / ||q - x_i||^power
/// ```
///
/// If the query coincides exactly with a data site, that site's value is
/// returned immediately.
///
/// # Arguments
///
/// * `query`  – Query point (slice of length `d`).
/// * `points` – Data sites, shape `(n, d)`.
/// * `values` – Function values at data sites, length `n`.
/// * `power`  – IDW power parameter `p > 0`. Larger values give sharper
///              local influence; `p = 2` is standard.
///
/// # Errors
///
/// Returns an error if `power <= 0`, if `points`/`values` sizes are
/// inconsistent, or if the input is empty.
pub fn idw(
    query: &[f64],
    points: &Array2<f64>,
    values: &Array1<f64>,
    power: f64,
) -> InterpolateResult<f64> {
    let n = points.nrows();
    let d = points.ncols();

    if power <= 0.0 {
        return Err(InterpolateError::InvalidInput {
            message: format!("IDW power must be > 0, got {power}"),
        });
    }
    if query.len() != d {
        return Err(InterpolateError::DimensionMismatch(format!(
            "Query has {} components but points have {d} dimensions",
            query.len()
        )));
    }
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "IDW requires at least one data point".to_string(),
        ));
    }
    if n != values.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "points has {n} rows but values has {} entries",
            values.len()
        )));
    }

    let mut wsum = 0.0_f64;
    let mut fsum = 0.0_f64;

    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|j| points[[i, j]]).collect();
        let dist = euclidean_dist(query, &row);

        if dist < 1e-14 {
            // Exactly on a data site
            return Ok(values[i]);
        }

        let w = 1.0 / dist.powf(power);
        wsum += w;
        fsum += w * values[i];
    }

    if wsum < 1e-30 {
        return Err(InterpolateError::NumericalError(
            "IDW weight sum is effectively zero — all points too far away".to_string(),
        ));
    }

    Ok(fsum / wsum)
}

/// Modified (local) Shepard interpolation with compact support.
///
/// Uses a weight function with support `[0, R]`:
///
/// ```text
/// w_i(q) = max(0,  (R - d_i) / (R · d_i) )^2
/// ```
///
/// Only data sites within radius `R` of the query contribute.
///
/// # Arguments
///
/// * `query`  – Query point.
/// * `points` – Data sites.
/// * `values` – Function values.
/// * `radius` – Compact-support radius `R > 0`.
///
/// # Errors
///
/// Returns an error if `radius <= 0` or dimensions are inconsistent.
/// Falls back to a nearest-neighbour result if no points are within `radius`.
pub fn modified_shepard(
    query: &[f64],
    points: &Array2<f64>,
    values: &Array1<f64>,
    radius: f64,
) -> InterpolateResult<f64> {
    let n = points.nrows();
    let d = points.ncols();

    if radius <= 0.0 {
        return Err(InterpolateError::InvalidInput {
            message: format!("Modified Shepard radius must be > 0, got {radius}"),
        });
    }
    if query.len() != d {
        return Err(InterpolateError::DimensionMismatch(format!(
            "Query has {} components but points have {d} dimensions",
            query.len()
        )));
    }
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "Modified Shepard requires at least one data point".to_string(),
        ));
    }
    if n != values.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "points has {n} rows but values has {} entries",
            values.len()
        )));
    }

    let mut wsum = 0.0_f64;
    let mut fsum = 0.0_f64;
    let mut nearest_idx = 0usize;
    let mut nearest_dist = f64::INFINITY;

    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|j| points[[i, j]]).collect();
        let dist = euclidean_dist(query, &row);

        if dist < nearest_dist {
            nearest_dist = dist;
            nearest_idx = i;
        }

        if dist < 1e-14 {
            return Ok(values[i]);
        }

        if dist < radius {
            let num = radius - dist;
            let w = (num / (radius * dist)).powi(2);
            wsum += w;
            fsum += w * values[i];
        }
    }

    if wsum < 1e-30 {
        // No points within radius — use nearest neighbour
        return Ok(values[nearest_idx]);
    }

    Ok(fsum / wsum)
}

/// Franke-Little modified Shepard interpolation.
///
/// Uses the smooth compact-support weight
///
/// ```text
/// w_i(q) = max(0, (R² - d_i²) / (R² · d_i²))²   (Franke-Little version)
/// ```
///
/// This fifth-order kernel vanishes with three continuous derivatives at
/// `d = R`, virtually eliminating the "bulls-eye" artefact of plain IDW.
///
/// # Arguments
///
/// * `query`  – Query point.
/// * `points` – Data sites.
/// * `values` – Function values.
/// * `radius` – Compact-support radius `R > 0`.
///
/// # Errors
///
/// Returns an error if `radius <= 0` or dimensions are inconsistent.
pub fn franke_little_shepard(
    query: &[f64],
    points: &Array2<f64>,
    values: &Array1<f64>,
    radius: f64,
) -> InterpolateResult<f64> {
    let n = points.nrows();
    let d = points.ncols();

    if radius <= 0.0 {
        return Err(InterpolateError::InvalidInput {
            message: format!("Franke-Little radius must be > 0, got {radius}"),
        });
    }
    if query.len() != d {
        return Err(InterpolateError::DimensionMismatch(format!(
            "Query has {} components but points have {d} dimensions",
            query.len()
        )));
    }
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "Franke-Little Shepard requires at least one data point".to_string(),
        ));
    }
    if n != values.len() {
        return Err(InterpolateError::DimensionMismatch(format!(
            "points has {n} rows but values has {} entries",
            values.len()
        )));
    }

    let r2 = radius * radius;
    let mut wsum = 0.0_f64;
    let mut fsum = 0.0_f64;
    let mut nearest_idx = 0usize;
    let mut nearest_dist2 = f64::INFINITY;

    for i in 0..n {
        let row: Vec<f64> = (0..d).map(|j| points[[i, j]]).collect();
        let dist2: f64 = query
            .iter()
            .zip(row.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        if dist2 < nearest_dist2 {
            nearest_dist2 = dist2;
            nearest_idx = i;
        }

        if dist2 < 1e-28 {
            return Ok(values[i]);
        }

        if dist2 < r2 {
            // Franke-Little weight: ((r² - d²) / (r² * d²))²
            let numer = r2 - dist2;
            let denom = r2 * dist2;
            let w = (numer / denom).powi(2);
            wsum += w;
            fsum += w * values[i];
        }
    }

    if wsum < 1e-30 {
        return Ok(values[nearest_idx]);
    }

    Ok(fsum / wsum)
}

/// Adaptively determine the compact-support radius for a query point.
///
/// Finds the `k`-th nearest data site and returns that distance multiplied
/// by `scale` (default scale = 1.5 is reasonable).  This ensures the support
/// always covers at least `k` neighbours regardless of local density.
///
/// # Arguments
///
/// * `query`   – The query point (slice of length `d`).
/// * `points`  – Data sites, shape `(n, d)`.
/// * `k`       – Desired minimum number of neighbours in the support ball.
/// * `scale`   – Multiplicative scale applied to the `k`-th distance.
///               Must be `>= 1.0`.
///
/// # Errors
///
/// Returns an error if `k == 0`, `scale < 1.0`, or dimensions are
/// inconsistent.
pub fn find_radius(
    query: &[f64],
    points: &Array2<f64>,
    k: usize,
    scale: f64,
) -> InterpolateResult<f64> {
    let n = points.nrows();
    let d = points.ncols();

    if k == 0 {
        return Err(InterpolateError::InvalidInput {
            message: "k must be at least 1 for adaptive radius selection".to_string(),
        });
    }
    if scale < 1.0 {
        return Err(InterpolateError::InvalidInput {
            message: format!("scale must be >= 1.0, got {scale}"),
        });
    }
    if query.len() != d {
        return Err(InterpolateError::DimensionMismatch(format!(
            "Query has {} components but points have {d} dimensions",
            query.len()
        )));
    }
    if n == 0 {
        return Err(InterpolateError::InsufficientData(
            "At least one data point is required".to_string(),
        ));
    }

    let k_eff = k.min(n);

    // Collect all distances, find k-th smallest
    let mut dists: Vec<f64> = (0..n)
        .map(|i| {
            let row: Vec<f64> = (0..d).map(|j| points[[i, j]]).collect();
            euclidean_dist(query, &row)
        })
        .collect();

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let kth_dist = dists[k_eff - 1];
    let radius = (kth_dist * scale).max(1e-12);
    Ok(radius)
}

// ---------------------------------------------------------------------------
// ShepardInterp struct
// ---------------------------------------------------------------------------

/// Configured Shepard interpolator for repeated evaluations.
///
/// Stores data sites and values once; each call to `evaluate` or
/// `evaluate_point` performs a fresh IDW computation.  For production use
/// with many queries, consider pairing this with a spatial index (e.g., the
/// `KdTree` in `scirs2-interpolate::spatial`).
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::shepard::ShepardInterp;
///
/// let pts = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let vals = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
///
/// let interp = ShepardInterp::new(pts, vals, 2.0).expect("doc example: should succeed");
///
/// let q = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("doc example: should succeed");
/// let result = interp.evaluate(&q).expect("doc example: should succeed");
/// assert!((result[0] - 1.0).abs() < 0.2);
/// ```
#[derive(Debug, Clone)]
pub struct ShepardInterp {
    /// Data sites, shape `(n, d)`.
    pub points: Array2<f64>,
    /// Function values, length `n`.
    pub values: Array1<f64>,
    /// IDW power parameter.
    pub power: f64,
}

impl ShepardInterp {
    /// Create a new `ShepardInterp`.
    ///
    /// # Arguments
    ///
    /// * `points` – Data sites, shape `(n, d)` with `n >= 1`.
    /// * `values` – Function values, length `n`.
    /// * `power`  – IDW exponent `p > 0`.
    ///
    /// # Errors
    ///
    /// Returns an error if `power <= 0`, dimensions are inconsistent, or
    /// the input is empty.
    pub fn new(
        points: Array2<f64>,
        values: Array1<f64>,
        power: f64,
    ) -> InterpolateResult<Self> {
        if power <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("ShepardInterp power must be > 0, got {power}"),
            });
        }
        let n = points.nrows();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "ShepardInterp requires at least one data point".to_string(),
            ));
        }
        if n != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows but values has {} entries",
                values.len()
            )));
        }
        Ok(Self { points, values, power })
    }

    /// Evaluate at multiple query points (rows of `query`).
    ///
    /// # Errors
    ///
    /// Returns an error if the number of columns in `query` does not match
    /// the spatial dimension of the stored data.
    pub fn evaluate(&self, query: &Array2<f64>) -> InterpolateResult<Array1<f64>> {
        let m = query.nrows();
        let d = query.ncols();

        if d != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {d} does not match data dimension {}",
                self.points.ncols()
            )));
        }

        let mut result = Array1::zeros(m);
        for i in 0..m {
            let q_slice: Vec<f64> = (0..d).map(|j| query[[i, j]]).collect();
            result[i] = idw(&q_slice, &self.points, &self.values, self.power)?;
        }
        Ok(result)
    }

    /// Evaluate at a single query point.
    pub fn evaluate_point(&self, query: &ArrayView1<f64>) -> InterpolateResult<f64> {
        let q_slice: Vec<f64> = query.iter().copied().collect();
        idw(&q_slice, &self.points, &self.values, self.power)
    }

    /// Evaluate using the modified (local/compact-support) Shepard method.
    ///
    /// # Arguments
    ///
    /// * `query`  – Query points, shape `(m, d)`.
    /// * `radius` – Compact-support radius.
    pub fn evaluate_modified(
        &self,
        query: &Array2<f64>,
        radius: f64,
    ) -> InterpolateResult<Array1<f64>> {
        let m = query.nrows();
        let d = query.ncols();

        if d != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {d} does not match data dimension {}",
                self.points.ncols()
            )));
        }

        let mut result = Array1::zeros(m);
        for i in 0..m {
            let q_slice: Vec<f64> = (0..d).map(|j| query[[i, j]]).collect();
            result[i] = modified_shepard(&q_slice, &self.points, &self.values, radius)?;
        }
        Ok(result)
    }

    /// Evaluate using the Franke-Little compact-support Shepard method.
    pub fn evaluate_franke_little(
        &self,
        query: &Array2<f64>,
        radius: f64,
    ) -> InterpolateResult<Array1<f64>> {
        let m = query.nrows();
        let d = query.ncols();

        if d != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {d} does not match data dimension {}",
                self.points.ncols()
            )));
        }

        let mut result = Array1::zeros(m);
        for i in 0..m {
            let q_slice: Vec<f64> = (0..d).map(|j| query[[i, j]]).collect();
            result[i] = franke_little_shepard(&q_slice, &self.points, &self.values, radius)?;
        }
        Ok(result)
    }

    /// Compute an adaptive radius for each row of `query` based on
    /// k-nearest-neighbour distance, then evaluate with the modified Shepard
    /// weight.
    ///
    /// # Arguments
    ///
    /// * `query` – Query points, shape `(m, d)`.
    /// * `k`     – Minimum number of neighbours that must fall inside the support.
    /// * `scale` – Multiplicative factor applied to the k-th nearest distance.
    pub fn evaluate_adaptive(
        &self,
        query: &Array2<f64>,
        k: usize,
        scale: f64,
    ) -> InterpolateResult<Array1<f64>> {
        let m = query.nrows();
        let d = query.ncols();

        if d != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimension {d} does not match data dimension {}",
                self.points.ncols()
            )));
        }

        let mut result = Array1::zeros(m);
        for i in 0..m {
            let q_slice: Vec<f64> = (0..d).map(|j| query[[i, j]]).collect();
            let r = find_radius(&q_slice, &self.points, k, scale)?;
            result[i] = modified_shepard(&q_slice, &self.points, &self.values, r)?;
        }
        Ok(result)
    }

    /// Number of data sites.
    pub fn n_points(&self) -> usize {
        self.points.nrows()
    }

    /// Spatial dimension of the data.
    pub fn dim(&self) -> usize {
        self.points.ncols()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a global Shepard (IDW) interpolator with power `p`.
pub fn make_shepard(
    points: Array2<f64>,
    values: Array1<f64>,
    power: f64,
) -> InterpolateResult<ShepardInterp> {
    ShepardInterp::new(points, values, power)
}

/// Create a Shepard interpolator with the standard `p = 2` exponent.
pub fn make_idw2(
    points: Array2<f64>,
    values: Array1<f64>,
) -> InterpolateResult<ShepardInterp> {
    ShepardInterp::new(points, values, 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn sample_2d() -> (Array2<f64>, Array1<f64>) {
        let pts = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("test: should succeed");
        let vals = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
        (pts, vals)
    }

    #[test]
    fn test_idw_exact_at_site() {
        let (pts, vals) = sample_2d();
        for i in 0..pts.nrows() {
            let q = vec![pts[[i, 0]], pts[[i, 1]]];
            let r = idw(&q, &pts, &vals, 2.0).expect("test: should succeed");
            assert!(
                (r - vals[i]).abs() < 1e-10,
                "Exact reproduction failed at site {i}"
            );
        }
    }

    #[test]
    fn test_idw_centre() {
        let (pts, vals) = sample_2d();
        let q = vec![0.5_f64, 0.5_f64];
        let r = idw(&q, &pts, &vals, 2.0).expect("test: should succeed");
        // All four corners equidistant → average = 1.0
        assert!((r - 1.0).abs() < 1e-10, "Expected 1.0, got {r}");
    }

    #[test]
    fn test_modified_shepard_exact() {
        let (pts, vals) = sample_2d();
        for i in 0..pts.nrows() {
            let q = vec![pts[[i, 0]], pts[[i, 1]]];
            let r = modified_shepard(&q, &pts, &vals, 2.0).expect("test: should succeed");
            assert!(
                (r - vals[i]).abs() < 1e-10,
                "Exact reproduction failed for modified Shepard at site {i}"
            );
        }
    }

    #[test]
    fn test_franke_little_exact() {
        let (pts, vals) = sample_2d();
        for i in 0..pts.nrows() {
            let q = vec![pts[[i, 0]], pts[[i, 1]]];
            let r = franke_little_shepard(&q, &pts, &vals, 2.0).expect("test: should succeed");
            assert!(
                (r - vals[i]).abs() < 1e-10,
                "Exact reproduction failed for Franke-Little at site {i}"
            );
        }
    }

    #[test]
    fn test_find_radius_k1() {
        let (pts, _) = sample_2d();
        let q = vec![0.5_f64, 0.5_f64];
        let r = find_radius(&q, &pts, 1, 1.5).expect("test: should succeed");
        // Nearest point is at distance sqrt(0.5) ≈ 0.707; radius ≥ 1.0
        assert!(r > 0.0, "Radius should be positive");
    }

    #[test]
    fn test_shepard_interp_struct() {
        let (pts, vals) = sample_2d();
        let interp = ShepardInterp::new(pts, vals, 2.0).expect("test: should succeed");

        let q = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("test: should succeed");
        let result = interp.evaluate(&q).expect("test: should succeed");
        assert!((result[0] - 1.0).abs() < 1e-10, "Expected 1.0, got {}", result[0]);
    }

    #[test]
    fn test_shepard_adaptive() {
        let (pts, vals) = sample_2d();
        let interp = ShepardInterp::new(pts, vals, 2.0).expect("test: should succeed");

        let q = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("test: should succeed");
        let result = interp.evaluate_adaptive(&q, 2, 2.0).expect("test: should succeed");
        assert!((result[0] - 1.0).abs() < 0.5, "Adaptive result far off: {}", result[0]);
    }

    #[test]
    fn test_idw_power_zero_error() {
        let (pts, vals) = sample_2d();
        let q = vec![0.5_f64, 0.5];
        assert!(idw(&q, &pts, &vals, 0.0).is_err());
        assert!(idw(&q, &pts, &vals, -1.0).is_err());
    }

    #[test]
    fn test_modified_shepard_outside_radius() {
        let (pts, vals) = sample_2d();
        // Tiny radius that excludes all points from the query at (5,5)
        let q = vec![5.0_f64, 5.0];
        // Should fall back to nearest-neighbour without error
        let r = modified_shepard(&q, &pts, &vals, 0.1).expect("test: should succeed");
        // Nearest point is (1,1) → value 2.0
        assert!((r - 2.0).abs() < 1e-10, "Expected NN fallback to 2.0, got {r}");
    }

    #[test]
    fn test_3d_idw() {
        let pts = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
            ],
        )
        .expect("test: should succeed");
        // f(x,y,z) = x + y + z
        let vals = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0, 3.0]);
        let q = vec![0.5_f64, 0.5, 0.5];
        let r = idw(&q, &pts, &vals, 2.0).expect("test: should succeed");
        // Should be close to 1.5
        assert!((r - 1.5).abs() < 0.5, "3D IDW result: {r}");
    }
}
