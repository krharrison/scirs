//! Polyharmonic Spline Interpolation
//!
//! Polyharmonic splines are a generalization of thin-plate splines to arbitrary
//! orders and dimensions. They use radial basis functions of the form:
//!
//! - `phi(r) = r^k` for odd `k`
//! - `phi(r) = r^k log(r)` for even `k`
//!
//! The interpolant is:
//!
//! ```text
//! f(x) = sum_i w_i phi(||x - x_i||) + polynomial terms
//! ```
//!
//! The weights and polynomial coefficients are found by solving the augmented
//! linear system:
//!
//! ```text
//! [Phi + lambda*I   P ] [w]   [y]
//! [P^T              0 ] [c] = [0]
//! ```
//!
//! ## Special Cases
//!
//! - **Order 1**: `phi(r) = r` (piecewise linear, no smoothness)
//! - **Order 2**: `phi(r) = r^2 log(r)` (thin-plate spline in 2D)
//! - **Order 3**: `phi(r) = r^3` (cubic polyharmonic)
//! - **Order 4**: `phi(r) = r^4 log(r)` (biharmonic)

use crate::error::{InterpolateError, InterpolateResult};

/// Evaluate the polyharmonic kernel function `phi(r)` for a given order.
///
/// - For odd `k`: `phi(r) = r^k`
/// - For even `k`: `phi(r) = r^k * log(r)` (with `phi(0) = 0` by continuity)
#[inline]
fn phi(r: f64, order: usize) -> f64 {
    if r < f64::EPSILON {
        return 0.0;
    }
    let rk = r.powi(order as i32);
    if order % 2 == 0 {
        rk * r.ln()
    } else {
        rk
    }
}

/// Compute Euclidean distance between two points stored as slices.
#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sq = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let d = ai - bi;
        sq += d * d;
    }
    sq.sqrt()
}

/// A polyharmonic spline interpolator for arbitrary-dimension data.
///
/// Given `n` data points in `d` dimensions with scalar values, the
/// polyharmonic spline builds an interpolant (or smoothing approximation)
/// using radial basis functions augmented with a polynomial of degree `<= order - 1`.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::polyharmonic::PolyharmonicSpline;
///
/// // 2D scattered data
/// let points = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![1.0, 1.0],
/// ];
/// let values = vec![0.0, 1.0, 1.0, 2.0];
///
/// let spline = PolyharmonicSpline::fit(&points, &values, 2, 0.0)
///     .expect("should fit successfully");
///
/// // Evaluate at a query point
/// let result = spline.evaluate(&[0.5, 0.5]).expect("should evaluate");
/// assert!((result - 1.0).abs() < 1e-8);
/// ```
pub struct PolyharmonicSpline {
    /// Data points (n x d), stored row-major.
    points: Vec<Vec<f64>>,
    /// Dimension of the data.
    dim: usize,
    /// Order of the polyharmonic kernel.
    order: usize,
    /// RBF weights (length n).
    weights: Vec<f64>,
    /// Polynomial coefficients. Length = number of polynomial basis terms.
    poly_coeffs: Vec<f64>,
}

/// Compute the number of polynomial basis terms for a polynomial of
/// degree `<= deg` in `d` dimensions (i.e., `C(d + deg, deg)`).
fn poly_term_count(dim: usize, deg: usize) -> usize {
    // C(dim + deg, deg) = (dim+deg)! / (dim! * deg!)
    let mut num = 1usize;
    let mut den = 1usize;
    for i in 1..=deg {
        num *= dim + i;
        den *= i;
    }
    num / den
}

/// Evaluate the polynomial basis at a point `x` for total degree `<= deg`.
/// Returns a vector of length `poly_term_count(dim, deg)`.
fn poly_basis(x: &[f64], deg: usize) -> Vec<f64> {
    let dim = x.len();
    let count = poly_term_count(dim, deg);
    let mut basis = Vec::with_capacity(count);

    // We enumerate monomials in graded lexicographic order.
    // For simplicity we use a recursive approach via multi-indices.
    let mut multi_index = vec![0usize; dim];
    loop {
        let total_deg: usize = multi_index.iter().sum();
        if total_deg <= deg {
            let mut val = 1.0;
            for (i, &exp) in multi_index.iter().enumerate() {
                val *= x[i].powi(exp as i32);
            }
            basis.push(val);
        }

        // Increment multi-index (enumerate all with total degree <= deg)
        if !increment_multi_index(&mut multi_index, deg) {
            break;
        }
    }

    basis
}

/// Increment a multi-index in graded lexicographic order, returning false
/// when all indices have been enumerated.
fn increment_multi_index(idx: &mut [usize], max_total: usize) -> bool {
    let d = idx.len();
    if d == 0 {
        return false;
    }

    // Try to increment the last component
    let last = d - 1;
    idx[last] += 1;
    if idx.iter().sum::<usize>() <= max_total {
        return true;
    }

    // Carry: find rightmost non-zero position that can be shifted
    idx[last] = 0;
    for i in (0..last).rev() {
        idx[i] += 1;
        if idx.iter().sum::<usize>() <= max_total {
            return true;
        }
        idx[i] = 0;
    }
    false
}

/// Solve a dense linear system `A * x = b` using Gaussian elimination with
/// partial pivoting. Returns the solution vector `x`.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> InterpolateResult<Vec<f64>> {
    let n = b.len();
    if a.len() != n {
        return Err(InterpolateError::DimensionMismatch(
            "matrix row count must match right-hand side length".to_string(),
        ));
    }
    for row in a.iter() {
        if row.len() != n {
            return Err(InterpolateError::DimensionMismatch(
                "matrix must be square".to_string(),
            ));
        }
    }

    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n + 1);
        row.extend_from_slice(&a[i]);
        row.push(b[i]);
        aug.push(row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_abs = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let abs_val = aug[row][col].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                max_row = row;
            }
        }

        if max_abs < 1e-15 {
            return Err(InterpolateError::LinalgError(
                "singular or near-singular matrix in polyharmonic spline system".to_string(),
            ));
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-15 {
            return Err(InterpolateError::LinalgError(
                "zero pivot in back substitution".to_string(),
            ));
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

impl PolyharmonicSpline {
    /// Fit a polyharmonic spline to scattered data.
    ///
    /// # Arguments
    ///
    /// * `points` - Data point coordinates, `n` points each of dimension `d`.
    /// * `values` - Function values at the data points (length `n`).
    /// * `order` - Order of the polyharmonic kernel (`>= 1`).
    ///   - `order = 1`: `phi(r) = r`
    ///   - `order = 2`: `phi(r) = r^2 log(r)` (thin-plate spline)
    ///   - `order = 3`: `phi(r) = r^3`
    /// * `smoothing` - Regularization parameter (`>= 0`). When `0`, exact
    ///   interpolation is performed. Larger values produce smoother results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `points` is empty
    /// - `points` and `values` have different lengths
    /// - The linear system is singular
    pub fn fit(
        points: &[Vec<f64>],
        values: &[f64],
        order: usize,
        smoothing: f64,
    ) -> InterpolateResult<Self> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "at least one data point is required for polyharmonic spline".to_string(),
            ));
        }
        if n != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points count ({}) must match values count ({})",
                n,
                values.len()
            )));
        }
        if order == 0 {
            return Err(InterpolateError::InvalidValue(
                "order must be >= 1 for polyharmonic spline".to_string(),
            ));
        }

        let dim = points[0].len();
        if dim == 0 {
            return Err(InterpolateError::InvalidValue(
                "point dimension must be >= 1".to_string(),
            ));
        }
        for (i, pt) in points.iter().enumerate() {
            if pt.len() != dim {
                return Err(InterpolateError::DimensionMismatch(format!(
                    "point {} has dimension {} but expected {}",
                    i,
                    pt.len(),
                    dim
                )));
            }
        }

        // Polynomial degree is order - 1
        let poly_deg = order - 1;
        let m = poly_term_count(dim, poly_deg);
        let total = n + m;

        // Build the augmented system matrix
        // [Phi + lambda*I   P ]
        // [P^T              0 ]
        let mut sys: Vec<Vec<f64>> = vec![vec![0.0; total]; total];
        let mut rhs = vec![0.0; total];

        // Fill Phi block (n x n)
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_distance(&points[i], &points[j]);
                sys[i][j] = phi(r, order);
            }
            // Add regularization to diagonal
            if smoothing > 0.0 {
                sys[i][i] += smoothing;
            }
        }

        // Fill P block (n x m) and P^T block (m x n)
        for i in 0..n {
            let basis = poly_basis(&points[i], poly_deg);
            for (j, &b) in basis.iter().enumerate() {
                sys[i][n + j] = b;
                sys[n + j][i] = b;
            }
        }

        // Right-hand side
        for i in 0..n {
            rhs[i] = values[i];
        }

        // Solve the system
        let solution = solve_linear_system(&sys, &rhs)?;

        let weights = solution[..n].to_vec();
        let poly_coeffs = solution[n..].to_vec();

        Ok(PolyharmonicSpline {
            points: points.to_vec(),
            dim,
            order,
            weights,
            poly_coeffs,
        })
    }

    /// Create a thin-plate spline interpolator (order = 2).
    ///
    /// This is a convenience method equivalent to `fit(points, values, 2, smoothing)`.
    pub fn thin_plate_spline(
        points: &[Vec<f64>],
        values: &[f64],
        smoothing: f64,
    ) -> InterpolateResult<Self> {
        Self::fit(points, values, 2, smoothing)
    }

    /// Evaluate the polyharmonic spline at a single query point.
    ///
    /// # Errors
    ///
    /// Returns an error if the query point dimension does not match the
    /// data point dimension.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "query dimension ({}) must match data dimension ({})",
                query.len(),
                self.dim
            )));
        }

        let mut result = 0.0;

        // RBF contribution
        for (i, pt) in self.points.iter().enumerate() {
            let r = euclidean_distance(query, pt);
            result += self.weights[i] * phi(r, self.order);
        }

        // Polynomial contribution
        let poly_deg = self.order - 1;
        let basis = poly_basis(query, poly_deg);
        for (j, &b) in basis.iter().enumerate() {
            result += self.poly_coeffs[j] * b;
        }

        Ok(result)
    }

    /// Evaluate the polyharmonic spline at multiple query points.
    ///
    /// Returns a vector of interpolated values, one per query point.
    pub fn evaluate_batch(&self, queries: &[Vec<f64>]) -> InterpolateResult<Vec<f64>> {
        let mut results = Vec::with_capacity(queries.len());
        for q in queries {
            results.push(self.evaluate(q)?);
        }
        Ok(results)
    }

    /// Return the order of the polyharmonic kernel.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Return the dimension of the data points.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the number of data points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Return the RBF weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Return the polynomial coefficients.
    pub fn poly_coefficients(&self) -> &[f64] {
        &self.poly_coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tps_interpolates_exactly() {
        // 2D thin-plate spline should interpolate exactly (smoothing = 0)
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let values = vec![0.0, 1.0, 2.0, 3.0, 1.5];

        let spline = PolyharmonicSpline::thin_plate_spline(&points, &values, 0.0)
            .expect("test: fit should succeed");

        for (pt, &expected) in points.iter().zip(values.iter()) {
            let val = spline.evaluate(pt).expect("test: evaluate should succeed");
            assert!(
                (val - expected).abs() < 1e-8,
                "TPS should interpolate exactly: expected {}, got {} at {:?}",
                expected,
                val,
                pt
            );
        }
    }

    #[test]
    fn test_tps_smooth_between_points() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let spline = PolyharmonicSpline::thin_plate_spline(&points, &values, 0.0)
            .expect("test: fit should succeed");

        // Evaluate at center: for additive data f(x,y) = x + y, should be ~1.0
        let center_val = spline
            .evaluate(&[0.5, 0.5])
            .expect("test: evaluate should succeed");
        assert!(
            (center_val - 1.0).abs() < 1e-6,
            "TPS at center should be ~1.0, got {}",
            center_val
        );
    }

    #[test]
    fn test_regularization_produces_smoother_surface() {
        let points = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
        // Add some noise to a linear function
        let values_noisy = vec![0.0, 0.3, 0.45, 0.8, 1.0];

        let spline_exact = PolyharmonicSpline::fit(&points, &values_noisy, 2, 0.0)
            .expect("test: fit should succeed");
        let spline_smooth = PolyharmonicSpline::fit(&points, &values_noisy, 2, 1.0)
            .expect("test: fit should succeed");

        // The smoothed spline should deviate from the exact interpolant
        // at data points (it doesn't interpolate exactly)
        let mut exact_residual = 0.0;
        let mut smooth_residual = 0.0;
        for (pt, &v) in points.iter().zip(values_noisy.iter()) {
            let e = (spline_exact
                .evaluate(pt)
                .expect("test: evaluate should succeed")
                - v)
                .abs();
            let s = (spline_smooth
                .evaluate(pt)
                .expect("test: evaluate should succeed")
                - v)
                .abs();
            exact_residual += e;
            smooth_residual += s;
        }

        // Exact interpolant should have near-zero residual
        assert!(
            exact_residual < 1e-8,
            "exact spline should interpolate exactly"
        );
        // Smooth interpolant should have nonzero residual (smoother)
        assert!(
            smooth_residual > 1e-6,
            "smoothed spline should not interpolate exactly"
        );
    }

    #[test]
    fn test_different_orders_produce_different_results() {
        // Use 1D data to avoid high polynomial term counts.
        // Order k in 1D has poly_deg = k-1, giving k polynomial terms.
        let points: Vec<Vec<f64>> = vec![
            vec![0.0],
            vec![0.2],
            vec![0.4],
            vec![0.6],
            vec![0.8],
            vec![1.0],
        ];
        let values = vec![0.0, 0.3, 0.5, 0.45, 0.8, 1.0];

        let spline_o1 = PolyharmonicSpline::fit(&points, &values, 1, 0.0)
            .expect("test: fit order 1 should succeed");
        let spline_o2 = PolyharmonicSpline::fit(&points, &values, 2, 0.0)
            .expect("test: fit order 2 should succeed");
        let spline_o3 = PolyharmonicSpline::fit(&points, &values, 3, 0.0)
            .expect("test: fit order 3 should succeed");

        let test_pt = vec![0.35];
        let v1 = spline_o1
            .evaluate(&test_pt)
            .expect("test: evaluate should succeed");
        let v2 = spline_o2
            .evaluate(&test_pt)
            .expect("test: evaluate should succeed");
        let v3 = spline_o3
            .evaluate(&test_pt)
            .expect("test: evaluate should succeed");

        // At least two of the three should differ noticeably
        let diff_12 = (v1 - v2).abs();
        let diff_13 = (v1 - v3).abs();
        let diff_23 = (v2 - v3).abs();
        let max_diff = diff_12.max(diff_13).max(diff_23);
        assert!(
            max_diff > 1e-6,
            "different orders should produce different results: v1={}, v2={}, v3={}",
            v1,
            v2,
            v3
        );
    }

    #[test]
    fn test_1d_polyharmonic() {
        // 1D case: should interpolate a simple function
        let points: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let values: Vec<f64> = points.iter().map(|p| p[0] * p[0]).collect();

        let spline =
            PolyharmonicSpline::fit(&points, &values, 3, 0.0).expect("test: fit should succeed");

        // Should interpolate exactly at data points
        for (pt, &expected) in points.iter().zip(values.iter()) {
            let val = spline.evaluate(pt).expect("test: evaluate should succeed");
            assert!(
                (val - expected).abs() < 1e-6,
                "1D polyharmonic should interpolate x^2 exactly at data points"
            );
        }
    }

    #[test]
    fn test_3d_polyharmonic() {
        // 3D case
        let points = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ];
        // f(x,y,z) = x + y + z
        let values: Vec<f64> = points.iter().map(|p| p[0] + p[1] + p[2]).collect();

        let spline =
            PolyharmonicSpline::fit(&points, &values, 1, 0.0).expect("test: fit should succeed");

        for (pt, &expected) in points.iter().zip(values.iter()) {
            let val = spline.evaluate(pt).expect("test: evaluate should succeed");
            assert!(
                (val - expected).abs() < 1e-6,
                "3D polyharmonic should interpolate linear function exactly"
            );
        }
    }

    #[test]
    fn test_batch_evaluation() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let values = vec![0.0, 1.0, 4.0];

        let spline =
            PolyharmonicSpline::fit(&points, &values, 2, 0.0).expect("test: fit should succeed");

        let queries = vec![vec![0.0], vec![1.0], vec![2.0]];
        let results = spline
            .evaluate_batch(&queries)
            .expect("test: batch evaluate should succeed");

        for (res, &expected) in results.iter().zip(values.iter()) {
            assert!(
                (res - expected).abs() < 1e-8,
                "batch evaluation should match single evaluation"
            );
        }
    }

    #[test]
    fn test_error_on_empty_data() {
        let points: Vec<Vec<f64>> = vec![];
        let values: Vec<f64> = vec![];
        let result = PolyharmonicSpline::fit(&points, &values, 2, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_on_dimension_mismatch() {
        // Need enough points for order 2 in 2D (poly_deg=1 => 3 poly terms,
        // so system size = n + 3, need n >= 3 for non-singularity)
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let spline =
            PolyharmonicSpline::fit(&points, &values, 2, 0.0).expect("test: fit should succeed");

        // Query with wrong dimension
        let result = spline.evaluate(&[0.5]);
        assert!(result.is_err());
    }
}
