//! Tensor product grid interpolation
//!
//! This module provides N-dimensional interpolation on structured (tensor product)
//! grids, where data points lie on a rectilinear grid defined by the Cartesian
//! product of 1D coordinate arrays.
//!
//! ## Methods
//!
//! - **Multilinear interpolation**: N-dimensional generalization of bilinear
//!   interpolation. Piecewise linear in each dimension. C0 continuous.
//!
//! - **Tensor product B-spline interpolation**: Uses B-splines along each
//!   dimension for smooth interpolation. Configurable spline degree.
//!   C^(k-1) continuous for degree-k splines.
//!
//! - **Nearest grid point**: Returns the value at the nearest grid point.
//!   Piecewise constant. Fast evaluation.
//!
//! ## Grid types
//!
//! All methods support non-uniform grid spacing. The grid is defined by
//! N one-dimensional coordinate arrays, one per dimension. Grid points
//! must be strictly increasing along each axis.
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_core::ndarray::{Array, Array1, IxDyn};
//! use scirs2_interpolate::tensor_product::{
//!     TensorProductGridInterpolator, TensorProductMethod,
//! };
//!
//! // Create a 2D grid with non-uniform spacing
//! let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 2.0]);
//! let y = Array1::from_vec(vec![0.0, 1.0, 3.0]);
//!
//! // Values: z = x * y
//! let mut values = Array::zeros(IxDyn(&[4, 3]));
//! for i in 0..4 {
//!     for j in 0..3 {
//!         values[[i, j].as_slice()] = x[i] * y[j];
//!     }
//! }
//!
//! let interp = TensorProductGridInterpolator::new(
//!     vec![x, y],
//!     values,
//!     TensorProductMethod::Multilinear,
//! ).expect("valid interpolator");
//!
//! let result = interp.evaluate_point(&[0.75, 2.0]).expect("valid");
//! // At (0.75, 2.0): 0.75 * 2.0 = 1.5
//! assert!((result - 1.5).abs() < 0.01);
//! ```

pub mod bicubic;
pub mod bilinear;
pub mod nd_grid;
pub mod trilinear;

pub use bicubic::BicubicInterp;
pub use bilinear::BilinearInterp;
pub use nd_grid::NdGridInterp;
pub use trilinear::TrilinearInterp;

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array, Array1, ArrayView1, IxDyn};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

// ---------------------------------------------------------------------------
// Interpolation method
// ---------------------------------------------------------------------------

/// Interpolation method for tensor product grids
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorProductMethod {
    /// Nearest grid point interpolation (piecewise constant)
    Nearest,

    /// Multilinear interpolation (N-dimensional extension of bilinear)
    Multilinear,

    /// Tensor product B-spline interpolation with specified degree
    BSpline {
        /// Degree of the B-spline (1 = linear, 3 = cubic)
        degree: usize,
    },
}

/// Boundary handling for tensor product interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryHandling {
    /// Return an error for points outside the grid
    Error,
    /// Clamp to the grid boundary
    Clamp,
    /// Return NaN for points outside the grid
    Nan,
    /// Extrapolate beyond the grid using the boundary cell
    Extrapolate,
}

// ---------------------------------------------------------------------------
// Tensor product grid interpolator
// ---------------------------------------------------------------------------

/// N-dimensional interpolator on a tensor product (rectilinear) grid
///
/// The grid is defined by the Cartesian product of 1D coordinate arrays.
/// Grid spacing may be non-uniform along each axis.
#[derive(Debug, Clone)]
pub struct TensorProductGridInterpolator<F: Float + FromPrimitive + Debug> {
    /// 1D coordinate arrays for each dimension
    axes: Vec<Array1<F>>,
    /// Values on the grid, shape matching the axes lengths
    values: Array<F, IxDyn>,
    /// Interpolation method
    method: TensorProductMethod,
    /// Boundary handling mode
    boundary: BoundaryHandling,
    /// Number of dimensions
    ndim: usize,
    /// Grid shape (length along each axis)
    shape: Vec<usize>,
    /// Precomputed 1D B-spline coefficients per axis (for BSpline method)
    bspline_coeffs: Option<Array<F, IxDyn>>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + Display
            + AddAssign
            + SubAssign
            + MulAssign
            + DivAssign
            + RemAssign
            + scirs2_core::numeric::Zero
            + 'static,
    > TensorProductGridInterpolator<F>
{
    /// Create a new tensor product grid interpolator
    ///
    /// # Arguments
    ///
    /// * `axes` - 1D coordinate arrays, one per dimension. Must be strictly increasing.
    /// * `values` - N-dimensional array of values on the grid.
    /// * `method` - Interpolation method.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `axes` is empty
    /// - Any axis has fewer than 2 points (or fewer than degree+1 for BSpline)
    /// - Axis coordinates are not strictly increasing
    /// - The values array shape does not match the axes lengths
    pub fn new(
        axes: Vec<Array1<F>>,
        values: Array<F, IxDyn>,
        method: TensorProductMethod,
    ) -> InterpolateResult<Self> {
        Self::with_boundary(axes, values, method, BoundaryHandling::Clamp)
    }

    /// Create a new tensor product grid interpolator with boundary handling
    ///
    /// # Arguments
    ///
    /// * `axes` - 1D coordinate arrays, one per dimension.
    /// * `values` - N-dimensional array of values on the grid.
    /// * `method` - Interpolation method.
    /// * `boundary` - How to handle out-of-bound query points.
    pub fn with_boundary(
        axes: Vec<Array1<F>>,
        values: Array<F, IxDyn>,
        method: TensorProductMethod,
        boundary: BoundaryHandling,
    ) -> InterpolateResult<Self> {
        let ndim = axes.len();

        if ndim == 0 {
            return Err(InterpolateError::empty_data(
                "TensorProductGridInterpolator",
            ));
        }

        if ndim != values.ndim() {
            return Err(InterpolateError::dimension_mismatch(
                ndim,
                values.ndim(),
                "TensorProductGridInterpolator: axes count vs values dimensions",
            ));
        }

        let mut shape = Vec::with_capacity(ndim);
        for (d, axis) in axes.iter().enumerate() {
            let n = axis.len();
            if n < 2 {
                return Err(InterpolateError::insufficient_points(
                    2,
                    n,
                    &format!("TensorProductGridInterpolator axis {}", d),
                ));
            }

            // Check strictly increasing
            for i in 1..n {
                if axis[i] <= axis[i - 1] {
                    return Err(InterpolateError::invalid_input(format!(
                        "Axis {} is not strictly increasing at index {}: {} <= {}",
                        d,
                        i,
                        axis[i],
                        axis[i - 1]
                    )));
                }
            }

            // Check shape matches
            if n != values.shape()[d] {
                return Err(InterpolateError::shape_mismatch(
                    format!("{}", n),
                    format!("{}", values.shape()[d]),
                    format!("axis {} vs values dimension {}", d, d),
                ));
            }

            // B-spline degree check
            if let TensorProductMethod::BSpline { degree } = method {
                if n < degree + 1 {
                    return Err(InterpolateError::insufficient_points(
                        degree + 1,
                        n,
                        &format!(
                            "TensorProductGridInterpolator axis {} for degree-{} B-spline",
                            d, degree
                        ),
                    ));
                }
            }

            shape.push(n);
        }

        // For B-spline method, precompute coefficients
        let bspline_coeffs = if let TensorProductMethod::BSpline { degree } = method {
            Some(Self::compute_bspline_coefficients(
                &axes, &values, &shape, ndim, degree,
            )?)
        } else {
            None
        };

        Ok(Self {
            axes,
            values,
            method,
            boundary,
            ndim,
            shape,
            bspline_coeffs,
        })
    }

    /// Evaluate the interpolator at a single point
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates of the query point, one per dimension
    ///
    /// # Errors
    ///
    /// Returns an error if the point dimension does not match the grid dimension,
    /// or if boundary handling is set to Error and the point is outside the grid.
    pub fn evaluate_point(&self, point: &[F]) -> InterpolateResult<F> {
        if point.len() != self.ndim {
            return Err(InterpolateError::dimension_mismatch(
                self.ndim,
                point.len(),
                "TensorProductGridInterpolator::evaluate_point",
            ));
        }

        match self.method {
            TensorProductMethod::Nearest => self.nearest_interpolate(point),
            TensorProductMethod::Multilinear => self.multilinear_interpolate(point),
            TensorProductMethod::BSpline { degree } => self.bspline_interpolate(point, degree),
        }
    }

    /// Evaluate the interpolator at a single point given as an ArrayView
    pub fn evaluate_point_array(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        let pt: Vec<F> = point.iter().copied().collect();
        self.evaluate_point(&pt)
    }

    /// Evaluate the interpolator at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Array of query points, shape (n_queries, n_dims)
    pub fn evaluate_batch(&self, points: &[Vec<F>]) -> InterpolateResult<Vec<F>> {
        let mut results = Vec::with_capacity(points.len());
        for pt in points {
            results.push(self.evaluate_point(pt)?);
        }
        Ok(results)
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the grid shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get a reference to the axes
    pub fn axes(&self) -> &[Array1<F>] {
        &self.axes
    }

    /// Get a reference to the values
    pub fn values(&self) -> &Array<F, IxDyn> {
        &self.values
    }

    // -----------------------------------------------------------------------
    // Private: locate point on grid
    // -----------------------------------------------------------------------

    /// Find the cell index and fractional position for a coordinate along one axis
    /// Returns (cell_index, fraction) where fraction is in [0, 1]
    fn locate_on_axis(&self, dim: usize, x: F) -> InterpolateResult<(usize, F)> {
        let axis = &self.axes[dim];
        let n = axis.len();
        let lo = axis[0];
        let hi = axis[n - 1];

        // Handle boundary
        if x < lo || x > hi {
            match self.boundary {
                BoundaryHandling::Error => {
                    return Err(InterpolateError::OutOfBounds(format!(
                        "Point coordinate {} in dimension {} is outside grid bounds [{}, {}]",
                        x, dim, lo, hi
                    )));
                }
                BoundaryHandling::Nan => {
                    return Ok((0, F::nan()));
                }
                BoundaryHandling::Clamp | BoundaryHandling::Extrapolate => {
                    // For Clamp, we clamp to boundary
                    // For Extrapolate, we still use the boundary cell but allow fraction outside [0,1]
                    if x < lo {
                        if self.boundary == BoundaryHandling::Clamp {
                            return Ok((0, F::zero()));
                        } else {
                            // Extrapolate: compute fraction (will be negative)
                            let h = axis[1] - axis[0];
                            let frac = if h > F::zero() {
                                (x - lo) / h
                            } else {
                                F::zero()
                            };
                            return Ok((0, frac));
                        }
                    } else {
                        if self.boundary == BoundaryHandling::Clamp {
                            return Ok((n - 2, F::one()));
                        } else {
                            let h = axis[n - 1] - axis[n - 2];
                            let frac = if h > F::zero() {
                                (x - axis[n - 2]) / h
                            } else {
                                F::one()
                            };
                            return Ok((n - 2, frac));
                        }
                    }
                }
            }
        }

        // Binary search for the cell containing x
        let mut lo_idx = 0usize;
        let mut hi_idx = n - 1;

        while hi_idx - lo_idx > 1 {
            let mid = (lo_idx + hi_idx) / 2;
            if x < axis[mid] {
                hi_idx = mid;
            } else {
                lo_idx = mid;
            }
        }

        // lo_idx is now the cell index (x is between axis[lo_idx] and axis[hi_idx])
        let cell_lo = axis[lo_idx];
        let cell_hi = axis[hi_idx];
        let h = cell_hi - cell_lo;

        let frac = if h > F::zero() {
            (x - cell_lo) / h
        } else {
            F::zero()
        };

        Ok((lo_idx, frac))
    }

    // -----------------------------------------------------------------------
    // Nearest interpolation
    // -----------------------------------------------------------------------

    fn nearest_interpolate(&self, point: &[F]) -> InterpolateResult<F> {
        let mut idx = Vec::with_capacity(self.ndim);

        for d in 0..self.ndim {
            let (cell, frac) = self.locate_on_axis(d, point[d])?;
            if frac.is_nan() {
                return Ok(F::nan());
            }
            // Pick the nearer grid point
            let half = F::from_f64(0.5).unwrap_or_else(|| F::one() / (F::one() + F::one()));
            if frac <= half {
                idx.push(cell);
            } else {
                idx.push((cell + 1).min(self.shape[d] - 1));
            }
        }

        Ok(self.values[idx.as_slice()])
    }

    // -----------------------------------------------------------------------
    // Multilinear interpolation
    // -----------------------------------------------------------------------

    fn multilinear_interpolate(&self, point: &[F]) -> InterpolateResult<F> {
        let mut cells = Vec::with_capacity(self.ndim);
        let mut fracs = Vec::with_capacity(self.ndim);

        for d in 0..self.ndim {
            let (cell, frac) = self.locate_on_axis(d, point[d])?;
            if frac.is_nan() {
                return Ok(F::nan());
            }
            cells.push(cell);
            fracs.push(frac);
        }

        // Compute the multilinear interpolation by iterating over all 2^ndim vertices
        // of the hypercube defined by the cell
        let n_vertices = 1usize << self.ndim;
        let mut result = F::zero();

        for vertex in 0..n_vertices {
            let mut vertex_idx = Vec::with_capacity(self.ndim);
            let mut weight = F::one();

            for d in 0..self.ndim {
                let use_upper = (vertex >> d) & 1 == 1;
                let idx = cells[d] + if use_upper { 1 } else { 0 };
                // Safety: the cell index guarantees idx and idx+1 are valid
                vertex_idx.push(idx.min(self.shape[d] - 1));

                weight = weight
                    * if use_upper {
                        fracs[d]
                    } else {
                        F::one() - fracs[d]
                    };
            }

            result = result + weight * self.values[vertex_idx.as_slice()];
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Tensor product B-spline interpolation
    // -----------------------------------------------------------------------

    /// Compute the B-spline coefficients by solving the tensor product system
    ///
    /// For each dimension, we solve a 1D B-spline fitting problem along that
    /// axis while keeping all other indices fixed. This is done dimension by
    /// dimension.
    fn compute_bspline_coefficients(
        axes: &[Array1<F>],
        values: &Array<F, IxDyn>,
        shape: &[usize],
        ndim: usize,
        degree: usize,
    ) -> InterpolateResult<Array<F, IxDyn>> {
        // Start with the original values
        let mut coeffs = values.clone();

        // Process each dimension
        for d in 0..ndim {
            let n = shape[d];
            let axis = &axes[d];

            // Create the B-spline basis matrix for this axis
            let knots = Self::create_clamped_knots(axis, degree);
            let basis = Self::compute_bspline_basis_matrix(axis, &knots, degree)?;

            // Solve the linear system along this dimension for each "fiber"
            // A fiber is obtained by fixing all indices except dimension d
            let total_fibers: usize = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != d)
                .map(|(_, &s)| s)
                .product::<usize>()
                .max(1);

            // For each fiber, extract the 1D data, solve, and put back
            let mut multi_idx = vec![0usize; ndim];
            for _fiber in 0..total_fibers {
                // Extract the 1D slice along dimension d
                let mut fiber_vals = Vec::with_capacity(n);
                for k in 0..n {
                    multi_idx[d] = k;
                    fiber_vals.push(coeffs[multi_idx.as_slice()]);
                }

                // Solve the 1D B-spline system: basis * c = fiber_vals
                let solved = Self::solve_bspline_system(&basis, &fiber_vals, n)?;

                // Write back
                for k in 0..n {
                    multi_idx[d] = k;
                    *coeffs.get_mut(multi_idx.as_slice()).ok_or_else(|| {
                        InterpolateError::IndexError(format!("Index {:?} out of bounds", multi_idx))
                    })? = solved[k];
                }

                // Advance the multi-index (skip dimension d)
                Self::advance_multi_index(&mut multi_idx, shape, d);
            }
        }

        Ok(coeffs)
    }

    /// Advance a multi-index by incrementing all dimensions except `skip_dim`
    fn advance_multi_index(idx: &mut [usize], shape: &[usize], skip_dim: usize) {
        for d in 0..idx.len() {
            if d == skip_dim {
                continue;
            }
            idx[d] += 1;
            if idx[d] < shape[d] {
                return;
            }
            idx[d] = 0;
        }
    }

    /// Create clamped knot vector for B-spline interpolation
    ///
    /// For n data points and degree p, the clamped knot vector has:
    /// - (p+1) copies of the first coordinate
    /// - (n-p-1) interior knots (averaging the data points)
    /// - (p+1) copies of the last coordinate
    /// Total: n + p + 1 knots
    fn create_clamped_knots(axis: &Array1<F>, degree: usize) -> Vec<F> {
        let n = axis.len();
        let p = degree;
        let n_knots = n + p + 1;
        let mut knots = Vec::with_capacity(n_knots);

        // (p+1) copies of first value
        for _ in 0..=p {
            knots.push(axis[0]);
        }

        // Interior knots: use averaging of data points (de Boor approach)
        if n > p + 1 {
            for j in 1..(n - p) {
                let mut sum = F::zero();
                for i in j..(j + p) {
                    sum = sum + axis[i];
                }
                let p_f = F::from_usize(p).unwrap_or_else(|| F::one());
                knots.push(sum / p_f);
            }
        }

        // (p+1) copies of last value
        for _ in 0..=p {
            knots.push(axis[n - 1]);
        }

        knots
    }

    /// Compute the B-spline basis matrix: B[i][j] = B_{j,degree}(axis[i])
    fn compute_bspline_basis_matrix(
        axis: &Array1<F>,
        knots: &[F],
        degree: usize,
    ) -> InterpolateResult<Vec<Vec<F>>> {
        let n = axis.len();
        let n_basis = n; // n basis functions for n data points
        let mut matrix = vec![vec![F::zero(); n_basis]; n];

        for i in 0..n {
            let x = axis[i];
            for j in 0..n_basis {
                matrix[i][j] = Self::bspline_basis_robust(j, degree, x, knots, n_basis);
            }
        }

        Ok(matrix)
    }

    /// Evaluate B-spline basis function B_{i,k}(x) using de Boor recursion
    /// with robust handling of the right endpoint
    fn bspline_basis_robust(i: usize, k: usize, x: F, knots: &[F], n_basis: usize) -> F {
        if k == 0 {
            if i + 1 >= knots.len() {
                return F::zero();
            }
            // Standard indicator: [knots[i], knots[i+1])
            if x >= knots[i] && x < knots[i + 1] {
                return F::one();
            }
            // Special handling for last basis function at the right endpoint:
            // The last basis function should be 1 at x = knots.last()
            if i == n_basis - 1 && x == knots[i + 1] {
                return F::one();
            }
            return F::zero();
        }

        let mut result = F::zero();

        // Left term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
        if i + k < knots.len() {
            let denom = knots[i + k] - knots[i];
            if denom > F::zero() {
                let left = Self::bspline_basis_robust(i, k - 1, x, knots, n_basis);
                result = result + (x - knots[i]) / denom * left;
            }
        }

        // Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        if i + k + 1 < knots.len() {
            let denom = knots[i + k + 1] - knots[i + 1];
            if denom > F::zero() {
                let right = Self::bspline_basis_robust(i + 1, k - 1, x, knots, n_basis);
                result = result + (knots[i + k + 1] - x) / denom * right;
            }
        }

        result
    }

    /// Solve a banded-like linear system B * c = f using simple Gaussian elimination
    fn solve_bspline_system(matrix: &[Vec<F>], rhs: &[F], n: usize) -> InterpolateResult<Vec<F>> {
        // Build augmented matrix
        let mut aug: Vec<Vec<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n + 1);
            for j in 0..n {
                row.push(matrix[i][j]);
            }
            row.push(rhs[i]);
            aug.push(row);
        }

        let eps = F::from_f64(1e-14).unwrap_or_else(|| F::epsilon());

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = aug[row][col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < eps {
                return Err(InterpolateError::numerical_error(
                    "Singular B-spline basis matrix; cannot compute coefficients",
                ));
            }

            // Swap rows
            if max_row != col {
                aug.swap(col, max_row);
            }

            // Eliminate
            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] = aug[row][j] - factor * val;
                }
            }
        }

        // Back substitution
        let mut result = vec![F::zero(); n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum = sum - aug[i][j] * result[j];
            }
            let diag = aug[i][i];
            if diag.abs() < eps {
                return Err(InterpolateError::numerical_error(
                    "Zero diagonal in back substitution",
                ));
            }
            result[i] = sum / diag;
        }

        Ok(result)
    }

    /// Evaluate tensor product B-spline at a point using precomputed coefficients
    fn bspline_interpolate(&self, point: &[F], degree: usize) -> InterpolateResult<F> {
        let coeffs = self.bspline_coeffs.as_ref().ok_or_else(|| {
            InterpolateError::InvalidState("B-spline coefficients not computed".to_string())
        })?;

        // For each dimension, compute the B-spline basis values at the query coordinate
        let mut basis_vals: Vec<Vec<(usize, F)>> = Vec::with_capacity(self.ndim);

        for d in 0..self.ndim {
            let axis = &self.axes[d];
            let knots = Self::create_clamped_knots(axis, degree);
            let n = axis.len();

            // Clamp point to grid bounds
            let x =
                match self.boundary {
                    BoundaryHandling::Error => {
                        if point[d] < axis[0] || point[d] > axis[n - 1] {
                            return Err(InterpolateError::OutOfBounds(format!(
                            "Point coordinate {} in dimension {} is outside grid bounds [{}, {}]",
                            point[d], d, axis[0], axis[n - 1]
                        )));
                        }
                        point[d]
                    }
                    BoundaryHandling::Nan => {
                        if point[d] < axis[0] || point[d] > axis[n - 1] {
                            return Ok(F::nan());
                        }
                        point[d]
                    }
                    BoundaryHandling::Clamp => point[d].max(axis[0]).min(axis[n - 1]),
                    BoundaryHandling::Extrapolate => point[d],
                };

            // Compute non-zero basis functions at x
            let mut vals = Vec::new();
            for j in 0..n {
                let b = Self::bspline_basis_robust(j, degree, x, &knots, n);
                if b.abs() > F::epsilon() {
                    vals.push((j, b));
                }
            }

            // If no basis functions are non-zero (edge case), use nearest
            if vals.is_empty() {
                // Find nearest grid point
                let mut nearest = 0;
                let mut min_d = (x - axis[0]).abs();
                for j in 1..n {
                    let dist = (x - axis[j]).abs();
                    if dist < min_d {
                        min_d = dist;
                        nearest = j;
                    }
                }
                vals.push((nearest, F::one()));
            }

            basis_vals.push(vals);
        }

        // Compute the tensor product sum:
        // f(x) = sum_{j1,..,jN} c[j1,..,jN] * B_{j1}(x1) * ... * B_{jN}(xN)
        // Only iterate over combinations where all basis values are non-zero
        self.tensor_product_sum(coeffs, &basis_vals, 0, &mut vec![0usize; self.ndim])
    }

    /// Recursively compute tensor product sum over non-zero basis function indices
    fn tensor_product_sum(
        &self,
        coeffs: &Array<F, IxDyn>,
        basis_vals: &[Vec<(usize, F)>],
        dim: usize,
        idx: &mut Vec<usize>,
    ) -> InterpolateResult<F> {
        if dim == self.ndim {
            // All dimensions have been indexed; get the coefficient
            return Ok(coeffs[idx.as_slice()]);
        }

        let mut sum = F::zero();
        for &(j, b) in &basis_vals[dim] {
            idx[dim] = j;
            let inner = self.tensor_product_sum(coeffs, basis_vals, dim + 1, idx)?;
            sum = sum + b * inner;
        }

        Ok(sum)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a multilinear interpolator on a tensor product grid
///
/// # Arguments
///
/// * `axes` - 1D coordinate arrays for each dimension
/// * `values` - Values on the grid
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array, Array1, IxDyn};
/// use scirs2_interpolate::tensor_product::make_multilinear_interpolator;
///
/// let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
/// let y = Array1::from_vec(vec![0.0, 1.0]);
/// let mut values = Array::zeros(IxDyn(&[3, 2]));
/// for i in 0..3 {
///     for j in 0..2 {
///         values[[i, j].as_slice()] = (i + j) as f64;
///     }
/// }
///
/// let interp = make_multilinear_interpolator(vec![x, y], values).expect("valid");
/// ```
pub fn make_multilinear_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + scirs2_core::numeric::Zero
        + 'static,
>(
    axes: Vec<Array1<F>>,
    values: Array<F, IxDyn>,
) -> InterpolateResult<TensorProductGridInterpolator<F>> {
    TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
}

/// Create a tensor product B-spline interpolator
///
/// # Arguments
///
/// * `axes` - 1D coordinate arrays for each dimension
/// * `values` - Values on the grid
/// * `degree` - B-spline degree (1=linear, 3=cubic)
pub fn make_tensor_bspline_interpolator<
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + scirs2_core::numeric::Zero
        + 'static,
>(
    axes: Vec<Array1<F>>,
    values: Array<F, IxDyn>,
    degree: usize,
) -> InterpolateResult<TensorProductGridInterpolator<F>> {
    TensorProductGridInterpolator::new(axes, values, TensorProductMethod::BSpline { degree })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array, Array1, IxDyn};

    fn make_2d_linear_grid() -> (Vec<Array1<f64>>, Array<f64, IxDyn>) {
        // z = x + 2y on a 4x3 grid
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let mut values = Array::zeros(IxDyn(&[4, 3]));
        for i in 0..4 {
            for j in 0..3 {
                values[[i, j].as_slice()] = x[i] + 2.0 * y[j];
            }
        }
        (vec![x, y], values)
    }

    fn make_2d_nonuniform_grid() -> (Vec<Array1<f64>>, Array<f64, IxDyn>) {
        // Non-uniform spacing: z = x * y
        let x = Array1::from_vec(vec![0.0, 0.5, 1.0, 2.0, 4.0]);
        let y = Array1::from_vec(vec![0.0, 0.1, 1.0, 3.0]);
        let mut values = Array::zeros(IxDyn(&[5, 4]));
        for i in 0..5 {
            for j in 0..4 {
                values[[i, j].as_slice()] = x[i] * y[j];
            }
        }
        (vec![x, y], values)
    }

    fn make_3d_grid() -> (Vec<Array1<f64>>, Array<f64, IxDyn>) {
        // z = x + y + z on a 3x3x3 grid
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let z = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let mut values = Array::zeros(IxDyn(&[3, 3, 3]));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    values[[i, j, k].as_slice()] = x[i] + y[j] + z[k];
                }
            }
        }
        (vec![x, y, z], values)
    }

    // === Multilinear interpolation tests ===

    #[test]
    fn test_multilinear_at_grid_points() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::new(
            axes.clone(),
            values.clone(),
            TensorProductMethod::Multilinear,
        )
        .expect("valid");

        // Test at every grid point
        for i in 0..4 {
            for j in 0..3 {
                let result = interp
                    .evaluate_point(&[axes[0][i], axes[1][j]])
                    .expect("valid");
                let expected = values[[i, j].as_slice()];
                assert!(
                    (result - expected).abs() < 1e-12,
                    "At grid point ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    result
                );
            }
        }
    }

    #[test]
    fn test_multilinear_reproduces_linear_function() {
        let (axes, values) = make_2d_linear_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        // Multilinear interpolation should reproduce linear functions exactly
        let test_points = vec![(0.5, 0.5), (1.5, 1.5), (2.5, 1.0), (0.3, 1.7)];
        for (x, y) in test_points {
            let result = interp.evaluate_point(&[x, y]).expect("valid");
            let expected = x + 2.0 * y;
            assert!(
                (result - expected).abs() < 1e-10,
                "Multilinear at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_multilinear_nonuniform_grid() {
        let (axes, values) = make_2d_nonuniform_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        // Test at a known interior point
        // Between x=0.5 and x=1.0, y=0.1 and y=1.0
        let result = interp.evaluate_point(&[0.75, 0.55]).expect("valid");
        // Bilinear interpolation of x*y at (0.75, 0.55):
        // x fraction: (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // y fraction: (0.55 - 0.1) / (1.0 - 0.1) = 0.5
        // Corners: (0.5,0.1)=0.05, (0.5,1.0)=0.5, (1.0,0.1)=0.1, (1.0,1.0)=1.0
        // Result: 0.25*0.05 + 0.25*0.5 + 0.25*0.1 + 0.25*1.0 = 0.4125
        assert!(
            (result - 0.4125).abs() < 1e-10,
            "Nonuniform bilinear: expected 0.4125, got {}",
            result
        );
    }

    #[test]
    fn test_multilinear_3d() {
        let (axes, values) = make_3d_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        // Linear function should be reproduced exactly
        let result = interp.evaluate_point(&[0.5, 1.5, 0.5]).expect("valid");
        let expected = 0.5 + 1.5 + 0.5;
        assert!(
            (result - expected).abs() < 1e-10,
            "3D multilinear at (0.5, 1.5, 0.5): expected {}, got {}",
            expected,
            result
        );
    }

    // === Nearest interpolation tests ===

    #[test]
    fn test_nearest_at_grid_points() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::new(
            axes.clone(),
            values.clone(),
            TensorProductMethod::Nearest,
        )
        .expect("valid");

        for i in 0..4 {
            for j in 0..3 {
                let result = interp
                    .evaluate_point(&[axes[0][i], axes[1][j]])
                    .expect("valid");
                let expected = values[[i, j].as_slice()];
                assert!(
                    (result - expected).abs() < 1e-12,
                    "Nearest at grid point ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    result
                );
            }
        }
    }

    #[test]
    fn test_nearest_between_points() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Nearest)
            .expect("valid");

        // (0.3, 0.3) is closest to grid point (0, 0) => value = 0+0 = 0
        let result = interp.evaluate_point(&[0.3, 0.3]).expect("valid");
        assert!(
            (result - 0.0).abs() < 1e-10,
            "Nearest at (0.3, 0.3): expected 0.0, got {}",
            result
        );

        // (2.7, 1.7) is closest to grid point (3, 2) => value = 3+4 = 7
        let result = interp.evaluate_point(&[2.7, 1.7]).expect("valid");
        assert!(
            (result - 7.0).abs() < 1e-10,
            "Nearest at (2.7, 1.7): expected 7.0, got {}",
            result
        );
    }

    // === B-spline interpolation tests ===

    #[test]
    fn test_bspline_linear_at_grid_points() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::new(
            axes.clone(),
            values.clone(),
            TensorProductMethod::BSpline { degree: 1 },
        )
        .expect("valid");

        // Degree-1 B-spline should reproduce grid values exactly
        for i in 0..4 {
            for j in 0..3 {
                let result = interp
                    .evaluate_point(&[axes[0][i], axes[1][j]])
                    .expect("valid");
                let expected = values[[i, j].as_slice()];
                assert!(
                    (result - expected).abs() < 1e-8,
                    "BSpline(1) at grid ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    result
                );
            }
        }
    }

    fn make_2d_linear_grid_4x4() -> (Vec<Array1<f64>>, Array<f64, IxDyn>) {
        // z = x + 2y on a 4x4 grid (enough for cubic B-spline)
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let mut values = Array::zeros(IxDyn(&[4, 4]));
        for i in 0..4 {
            for j in 0..4 {
                values[[i, j].as_slice()] = x[i] + 2.0 * y[j];
            }
        }
        (vec![x, y], values)
    }

    #[test]
    fn test_bspline_cubic_at_grid_points() {
        let (axes, values) = make_2d_linear_grid_4x4();
        let interp = TensorProductGridInterpolator::new(
            axes.clone(),
            values.clone(),
            TensorProductMethod::BSpline { degree: 3 },
        )
        .expect("valid");

        // Cubic B-spline should reproduce grid values exactly
        for i in 0..4 {
            for j in 0..4 {
                let result = interp
                    .evaluate_point(&[axes[0][i], axes[1][j]])
                    .expect("valid");
                let expected = values[[i, j].as_slice()];
                assert!(
                    (result - expected).abs() < 1e-6,
                    "BSpline(3) at grid ({}, {}): expected {}, got {}",
                    i,
                    j,
                    expected,
                    result
                );
            }
        }
    }

    #[test]
    fn test_bspline_cubic_interior_points() {
        let (axes, values) = make_2d_linear_grid_4x4();
        let interp = TensorProductGridInterpolator::new(
            axes,
            values,
            TensorProductMethod::BSpline { degree: 3 },
        )
        .expect("valid");

        // Cubic B-spline should also reproduce linear functions well
        let result = interp.evaluate_point(&[1.5, 0.5]).expect("valid");
        let expected = 1.5 + 2.0 * 0.5;
        assert!(
            (result - expected).abs() < 0.5,
            "BSpline(3) at (1.5, 0.5): expected {}, got {}",
            expected,
            result
        );
    }

    // === Boundary handling tests ===

    #[test]
    fn test_boundary_clamp() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::with_boundary(
            axes,
            values,
            TensorProductMethod::Multilinear,
            BoundaryHandling::Clamp,
        )
        .expect("valid");

        // Point outside grid gets clamped
        let result = interp.evaluate_point(&[-1.0, -1.0]).expect("valid");
        // Clamped to (0, 0) => 0 + 0 = 0
        assert!(
            (result - 0.0).abs() < 1e-10,
            "Clamped at (-1,-1): expected 0.0, got {}",
            result
        );

        let result = interp.evaluate_point(&[10.0, 10.0]).expect("valid");
        // Clamped to (3, 2) => 3 + 4 = 7
        assert!(
            (result - 7.0).abs() < 1e-10,
            "Clamped at (10,10): expected 7.0, got {}",
            result
        );
    }

    #[test]
    fn test_boundary_error() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::with_boundary(
            axes,
            values,
            TensorProductMethod::Multilinear,
            BoundaryHandling::Error,
        )
        .expect("valid");

        let result = interp.evaluate_point(&[-1.0, 0.5]);
        assert!(result.is_err(), "Should error for out-of-bounds point");
    }

    #[test]
    fn test_boundary_nan() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::with_boundary(
            axes,
            values,
            TensorProductMethod::Multilinear,
            BoundaryHandling::Nan,
        )
        .expect("valid");

        let result = interp.evaluate_point(&[-1.0, 0.5]).expect("valid");
        assert!(result.is_nan(), "Should return NaN for out-of-bounds point");
    }

    #[test]
    fn test_boundary_extrapolate() {
        let (axes, values) = make_2d_linear_grid();
        let interp = TensorProductGridInterpolator::with_boundary(
            axes,
            values,
            TensorProductMethod::Multilinear,
            BoundaryHandling::Extrapolate,
        )
        .expect("valid");

        // For a linear function, extrapolation should give the correct value
        let result = interp.evaluate_point(&[-0.5, 0.5]).expect("valid");
        // z = x + 2y at (-0.5, 0.5) = -0.5 + 1.0 = 0.5
        assert!(
            (result - 0.5).abs() < 1e-10,
            "Extrapolated at (-0.5, 0.5): expected 0.5, got {}",
            result
        );
    }

    // === Batch evaluation tests ===

    #[test]
    fn test_batch_evaluation() {
        let (axes, values) = make_2d_linear_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        let points = vec![vec![0.5, 0.5], vec![1.5, 1.0], vec![2.0, 1.5]];
        let results = interp.evaluate_batch(&points).expect("valid");

        assert_eq!(results.len(), 3);
        assert!((results[0] - (0.5 + 1.0)).abs() < 1e-10);
        assert!((results[1] - (1.5 + 2.0)).abs() < 1e-10);
        assert!((results[2] - (2.0 + 3.0)).abs() < 1e-10);
    }

    // === Edge case tests ===

    #[test]
    fn test_empty_axes_rejected() {
        let axes: Vec<Array1<f64>> = vec![];
        let values = Array::zeros(IxDyn(&[]));
        let result =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear);
        assert!(result.is_err(), "Empty axes should be rejected");
    }

    #[test]
    fn test_too_few_points_rejected() {
        let x = Array1::from_vec(vec![0.0]); // Only 1 point
        let values = Array::zeros(IxDyn(&[1]));
        let result =
            TensorProductGridInterpolator::new(vec![x], values, TensorProductMethod::Multilinear);
        assert!(result.is_err(), "Single-point axis should be rejected");
    }

    #[test]
    fn test_nonsorted_axis_rejected() {
        let x = Array1::from_vec(vec![0.0, 2.0, 1.0]); // Not sorted
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let values = Array::zeros(IxDyn(&[3, 2]));
        let result = TensorProductGridInterpolator::new(
            vec![x, y],
            values,
            TensorProductMethod::Multilinear,
        );
        assert!(result.is_err(), "Non-sorted axis should be rejected");
    }

    #[test]
    fn test_shape_mismatch_rejected() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![0.0, 1.0]);
        let values = Array::zeros(IxDyn(&[3, 3])); // Wrong shape: should be (3, 2)
        let result = TensorProductGridInterpolator::new(
            vec![x, y],
            values,
            TensorProductMethod::Multilinear,
        );
        assert!(result.is_err(), "Shape mismatch should be rejected");
    }

    #[test]
    fn test_wrong_dimension_query_rejected() {
        let (axes, values) = make_2d_linear_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        let result = interp.evaluate_point(&[1.0]); // 1D query for 2D grid
        assert!(result.is_err(), "Wrong dimension query should be rejected");
    }

    // === Accessor tests ===

    #[test]
    fn test_accessors() {
        let (axes, values) = make_2d_linear_grid();
        let interp =
            TensorProductGridInterpolator::new(axes, values, TensorProductMethod::Multilinear)
                .expect("valid");

        assert_eq!(interp.ndim(), 2);
        assert_eq!(interp.shape(), &[4, 3]);
        assert_eq!(interp.axes().len(), 2);
    }

    // === Convenience constructor tests ===

    #[test]
    fn test_make_multilinear_interpolator() {
        let (axes, values) = make_2d_linear_grid();
        let interp = make_multilinear_interpolator(axes, values).expect("valid");
        let result = interp.evaluate_point(&[1.0, 1.0]).expect("valid");
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_make_tensor_bspline_interpolator() {
        let (axes, values) = make_2d_linear_grid();
        let interp = make_tensor_bspline_interpolator(axes, values, 1).expect("valid");
        let result = interp.evaluate_point(&[1.0, 1.0]).expect("valid");
        assert!(
            (result - 3.0).abs() < 1e-6,
            "BSpline at (1,1): expected 3.0, got {}",
            result
        );
    }

    // === Convergence tests ===

    #[test]
    fn test_multilinear_convergence_quadratic() {
        // For f(x,y) = x^2 + y^2 (not linear), multilinear interpolation
        // should converge as the grid is refined.
        // Use an off-grid test point to avoid zero error from hitting a grid node
        let test_point = [0.37_f64, 0.63];
        let exact_value = 0.37 * 0.37 + 0.63 * 0.63;

        let mut errors = Vec::new();
        for &n in &[5, 10, 20, 40] {
            let x = Array1::linspace(0.0, 1.0, n);
            let y = Array1::linspace(0.0, 1.0, n);
            let mut values = Array::zeros(IxDyn(&[n, n]));
            for i in 0..n {
                for j in 0..n {
                    values[[i, j].as_slice()] = x[i] * x[i] + y[j] * y[j];
                }
            }

            let interp = TensorProductGridInterpolator::new(
                vec![x, y],
                values,
                TensorProductMethod::Multilinear,
            )
            .expect("valid");

            let result = interp.evaluate_point(&test_point).expect("valid");
            let error = (result - exact_value).abs();
            errors.push(error);
        }

        // Overall error should decrease with refinement
        assert!(
            errors[errors.len() - 1] < errors[0],
            "Error should decrease: first={}, last={}",
            errors[0],
            errors[errors.len() - 1]
        );

        assert!(
            errors[errors.len() - 1] < 0.01,
            "Multilinear should converge to the exact value: final error = {}",
            errors[errors.len() - 1]
        );
    }

    // === 1D test ===

    #[test]
    fn test_1d_multilinear() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
        let mut values = Array::zeros(IxDyn(&[4]));
        for i in 0..4 {
            values[[i].as_slice()] = (i as f64) * (i as f64); // x^2
        }

        let interp =
            TensorProductGridInterpolator::new(vec![x], values, TensorProductMethod::Multilinear)
                .expect("valid");

        // At x=0.5: linear interp between 0 and 1 = 0.5
        let result = interp.evaluate_point(&[0.5]).expect("valid");
        assert!(
            (result - 0.5).abs() < 1e-10,
            "1D multilinear at 0.5: expected 0.5, got {}",
            result
        );
    }

    // === BSpline degree check ===

    #[test]
    fn test_bspline_insufficient_points_for_degree() {
        // Degree 3 needs at least 4 points per axis
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0]); // Only 3 points
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let values = Array::zeros(IxDyn(&[3, 3]));
        let result = TensorProductGridInterpolator::new(
            vec![x, y],
            values,
            TensorProductMethod::BSpline { degree: 3 },
        );
        assert!(
            result.is_err(),
            "Should reject degree 3 with only 3 points per axis"
        );
    }
}
