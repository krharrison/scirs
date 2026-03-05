//! Radial Basis Function (RBF) interpolation
//!
//! This module provides a comprehensive RBF interpolation framework for N-dimensional
//! scattered data, with support for polynomial augmentation, regularization (smoothing),
//! and multiple kernel types.
//!
//! ## Mathematical Background
//!
//! The RBF interpolant has the form:
//!
//! ```text
//! s(x) = sum_i  w_i * phi(||x - x_i||) + p(x)
//! ```
//!
//! where `phi` is the radial basis function, `w_i` are the weights, and `p(x)` is
//! an optional polynomial term that improves conditioning for conditionally positive
//! definite kernels.
//!
//! ## Kernel Overview
//!
//! | Kernel | Formula | Type |
//! |--------|---------|------|
//! | `Multiquadric(c)` | sqrt(r² + c²) | CPD order 1 |
//! | `InverseMultiquadric(c)` | 1/sqrt(r² + c²) | SPD |
//! | `Gaussian(eps)` | exp(-r²/eps²) | SPD |
//! | `ThinPlateSpline` | r² ln(r) | CPD order 2 |
//! | `Linear` | r | CPD order 1 |
//! | `Cubic` | r³ | CPD order 2 |
//! | `Quintic` | r⁵ | CPD order 3 |
//!
//! CPD = Conditionally Positive Definite (requires polynomial augmentation)
//! SPD = Strictly Positive Definite

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Kernel enum
// ---------------------------------------------------------------------------

/// Radial basis function kernel.
///
/// Parameterized variants carry their shape parameter inline so that
/// the type captures all configuration needed to evaluate the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// Multiquadric: `sqrt(r² + c²)`.
    ///
    /// Conditionally positive definite of order 1.
    /// Larger `c` gives smoother interpolation.
    Multiquadric(f64),

    /// Inverse multiquadric: `1 / sqrt(r² + c²)`.
    ///
    /// Strictly positive definite. Larger `c` broadens influence.
    InverseMultiquadric(f64),

    /// Gaussian: `exp(-r² / eps²)`.
    ///
    /// Strictly positive definite. Smaller `eps` gives more peaked functions.
    Gaussian(f64),

    /// Thin plate spline: `r² * ln(r)` (2-D convention).
    ///
    /// Conditionally positive definite of order 2.
    /// Minimises the thin-plate bending energy.
    ThinPlateSpline,

    /// Linear: `r`.
    ///
    /// Conditionally positive definite of order 1.
    Linear,

    /// Cubic: `r³`.
    ///
    /// Conditionally positive definite of order 2.
    Cubic,

    /// Quintic: `r⁵`.
    ///
    /// Conditionally positive definite of order 3.
    Quintic,
}

// ---------------------------------------------------------------------------
// Kernel evaluation function
// ---------------------------------------------------------------------------

/// Evaluate a radial basis function kernel at distance `r`.
///
/// # Arguments
///
/// * `r`      - Non-negative Euclidean distance.
/// * `kernel` - Kernel variant to evaluate.
///
/// # Returns
///
/// Scalar kernel value.
///
/// # Errors
///
/// Returns an error only for degenerate parameter values
/// (e.g., `Gaussian(0.0)`).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::rbf::{rbf_kernel, RbfKernel};
///
/// let v = rbf_kernel(1.5, RbfKernel::Gaussian(1.0)).expect("doc example: should succeed");
/// assert!(v > 0.0 && v < 1.0);
///
/// // TPS is zero at r = 0
/// assert_eq!(rbf_kernel(0.0, RbfKernel::ThinPlateSpline).expect("doc example: should succeed"), 0.0);
/// ```
pub fn rbf_kernel(r: f64, kernel: RbfKernel) -> InterpolateResult<f64> {
    match kernel {
        RbfKernel::Multiquadric(c) => Ok((r * r + c * c).sqrt()),
        RbfKernel::InverseMultiquadric(c) => {
            let denom = (r * r + c * c).sqrt();
            if denom < f64::EPSILON {
                return Err(InterpolateError::NumericalError(
                    "InverseMultiquadric denominator is effectively zero".to_string(),
                ));
            }
            Ok(1.0 / denom)
        }
        RbfKernel::Gaussian(eps) => {
            if eps.abs() < f64::EPSILON {
                return Err(InterpolateError::invalid_input(
                    "Gaussian kernel requires eps > 0".to_string(),
                ));
            }
            Ok((-r * r / (eps * eps)).exp())
        }
        RbfKernel::ThinPlateSpline => {
            if r < f64::EPSILON {
                Ok(0.0)
            } else {
                Ok(r * r * r.ln())
            }
        }
        RbfKernel::Linear => Ok(r),
        RbfKernel::Cubic => Ok(r * r * r),
        RbfKernel::Quintic => {
            let r2 = r * r;
            Ok(r2 * r2 * r)
        }
    }
}

/// Evaluate an `RbfKernel` for a generic `InterpolationFloat` type.
///
/// This helper is used internally by generic structs.
fn eval_kernel_generic<F: InterpolationFloat>(r: F, kernel: RbfKernel) -> InterpolateResult<F> {
    // Convert to f64, evaluate, convert back
    let r_f64 = r.to_f64().ok_or_else(|| {
        InterpolateError::ComputationError("float conversion to f64 failed".to_string())
    })?;
    let v_f64 = rbf_kernel(r_f64, kernel)?;
    F::from_f64(v_f64).ok_or_else(|| {
        InterpolateError::ComputationError("float conversion from f64 failed".to_string())
    })
}

// ---------------------------------------------------------------------------
// Polynomial degree for each kernel (governs augmentation)
// ---------------------------------------------------------------------------

/// Return the minimum polynomial degree needed to make a given kernel positive definite.
///
/// Returns `0` for strictly positive definite kernels (no augmentation needed).
fn poly_degree(kernel: RbfKernel) -> usize {
    match kernel {
        RbfKernel::InverseMultiquadric(_) | RbfKernel::Gaussian(_) => 0,
        RbfKernel::Multiquadric(_) | RbfKernel::Linear => 1,
        RbfKernel::ThinPlateSpline | RbfKernel::Cubic => 2,
        RbfKernel::Quintic => 3,
    }
}

/// Number of polynomial basis terms for a given degree and dimension.
///
/// For degree 0: 1 (constant).
/// For degree 1: 1 + d (constant + linear).
/// For degree 2: 1 + d + d*(d+1)/2 (constant + linear + quadratic).
/// Higher degrees fall back to 1 + d (conservative augmentation).
fn poly_terms(degree: usize, dim: usize) -> usize {
    match degree {
        0 => 1,
        1 => 1 + dim,
        2 => 1 + dim + dim * (dim + 1) / 2,
        _ => 1 + dim, // conservative fallback for degree >= 3
    }
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

fn euclidean_dist_rows<F: InterpolationFloat>(
    a: &Array2<F>,
    i: usize,
    b: &Array2<F>,
    j: usize,
) -> InterpolateResult<F> {
    let d = a.ncols();
    if b.ncols() != d {
        return Err(InterpolateError::DimensionMismatch(
            "dimension mismatch in distance computation".to_string(),
        ));
    }
    let mut sq = F::zero();
    for k in 0..d {
        let diff = a[[i, k]] - b[[j, k]];
        sq = sq + diff * diff;
    }
    Ok(sq.sqrt())
}

fn euclidean_dist_point<F: InterpolationFloat>(a: &Array2<F>, i: usize, pt: &[F]) -> F {
    let d = a.ncols().min(pt.len());
    let mut sq = F::zero();
    for k in 0..d {
        let diff = a[[i, k]] - pt[k];
        sq = sq + diff * diff;
    }
    sq.sqrt()
}

// ---------------------------------------------------------------------------
// Polynomial evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluate the polynomial basis vector `p(x)` at a query point.
///
/// For degree 0: `[1]`
/// For degree 1: `[1, x_1, ..., x_d]`
/// For degree 2: `[1, x_1, ..., x_d, x_1^2, x_1*x_2, ..., x_d^2]`
fn poly_basis_at<F: InterpolationFloat>(point: &[F], degree: usize) -> Vec<F> {
    let d = point.len();
    let mut basis = vec![F::one()]; // constant term
    if degree >= 1 {
        for &xi in point.iter() {
            basis.push(xi);
        }
    }
    if degree >= 2 {
        for i in 0..d {
            for j in i..d {
                basis.push(point[i] * point[j]);
            }
        }
    }
    basis
}

/// Fill the polynomial row `p[(row, :)]` in the augmented matrix for point `i`.
fn fill_poly_row<F: InterpolationFloat>(
    p_mat: &mut Array2<F>,
    row: usize,
    points: &Array2<F>,
    point_idx: usize,
    degree: usize,
) {
    let d = points.ncols();
    let pt: Vec<F> = (0..d).map(|k| points[[point_idx, k]]).collect();
    let basis = poly_basis_at(&pt, degree);
    for (col, &b) in basis.iter().enumerate() {
        p_mat[[row, col]] = b;
    }
}

// ---------------------------------------------------------------------------
// Gaussian elimination solver (no external dependency)
// ---------------------------------------------------------------------------

fn solve_linear_system<F: InterpolationFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(InterpolateError::invalid_input(
            "Matrix must be square and match RHS".to_string(),
        ));
    }

    let tiny = F::from_f64(1e-14).unwrap_or(F::epsilon());
    let tiny_pivot = F::from_f64(1e-30).unwrap_or(F::epsilon());

    // Augmented matrix [A | b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot row
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < tiny {
            // Nudge diagonal for near-singular systems
            let reg = F::from_f64(1e-10).unwrap_or(F::epsilon());
            aug[[col, col]] = aug[[col, col]] + reg;
        }

        if max_row != col {
            for k in 0..=n {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[max_row, k]];
                aug[[max_row, k]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.abs() < tiny_pivot {
            return Err(InterpolateError::LinalgError(
                "Singular or near-singular RBF system matrix".to_string(),
            ));
        }

        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for k in col..=n {
                aug[[row, k]] = aug[[row, k]] - factor * aug[[col, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[[i, n]];
        for j in (i + 1)..n {
            s = s - aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < tiny_pivot {
            return Err(InterpolateError::LinalgError(
                "Zero pivot in back substitution".to_string(),
            ));
        }
        x[i] = s / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Auto-epsilon selection
// ---------------------------------------------------------------------------

/// Automatically select a shape parameter based on the average nearest-neighbor
/// distance among the data points.
pub fn auto_epsilon<F: InterpolationFloat>(points: &Array2<F>) -> InterpolateResult<F> {
    let n = points.nrows();
    if n <= 1 {
        return Ok(F::one());
    }
    let mut sum_nn = F::zero();
    for i in 0..n {
        let mut min_d = F::infinity();
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean_dist_rows(points, i, points, j)?;
            if d < min_d {
                min_d = d;
            }
        }
        sum_nn = sum_nn + min_d;
    }
    let n_f = F::from_usize(n).ok_or_else(|| {
        InterpolateError::ComputationError("usize to float conversion failed".to_string())
    })?;
    let avg = sum_nn / n_f;
    if avg < F::epsilon() {
        Ok(F::one())
    } else {
        Ok(avg)
    }
}

// ---------------------------------------------------------------------------
// RbfInterpolator
// ---------------------------------------------------------------------------

/// N-dimensional RBF interpolator with optional polynomial augmentation.
///
/// Solves the system
///
/// ```text
/// [K  P ] [w]   [y]
/// [P' 0 ] [c] = [0]
/// ```
///
/// where `K[i,j] = phi(||x_i - x_j||)`, `P` is the polynomial basis matrix,
/// `w` are the RBF weights, and `c` are the polynomial coefficients.
///
/// # Type parameter
///
/// `F`: floating-point type implementing [`InterpolationFloat`].
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::rbf::{RbfInterpolator, RbfKernel};
///
/// // 2-D scattered data: f(x,y) = x + y
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let values = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
///
/// let interp = RbfInterpolator::new(
///     points.clone(), values, RbfKernel::ThinPlateSpline
/// ).expect("doc example: should succeed");
///
/// let query = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("doc example: should succeed");
/// let result = interp.interpolate(&query).expect("doc example: should succeed");
/// assert!((result[0] - 1.0).abs() < 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct RbfInterpolator<F: InterpolationFloat> {
    /// Training points, shape `(n, d)`.
    centers: Array2<F>,
    /// All weights (RBF + polynomial), length `n + poly_terms`.
    weights: Array1<F>,
    /// Kernel type.
    kernel: RbfKernel,
    /// Polynomial degree used for augmentation.
    degree: usize,
    /// Number of polynomial terms in augmentation.
    n_poly: usize,
    /// Spatial dimension.
    dim: usize,
}

impl<F: InterpolationFloat> RbfInterpolator<F> {
    /// Construct a new `RbfInterpolator`.
    ///
    /// # Arguments
    ///
    /// * `points`  - Training points, shape `(n, d)`.
    /// * `values`  - Function values at training points, length `n`.
    /// * `kernel`  - RBF kernel variant (carries shape parameter inline).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions are inconsistent.
    /// - The linear system is singular.
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        kernel: RbfKernel,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::invalid_input(format!(
                "points has {} rows but values has {} elements",
                n,
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("RbfInterpolator"));
        }

        let degree = poly_degree(kernel);
        let n_poly = poly_terms(degree, d);
        let total = n + n_poly;

        // Build augmented system matrix
        let mut mat = Array2::<F>::zeros((total, total));

        // Upper-left: kernel matrix K
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_dist_rows(&points, i, &points, j)?;
                mat[[i, j]] = eval_kernel_generic(r, kernel)?;
            }
        }

        // Upper-right / Lower-left: polynomial blocks
        if n_poly > 0 {
            let mut p_mat = Array2::<F>::zeros((n, n_poly));
            for i in 0..n {
                fill_poly_row(&mut p_mat, i, &points, i, degree);
            }
            for i in 0..n {
                for j in 0..n_poly {
                    mat[[i, n + j]] = p_mat[[i, j]];
                    mat[[n + j, i]] = p_mat[[i, j]];
                }
            }
        }

        // Build RHS [y; 0]
        let mut rhs = Array1::<F>::zeros(total);
        for i in 0..n {
            rhs[i] = values[i];
        }

        let weights = solve_linear_system(&mat, &rhs)?;

        Ok(Self {
            centers: points,
            weights,
            kernel,
            degree,
            n_poly,
            dim: d,
        })
    }

    /// Evaluate the interpolant at a single point.
    pub fn evaluate_point(&self, point: &[F]) -> InterpolateResult<F> {
        if point.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected {} dimensions, got {}",
                self.dim,
                point.len()
            )));
        }

        let n = self.centers.nrows();
        let mut val = F::zero();

        // RBF sum
        for i in 0..n {
            let r = euclidean_dist_point(&self.centers, i, point);
            let phi = eval_kernel_generic(r, self.kernel)?;
            val = val + self.weights[i] * phi;
        }

        // Polynomial terms
        if self.n_poly > 0 {
            let basis = poly_basis_at(point, self.degree);
            for (j, &b) in basis.iter().enumerate() {
                val = val + self.weights[n + j] * b;
            }
        }

        Ok(val)
    }

    /// Evaluate the interpolant at multiple query points.
    ///
    /// # Arguments
    ///
    /// * `query_points` - Query points, shape `(m, d)`.
    ///
    /// # Returns
    ///
    /// Array of interpolated values, length `m`.
    pub fn interpolate(&self, query_points: &Array2<F>) -> InterpolateResult<Array1<F>> {
        if query_points.ncols() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "expected {} dimensions, got {}",
                self.dim,
                query_points.ncols()
            )));
        }
        let m = query_points.nrows();
        let mut result = Array1::<F>::zeros(m);
        for i in 0..m {
            let pt: Vec<F> = (0..self.dim).map(|k| query_points[[i, k]]).collect();
            result[i] = self.evaluate_point(&pt)?;
        }
        Ok(result)
    }

    /// Return the kernel type.
    pub fn kernel(&self) -> RbfKernel {
        self.kernel
    }

    /// Return the data centers (training points).
    pub fn centers(&self) -> &Array2<F> {
        &self.centers
    }

    /// Return the combined weight vector (RBF weights + polynomial coefficients).
    pub fn weights(&self) -> &Array1<F> {
        &self.weights
    }

    /// Return the spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// RbfSmoothing — regularized RBF for noisy data
// ---------------------------------------------------------------------------

/// Regularized RBF interpolator for noisy data.
///
/// Unlike `RbfInterpolator`, this struct does **not** pass exactly through the
/// data points. Instead, it minimises:
///
/// ```text
/// || K w - y ||² + lambda * w' K w
/// ```
///
/// via the regularized normal equations `(K + lambda * I) w = y`.
///
/// For conditionally positive definite kernels a polynomial augmentation is
/// still included, but the diagonal shift is only applied to the kernel block.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::rbf::{RbfSmoothing, RbfKernel};
///
/// let points = Array2::from_shape_vec((5, 1), vec![
///     0.0_f64, 1.0, 2.0, 3.0, 4.0
/// ]).expect("doc example: should succeed");
/// let values = Array1::from_vec(vec![0.0_f64, 1.1, 3.9, 9.1, 16.2]); // noisy x²
///
/// let smoother = RbfSmoothing::new(
///     points.clone(), values, RbfKernel::ThinPlateSpline, 0.01
/// ).expect("doc example: should succeed");
///
/// let query = Array2::from_shape_vec((1, 1), vec![2.0_f64]).expect("doc example: should succeed");
/// let result = smoother.interpolate(&query).expect("doc example: should succeed");
/// assert!((result[0] - 4.0).abs() < 1.0); // approximate
/// ```
#[derive(Debug, Clone)]
pub struct RbfSmoothing<F: InterpolationFloat> {
    inner: RbfInterpolator<F>,
    lambda: F,
}

impl<F: InterpolationFloat> RbfSmoothing<F> {
    /// Construct a regularized RBF smoother.
    ///
    /// # Arguments
    ///
    /// * `points` - Training points, shape `(n, d)`.
    /// * `values` - (Noisy) function values at training points, length `n`.
    /// * `kernel` - RBF kernel variant.
    /// * `lambda` - Non-negative regularization parameter. `0.0` degenerates to
    ///              exact interpolation.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or the system is singular.
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        kernel: RbfKernel,
        lambda: F,
    ) -> InterpolateResult<Self> {
        if lambda < F::zero() {
            return Err(InterpolateError::invalid_input(
                "regularization parameter lambda must be non-negative".to_string(),
            ));
        }

        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::invalid_input(format!(
                "points has {} rows but values has {} elements",
                n,
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("RbfSmoothing"));
        }

        let degree = poly_degree(kernel);
        let n_poly = poly_terms(degree, d);
        let total = n + n_poly;

        let mut mat = Array2::<F>::zeros((total, total));

        // Kernel block with regularization on diagonal
        for i in 0..n {
            for j in 0..n {
                let r = euclidean_dist_rows(&points, i, &points, j)?;
                mat[[i, j]] = eval_kernel_generic(r, kernel)?;
            }
            mat[[i, i]] = mat[[i, i]] + lambda;
        }

        // Polynomial augmentation
        if n_poly > 0 {
            let mut p_mat = Array2::<F>::zeros((n, n_poly));
            for i in 0..n {
                fill_poly_row(&mut p_mat, i, &points, i, degree);
            }
            for i in 0..n {
                for j in 0..n_poly {
                    mat[[i, n + j]] = p_mat[[i, j]];
                    mat[[n + j, i]] = p_mat[[i, j]];
                }
            }
        }

        let mut rhs = Array1::<F>::zeros(total);
        for i in 0..n {
            rhs[i] = values[i];
        }

        let weights = solve_linear_system(&mat, &rhs)?;

        let inner = RbfInterpolator {
            centers: points,
            weights,
            kernel,
            degree,
            n_poly,
            dim: d,
        };

        Ok(Self { inner, lambda })
    }

    /// Evaluate the smoother at multiple query points.
    pub fn interpolate(&self, query_points: &Array2<F>) -> InterpolateResult<Array1<F>> {
        self.inner.interpolate(query_points)
    }

    /// Evaluate at a single point.
    pub fn evaluate_point(&self, point: &[F]) -> InterpolateResult<F> {
        self.inner.evaluate_point(point)
    }

    /// Return the regularization parameter.
    pub fn lambda(&self) -> F {
        self.lambda
    }

    /// Access the underlying `RbfInterpolator`.
    pub fn inner(&self) -> &RbfInterpolator<F> {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// 1-D convenience function
// ---------------------------------------------------------------------------

/// Convenience function for 1-D RBF interpolation.
///
/// Wraps `RbfInterpolator` for the common case of scalar input/output.
///
/// # Arguments
///
/// * `x_train`  - 1-D training abscissae.
/// * `y_train`  - Function values at training points, same length as `x_train`.
/// * `kernel`   - RBF kernel.
/// * `x_query`  - Query abscissae.
///
/// # Returns
///
/// Interpolated values at `x_query`.
///
/// # Errors
///
/// Returns an error if the inputs are inconsistent or the system cannot be solved.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_interpolate::rbf::{rbf_1d, RbfKernel};
///
/// let x_train = Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]);
/// let y_train = Array1::from_vec(vec![0.0_f64, 1.0, 4.0, 9.0, 16.0]); // x²
///
/// let x_query = Array1::from_vec(vec![0.5_f64, 1.5, 2.5]);
/// let y_query = rbf_1d(
///     &x_train, &y_train, RbfKernel::ThinPlateSpline, &x_query
/// ).expect("doc example: should succeed");
///
/// assert!((y_query[0] - 0.25).abs() < 0.1); // approximate
/// ```
pub fn rbf_1d(
    x_train: &Array1<f64>,
    y_train: &Array1<f64>,
    kernel: RbfKernel,
    x_query: &Array1<f64>,
) -> InterpolateResult<Array1<f64>> {
    let n = x_train.len();
    if y_train.len() != n {
        return Err(InterpolateError::invalid_input(format!(
            "x_train has {} elements but y_train has {}",
            n,
            y_train.len()
        )));
    }
    if n == 0 {
        return Err(InterpolateError::empty_data("rbf_1d"));
    }

    // Reshape x_train into (n, 1) points matrix
    let points = Array2::from_shape_vec(
        (n, 1),
        x_train.iter().copied().collect::<Vec<f64>>(),
    )
    .map_err(|e| InterpolateError::ShapeError(e.to_string()))?;

    let interp = RbfInterpolator::new(points, y_train.clone(), kernel)?;

    let m = x_query.len();
    let query = Array2::from_shape_vec(
        (m, 1),
        x_query.iter().copied().collect::<Vec<f64>>(),
    )
    .map_err(|e| InterpolateError::ShapeError(e.to_string()))?;

    interp.interpolate(&query)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{Array1, Array2};

    // -----------------------------------------------------------------------
    // Helper factories
    // -----------------------------------------------------------------------

    fn make_2d_data() -> (Array2<f64>, Array1<f64>) {
        // f(x, y) = x + y over unit square corners + center
        let points = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .expect("shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.0]);
        (points, values)
    }

    fn make_1d_data() -> (Array2<f64>, Array1<f64>) {
        // f(x) = x²
        let points =
            Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).expect("shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
        (points, values)
    }

    // -----------------------------------------------------------------------
    // rbf_kernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rbf_kernel_tps_zero_at_origin() {
        let v = rbf_kernel(0.0, RbfKernel::ThinPlateSpline).expect("eval");
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_rbf_kernel_tps_positive() {
        let v = rbf_kernel(1.0, RbfKernel::ThinPlateSpline).expect("eval");
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10); // r²ln(r) at r=1 is 0
    }

    #[test]
    fn test_rbf_kernel_tps_at_two() {
        let r = 2.0_f64;
        let v = rbf_kernel(r, RbfKernel::ThinPlateSpline).expect("eval");
        assert_abs_diff_eq!(v, r * r * r.ln(), epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_gaussian_at_origin() {
        let v = rbf_kernel(0.0, RbfKernel::Gaussian(1.0)).expect("eval");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_gaussian_decay() {
        let v1 = rbf_kernel(1.0, RbfKernel::Gaussian(1.0)).expect("eval");
        let v2 = rbf_kernel(2.0, RbfKernel::Gaussian(1.0)).expect("eval");
        assert!(v1 > v2, "Gaussian should decay with distance");
        assert!(v1 > 0.0);
    }

    #[test]
    fn test_rbf_kernel_gaussian_zero_eps_error() {
        assert!(rbf_kernel(1.0, RbfKernel::Gaussian(0.0)).is_err());
    }

    #[test]
    fn test_rbf_kernel_multiquadric_at_origin() {
        let c = 1.5;
        let v = rbf_kernel(0.0, RbfKernel::Multiquadric(c)).expect("eval");
        assert_abs_diff_eq!(v, c, epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_inv_multiquadric_at_origin() {
        let c = 2.0;
        let v = rbf_kernel(0.0, RbfKernel::InverseMultiquadric(c)).expect("eval");
        assert_abs_diff_eq!(v, 1.0 / c, epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_linear() {
        let v = rbf_kernel(3.5, RbfKernel::Linear).expect("eval");
        assert_abs_diff_eq!(v, 3.5, epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_cubic() {
        let v = rbf_kernel(2.0, RbfKernel::Cubic).expect("eval");
        assert_abs_diff_eq!(v, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rbf_kernel_quintic() {
        let v = rbf_kernel(2.0, RbfKernel::Quintic).expect("eval");
        assert_abs_diff_eq!(v, 32.0, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // RbfInterpolator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rbf_tps_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::ThinPlateSpline)
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rbf_gaussian_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::Gaussian(1.0))
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rbf_multiquadric_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::Multiquadric(1.0))
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rbf_inv_multiquadric_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp =
            RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::InverseMultiquadric(1.0))
                .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rbf_cubic_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::Cubic)
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rbf_quintic_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::Quintic)
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rbf_linear_interpolates_at_centers() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::Linear)
            .expect("construction");
        for i in 0..pts.nrows() {
            let pt = vec![pts[[i, 0]], pts[[i, 1]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, vals[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rbf_batch_interpolate() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::ThinPlateSpline)
            .expect("construction");
        let result = interp.interpolate(&pts).expect("batch eval");
        for i in 0..vals.len() {
            assert_abs_diff_eq!(result[i], vals[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_rbf_dimension_mismatch() {
        let (pts, vals) = make_2d_data();
        let interp = RbfInterpolator::new(pts, vals, RbfKernel::ThinPlateSpline)
            .expect("construction");
        let bad_query = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).expect("shape");
        assert!(interp.interpolate(&bad_query).is_err());
    }

    #[test]
    fn test_rbf_length_mismatch_error() {
        let pts = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            .expect("shape");
        let vals = Array1::from_vec(vec![0.0, 1.0]); // wrong length
        assert!(RbfInterpolator::new(pts, vals, RbfKernel::Gaussian(1.0)).is_err());
    }

    #[test]
    fn test_rbf_empty_data_error() {
        let pts = Array2::<f64>::zeros((0, 2));
        let vals = Array1::<f64>::zeros(0);
        assert!(RbfInterpolator::new(pts, vals, RbfKernel::ThinPlateSpline).is_err());
    }

    #[test]
    fn test_rbf_1d_accessor() {
        let (pts, vals) = make_1d_data();
        let interp = RbfInterpolator::new(pts.clone(), vals.clone(), RbfKernel::ThinPlateSpline)
            .expect("construction");
        assert_eq!(interp.dim(), 1);
        assert_eq!(interp.kernel(), RbfKernel::ThinPlateSpline);
        assert_eq!(interp.centers().nrows(), 5);
    }

    #[test]
    fn test_rbf_3d_data() {
        // f(x,y,z) = x + y + z
        let points = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0]);
        let interp = RbfInterpolator::new(points.clone(), values.clone(), RbfKernel::Gaussian(1.0))
            .expect("construction");
        for i in 0..points.nrows() {
            let pt = vec![points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            let v = interp.evaluate_point(&pt).expect("eval");
            assert_abs_diff_eq!(v, values[i], epsilon = 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // RbfSmoothing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smoothing_reduces_exact_fit() {
        let (pts, vals) = make_2d_data();
        let smoother =
            RbfSmoothing::new(pts.clone(), vals.clone(), RbfKernel::ThinPlateSpline, 1e-3)
                .expect("construction");
        let result = smoother.interpolate(&pts).expect("eval");
        // With small lambda, fit should be close (but not necessarily exact)
        for i in 0..vals.len() {
            assert!((result[i] - vals[i]).abs() < 0.5, "index {i}");
        }
    }

    #[test]
    fn test_smoothing_lambda_zero_exact() {
        let (pts, vals) = make_2d_data();
        let smoother =
            RbfSmoothing::new(pts.clone(), vals.clone(), RbfKernel::ThinPlateSpline, 0.0)
                .expect("construction");
        let result = smoother.interpolate(&pts).expect("eval");
        for i in 0..vals.len() {
            assert_abs_diff_eq!(result[i], vals[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_smoothing_negative_lambda_error() {
        let (pts, vals) = make_2d_data();
        assert!(RbfSmoothing::new(pts, vals, RbfKernel::Gaussian(1.0), -1.0).is_err());
    }

    #[test]
    fn test_smoothing_lambda_accessor() {
        let (pts, vals) = make_2d_data();
        let s = RbfSmoothing::new(pts, vals, RbfKernel::ThinPlateSpline, 0.05).expect("ok");
        assert_abs_diff_eq!(s.lambda(), 0.05, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // rbf_1d convenience tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rbf_1d_interpolates_at_training() {
        let x_train = Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]);
        let y_train = Array1::from_vec(vec![0.0_f64, 1.0, 4.0, 9.0, 16.0]);
        let result = rbf_1d(&x_train, &y_train, RbfKernel::ThinPlateSpline, &x_train)
            .expect("rbf_1d");
        for i in 0..x_train.len() {
            assert_abs_diff_eq!(result[i], y_train[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_rbf_1d_between_points_finite() {
        let x_train = Array1::from_vec(vec![0.0_f64, 1.0, 2.0, 3.0, 4.0]);
        let y_train = Array1::from_vec(vec![0.0_f64, 1.0, 4.0, 9.0, 16.0]);
        let x_query = Array1::from_vec(vec![0.5_f64, 1.5, 2.5, 3.5]);
        let result = rbf_1d(&x_train, &y_train, RbfKernel::Gaussian(1.0), &x_query)
            .expect("rbf_1d");
        assert!(result.iter().all(|v| v.is_finite()), "all finite");
    }

    #[test]
    fn test_rbf_1d_length_mismatch_error() {
        let x_train = Array1::from_vec(vec![0.0_f64, 1.0, 2.0]);
        let y_train = Array1::from_vec(vec![0.0_f64, 1.0]); // wrong
        let x_query = Array1::from_vec(vec![0.5_f64]);
        assert!(rbf_1d(&x_train, &y_train, RbfKernel::Gaussian(1.0), &x_query).is_err());
    }

    #[test]
    fn test_rbf_1d_empty_error() {
        let x_train = Array1::<f64>::zeros(0);
        let y_train = Array1::<f64>::zeros(0);
        let x_query = Array1::from_vec(vec![0.5_f64]);
        assert!(rbf_1d(&x_train, &y_train, RbfKernel::Gaussian(1.0), &x_query).is_err());
    }

    // -----------------------------------------------------------------------
    // auto_epsilon
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_epsilon_positive() {
        let pts = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("shape");
        let eps = auto_epsilon::<f64>(&pts).expect("auto_eps");
        assert!(eps > 0.0);
        assert!(eps.is_finite());
    }

    #[test]
    fn test_auto_epsilon_single_point() {
        let pts = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("shape");
        let eps = auto_epsilon::<f64>(&pts).expect("auto_eps");
        assert_abs_diff_eq!(eps, 1.0, epsilon = 1e-12);
    }
}
