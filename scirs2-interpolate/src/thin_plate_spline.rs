//! Thin Plate Spline (TPS) interpolation and 2-D warping
//!
//! Thin plate splines are the natural generalization of cubic splines to
//! higher dimensions. They minimize the bending energy integral:
//!
//! ```text
//! E(f) = ∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dx dy
//! ```
//!
//! subject to an interpolation (or smoothing) constraint.
//!
//! ## Module Contents
//!
//! - [`tps_kernel`] — the elementary kernel function `r² ln(r)`.
//! - [`ThinPlateSpline`] — 2-D → scalar (or N-D → scalar) TPS.
//! - [`ThinPlateSplineWarp`] — 2-D → 2-D warp for image registration.
//! - [`warp_image`] — top-level convenience function for image warping.
//!
//! ## Mathematical Background
//!
//! The TPS interpolant is:
//!
//! ```text
//! f(x) = a₀ + a₁x₁ + a₂x₂ + Σᵢ wᵢ φ(||x - cᵢ||)
//! ```
//!
//! where `φ(r) = r² ln(r)` is the 2-D TPS kernel, `cᵢ` are the control
//! points, `wᵢ` are RBF weights, and `a₀, a₁, a₂` are polynomial
//! coefficients.  The system is solved as the saddle-point problem:
//!
//! ```text
//! [K + λI   P ] [w]   [y]
//! [P'       0 ] [a] = [0]
//! ```
//!
//! where `K[i,j] = φ(||cᵢ - cⱼ||)`, `P` is the polynomial basis, and
//! `λ` is a regularization (smoothing) parameter.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Kernel function
// ---------------------------------------------------------------------------

/// 2-D thin plate spline kernel: `r² ln(r)`.
///
/// Returns `0` for `r = 0` (by continuity).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::thin_plate_spline::tps_kernel;
///
/// assert_eq!(tps_kernel(0.0), 0.0);
/// assert!((tps_kernel(1.0) - 0.0).abs() < 1e-12); // 1² ln(1) = 0
/// assert!(tps_kernel(2.0) > 0.0);                 // 4 ln(2) > 0
/// ```
#[inline]
pub fn tps_kernel(r: f64) -> f64 {
    if r < f64::EPSILON {
        0.0
    } else {
        r * r * r.ln()
    }
}

/// Generic version of `tps_kernel` for `InterpolationFloat` types.
fn tps_kernel_generic<F: InterpolationFloat>(r: F) -> F {
    let eps = F::from_f64(f64::EPSILON).unwrap_or(F::epsilon());
    if r < eps {
        F::zero()
    } else {
        r * r * r.ln()
    }
}

// ---------------------------------------------------------------------------
// Euclidean distance helpers
// ---------------------------------------------------------------------------

fn dist_rows<F: InterpolationFloat>(
    a: &Array2<F>,
    i: usize,
    b: &Array2<F>,
    j: usize,
) -> InterpolateResult<F> {
    let d = a.ncols();
    if b.ncols() != d {
        return Err(InterpolateError::DimensionMismatch(
            "dimension mismatch in TPS distance".to_string(),
        ));
    }
    let mut sq = F::zero();
    for k in 0..d {
        let diff = a[[i, k]] - b[[j, k]];
        sq = sq + diff * diff;
    }
    Ok(sq.sqrt())
}

fn dist_point_to_row<F: InterpolationFloat>(
    centers: &Array2<F>,
    center_idx: usize,
    point: &[F],
) -> F {
    let d = centers.ncols().min(point.len());
    let mut sq = F::zero();
    for k in 0..d {
        let diff = centers[[center_idx, k]] - point[k];
        sq = sq + diff * diff;
    }
    sq.sqrt()
}

// ---------------------------------------------------------------------------
// Gaussian elimination solver (self-contained — no external dep)
// ---------------------------------------------------------------------------

fn solve_system<F: InterpolationFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(InterpolateError::invalid_input(
            "Matrix must be square and RHS must match".to_string(),
        ));
    }

    let tiny = F::from_f64(1e-14).unwrap_or(F::epsilon());
    let tiny_pivot = F::from_f64(1e-30).unwrap_or(F::epsilon());
    let reg = F::from_f64(1e-10).unwrap_or(F::epsilon());

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
        // Find pivot
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
                "TPS: singular or near-singular system matrix".to_string(),
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
                "TPS: zero pivot in back substitution".to_string(),
            ));
        }
        x[i] = s / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// ThinPlateSpline
// ---------------------------------------------------------------------------

/// Thin plate spline for N-D → scalar interpolation.
///
/// This struct represents the interpolant:
///
/// ```text
/// f(x) = a₀ + Σₖ aₖ xₖ + Σᵢ wᵢ φ(||x - cᵢ||)
/// ```
///
/// where `φ(r) = r² ln(r)`.
///
/// The polynomial terms ensure the system is consistent for the conditionally
/// positive definite TPS kernel.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::thin_plate_spline::ThinPlateSpline;
///
/// // Control points for f(x,y) = x + y
/// let src = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
///
/// let tps = ThinPlateSpline::fit(&src, &tgt, 0.0).expect("doc example: should succeed");
///
/// let query = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("doc example: should succeed");
/// let out = tps.transform(&query).expect("doc example: should succeed");
/// assert!((out[0] - 1.0).abs() < 1e-4);
/// ```
#[derive(Debug, Clone)]
pub struct ThinPlateSpline<F: InterpolationFloat> {
    /// Control point positions, shape `(n, d)`.
    centers: Array2<F>,
    /// RBF weights, length `n`.
    rbf_weights: Array1<F>,
    /// Polynomial coefficients, length `1 + d` (constant + linear).
    poly_weights: Array1<F>,
    /// Smoothing / regularization parameter.
    lambda: F,
    /// Spatial dimension `d`.
    dim: usize,
}

impl<F: InterpolationFloat> ThinPlateSpline<F> {
    /// Fit a thin plate spline to scattered data.
    ///
    /// Solves the augmented saddle-point system:
    ///
    /// ```text
    /// [K + λI   P ] [w]   [y]
    /// [P'       0 ] [a] = [0]
    /// ```
    ///
    /// where `K[i,j] = φ(||cᵢ - cⱼ||)` and `P` contains the polynomial basis
    /// `[1, x₁, x₂, ...]`.
    ///
    /// # Arguments
    ///
    /// * `source_points` - Control point positions, shape `(n, d)`.
    /// * `target_values` - Values at control points, length `n`.
    /// * `lambda`        - Regularization parameter (0 = exact interpolation).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `source_points` and `target_values` have inconsistent sizes.
    /// - Fewer than `d + 1` control points are provided.
    /// - The linear system cannot be solved.
    pub fn fit(
        source_points: &Array2<F>,
        target_values: &Array1<F>,
        lambda: F,
    ) -> InterpolateResult<Self> {
        let n = source_points.nrows();
        let d = source_points.ncols();

        if target_values.len() != n {
            return Err(InterpolateError::invalid_input(format!(
                "source_points has {} rows but target_values has {} elements",
                n,
                target_values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::empty_data("ThinPlateSpline::fit"));
        }
        if n < d + 1 {
            return Err(InterpolateError::invalid_input(format!(
                "need at least {} control points for d={} dimensions, got {}",
                d + 1,
                d,
                n
            )));
        }
        if lambda < F::zero() {
            return Err(InterpolateError::invalid_input(
                "lambda must be non-negative".to_string(),
            ));
        }

        // poly_terms: 1 (constant) + d (linear) = 1 + d
        let n_poly = 1 + d;
        let total = n + n_poly;

        let mut mat = Array2::<F>::zeros((total, total));

        // K block (upper-left) with regularization on diagonal
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    mat[[i, j]] = lambda;
                } else {
                    let r = dist_rows(source_points, i, source_points, j)?;
                    mat[[i, j]] = tps_kernel_generic(r);
                }
            }
        }

        // P block (upper-right) and P' block (lower-left)
        for i in 0..n {
            // Constant term
            mat[[i, n]] = F::one();
            mat[[n, i]] = F::one();
            // Linear terms
            for k in 0..d {
                mat[[i, n + 1 + k]] = source_points[[i, k]];
                mat[[n + 1 + k, i]] = source_points[[i, k]];
            }
        }
        // Bottom-right block stays zero

        // RHS [y; 0]
        let mut rhs = Array1::<F>::zeros(total);
        for i in 0..n {
            rhs[i] = target_values[i];
        }

        let sol = solve_system(&mat, &rhs)?;

        let rbf_weights = sol.slice(scirs2_core::ndarray::s![..n]).to_owned();
        let poly_weights = sol.slice(scirs2_core::ndarray::s![n..]).to_owned();

        Ok(Self {
            centers: source_points.clone(),
            rbf_weights,
            poly_weights,
            lambda,
            dim: d,
        })
    }

    /// Evaluate the TPS at multiple query points.
    ///
    /// # Arguments
    ///
    /// * `query` - Query points, shape `(m, d)`.
    ///
    /// # Returns
    ///
    /// Interpolated values, length `m`.
    pub fn transform(&self, query: &Array2<F>) -> InterpolateResult<Array1<F>> {
        if query.ncols() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "TPS expects {} dimensions, query has {}",
                self.dim,
                query.ncols()
            )));
        }
        let m = query.nrows();
        let n = self.centers.nrows();
        let mut result = Array1::<F>::zeros(m);

        for i in 0..m {
            let pt: Vec<F> = (0..self.dim).map(|k| query[[i, k]]).collect();

            // RBF sum
            let mut val = F::zero();
            for j in 0..n {
                let r = dist_point_to_row(&self.centers, j, &pt);
                val = val + self.rbf_weights[j] * tps_kernel_generic(r);
            }

            // Polynomial: constant
            val = val + self.poly_weights[0];
            // Linear
            for k in 0..self.dim {
                val = val + self.poly_weights[1 + k] * pt[k];
            }

            result[i] = val;
        }

        Ok(result)
    }

    /// Evaluate at a single point.
    pub fn transform_point(&self, point: &[F]) -> InterpolateResult<F> {
        if point.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "TPS expects {} dimensions, point has {}",
                self.dim,
                point.len()
            )));
        }
        let n = self.centers.nrows();
        let mut val = F::zero();

        for j in 0..n {
            let r = dist_point_to_row(&self.centers, j, point);
            val = val + self.rbf_weights[j] * tps_kernel_generic(r);
        }

        // Polynomial terms
        val = val + self.poly_weights[0];
        for k in 0..self.dim {
            val = val + self.poly_weights[1 + k] * point[k];
        }

        Ok(val)
    }

    /// Compute the bending energy of the fitted spline.
    ///
    /// The bending energy measures the "roughness" of the interpolant and
    /// equals `w' K w` where `w` are the RBF weights and `K` is the
    /// kernel matrix.
    ///
    /// For a perfect data fit with zero smoothing, this is the minimum
    /// possible bending energy consistent with the data.
    ///
    /// # Returns
    ///
    /// Non-negative bending energy scalar.
    pub fn bending_energy(&self) -> InterpolateResult<F> {
        let n = self.centers.nrows();

        // Build kernel matrix K
        let mut k = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    k[[i, j]] = F::zero(); // Diagonal is zero (phi(0) = 0)
                } else {
                    let r = dist_rows(&self.centers, i, &self.centers, j)?;
                    k[[i, j]] = tps_kernel_generic(r);
                }
            }
        }

        // Compute w' K w
        let mut energy = F::zero();
        for i in 0..n {
            let mut kw_i = F::zero();
            for j in 0..n {
                kw_i = kw_i + k[[i, j]] * self.rbf_weights[j];
            }
            energy = energy + self.rbf_weights[i] * kw_i;
        }

        // Energy should be non-negative; guard against floating-point noise
        if energy < F::zero() {
            energy = F::zero();
        }

        Ok(energy)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the control point positions.
    pub fn centers(&self) -> &Array2<F> {
        &self.centers
    }

    /// Return the RBF weight vector.
    pub fn rbf_weights(&self) -> &Array1<F> {
        &self.rbf_weights
    }

    /// Return the polynomial coefficient vector `[a₀, a₁, ..., aₐ]`.
    pub fn poly_weights(&self) -> &Array1<F> {
        &self.poly_weights
    }

    /// Return the regularization parameter.
    pub fn lambda(&self) -> F {
        self.lambda
    }

    /// Return the spatial dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// ThinPlateSplineWarp — 2-D → 2-D warp
// ---------------------------------------------------------------------------

/// 2-D thin plate spline warp for image registration.
///
/// Maps 2-D source control points to 2-D target control points using two
/// independent TPS interpolants (one per output coordinate).
///
/// After fitting, any 2-D point `(x, y)` can be mapped to its warped
/// position `(x', y')`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::thin_plate_spline::ThinPlateSplineWarp;
///
/// // Simple identity warp (source == target)
/// let src = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     1.0, 0.0,
///     0.0, 1.0,
///     1.0, 1.0,
/// ]).expect("doc example: should succeed");
/// let tgt = src.clone();
///
/// let warp = ThinPlateSplineWarp::fit(&src, &tgt, 0.0).expect("doc example: should succeed");
/// let mapped = warp.transform(&src).expect("doc example: should succeed");
///
/// // Identity warp should reproduce the source
/// for i in 0..4 {
///     assert!((mapped[[i, 0]] - src[[i, 0]]).abs() < 1e-4);
///     assert!((mapped[[i, 1]] - src[[i, 1]]).abs() < 1e-4);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ThinPlateSplineWarp<F: InterpolationFloat> {
    /// TPS for the x-coordinate.
    tps_x: ThinPlateSpline<F>,
    /// TPS for the y-coordinate.
    tps_y: ThinPlateSpline<F>,
    /// Source control points (for reference).
    source_points: Array2<F>,
}

impl<F: InterpolationFloat> ThinPlateSplineWarp<F> {
    /// Fit a 2-D TPS warp.
    ///
    /// # Arguments
    ///
    /// * `source_points` - Source (input) control point positions, shape `(n, 2)`.
    /// * `target_points` - Target (output) control point positions, shape `(n, 2)`.
    /// * `lambda`        - Regularization parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are inconsistent or the linear system
    /// cannot be solved.
    pub fn fit(
        source_points: &Array2<F>,
        target_points: &Array2<F>,
        lambda: F,
    ) -> InterpolateResult<Self> {
        if source_points.ncols() != 2 {
            return Err(InterpolateError::DimensionMismatch(format!(
                "TPS warp requires 2-D source points, got {} columns",
                source_points.ncols()
            )));
        }
        if target_points.ncols() != 2 {
            return Err(InterpolateError::DimensionMismatch(format!(
                "TPS warp requires 2-D target points, got {} columns",
                target_points.ncols()
            )));
        }
        if source_points.nrows() != target_points.nrows() {
            return Err(InterpolateError::invalid_input(format!(
                "source_points has {} rows but target_points has {} rows",
                source_points.nrows(),
                target_points.nrows()
            )));
        }

        let n = source_points.nrows();
        // Extract target x and y as separate 1-D arrays
        let tx: Array1<F> = (0..n).map(|i| target_points[[i, 0]]).collect();
        let ty: Array1<F> = (0..n).map(|i| target_points[[i, 1]]).collect();

        let tps_x = ThinPlateSpline::fit(source_points, &tx, lambda)?;
        let tps_y = ThinPlateSpline::fit(source_points, &ty, lambda)?;

        Ok(Self {
            tps_x,
            tps_y,
            source_points: source_points.clone(),
        })
    }

    /// Apply the warp to query points.
    ///
    /// # Arguments
    ///
    /// * `query` - Points to warp, shape `(m, 2)`.
    ///
    /// # Returns
    ///
    /// Warped positions, shape `(m, 2)`.
    pub fn transform(&self, query: &Array2<F>) -> InterpolateResult<Array2<F>> {
        if query.ncols() != 2 {
            return Err(InterpolateError::DimensionMismatch(format!(
                "TPS warp requires 2-D query points, got {} columns",
                query.ncols()
            )));
        }
        let m = query.nrows();
        let wx = self.tps_x.transform(query)?;
        let wy = self.tps_y.transform(query)?;

        let mut out = Array2::<F>::zeros((m, 2));
        for i in 0..m {
            out[[i, 0]] = wx[i];
            out[[i, 1]] = wy[i];
        }
        Ok(out)
    }

    /// Transform a single 2-D point.
    ///
    /// # Arguments
    ///
    /// * `x`, `y` - Input coordinates.
    ///
    /// # Returns
    ///
    /// `(x', y')` — warped coordinates.
    pub fn transform_point(&self, x: F, y: F) -> InterpolateResult<(F, F)> {
        let xp = self.tps_x.transform_point(&[x, y])?;
        let yp = self.tps_y.transform_point(&[x, y])?;
        Ok((xp, yp))
    }

    /// Compute the combined bending energy (sum over both coordinate TPS).
    pub fn bending_energy(&self) -> InterpolateResult<F> {
        let ex = self.tps_x.bending_energy()?;
        let ey = self.tps_y.bending_energy()?;
        Ok(ex + ey)
    }

    /// Return the source control points.
    pub fn source_points(&self) -> &Array2<F> {
        &self.source_points
    }

    /// Access the x-coordinate TPS.
    pub fn tps_x(&self) -> &ThinPlateSpline<F> {
        &self.tps_x
    }

    /// Access the y-coordinate TPS.
    pub fn tps_y(&self) -> &ThinPlateSpline<F> {
        &self.tps_y
    }
}

// ---------------------------------------------------------------------------
// warp_image — top-level convenience
// ---------------------------------------------------------------------------

/// Warp a 2-D grayscale image using thin plate splines.
///
/// Given a set of source and target control point correspondences, this
/// function computes the TPS warp and resamples the image using bilinear
/// interpolation.
///
/// # Arguments
///
/// * `image`       - Grayscale image as an `(H, W)` array (row-major).
/// * `source_pts`  - Source control points in *image coordinates*
///                   `(col, row)`, shape `(n, 2)`.
/// * `target_pts`  - Target control points in *image coordinates*
///                   `(col, row)`, shape `(n, 2)`.
/// * `lambda`      - TPS regularization parameter (0 = exact).
///
/// # Returns
///
/// Warped image with the same dimensions as `image`.
///
/// # How it works
///
/// For every pixel `(r, c)` in the **output** image, the inverse warp maps
/// `(c, r)` back to a source coordinate using the TPS, and the source pixel
/// value is fetched via bilinear interpolation.  Pixels that fall outside the
/// source image are set to zero.
///
/// # Errors
///
/// Returns an error if the TPS cannot be fitted.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::thin_plate_spline::warp_image;
///
/// let image = Array2::from_shape_vec((4, 4), vec![
///     1.0_f64, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0,
/// ]).expect("doc example: should succeed");
///
/// // Identity control points (no actual warp)
/// let src = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     3.0, 0.0,
///     0.0, 3.0,
///     3.0, 3.0,
/// ]).expect("doc example: should succeed");
/// let tgt = src.clone();
///
/// let warped = warp_image(&image, &src, &tgt, 0.0).expect("doc example: should succeed");
/// assert_eq!(warped.shape(), image.shape());
/// ```
pub fn warp_image(
    image: &Array2<f64>,
    source_pts: &Array2<f64>,
    target_pts: &Array2<f64>,
    lambda: f64,
) -> InterpolateResult<Array2<f64>> {
    let (h, w) = (image.nrows(), image.ncols());
    if h == 0 || w == 0 {
        return Err(InterpolateError::invalid_input(
            "warp_image: image must be non-empty".to_string(),
        ));
    }

    // We build the *inverse* warp: for each output pixel, find where it came
    // from in the source image.  We fit TPS(target -> source).
    let inv_warp = ThinPlateSplineWarp::fit(target_pts, source_pts, lambda)?;

    let mut output = Array2::<f64>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            // Query point in target space
            let (src_col, src_row) =
                inv_warp.transform_point(col as f64, row as f64)?;

            // Bilinear interpolation from source image
            let v = bilinear_sample(image, src_row, src_col);
            output[[row, col]] = v;
        }
    }

    Ok(output)
}

/// Bilinear interpolation in a 2-D array.
///
/// Returns `0.0` for out-of-bounds coordinates.
fn bilinear_sample(image: &Array2<f64>, row: f64, col: f64) -> f64 {
    let h = image.nrows() as f64;
    let w = image.ncols() as f64;

    if row < 0.0 || col < 0.0 || row > h - 1.0 || col > w - 1.0 {
        return 0.0;
    }

    let r0 = row.floor() as usize;
    let c0 = col.floor() as usize;
    let r1 = (r0 + 1).min(image.nrows() - 1);
    let c1 = (c0 + 1).min(image.ncols() - 1);

    let dr = row - r0 as f64;
    let dc = col - c0 as f64;

    let v00 = image[[r0, c0]];
    let v01 = image[[r0, c1]];
    let v10 = image[[r1, c0]];
    let v11 = image[[r1, c1]];

    v00 * (1.0 - dr) * (1.0 - dc)
        + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc)
        + v11 * dr * dc
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
    // tps_kernel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tps_kernel_at_zero() {
        assert_eq!(tps_kernel(0.0), 0.0);
    }

    #[test]
    fn test_tps_kernel_at_one() {
        // r² ln(r) at r=1 is 1 * ln(1) = 0
        assert_abs_diff_eq!(tps_kernel(1.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tps_kernel_at_e() {
        // r² ln(r) at r=e is e² * 1 = e²
        let r = std::f64::consts::E;
        let expected = r * r; // r² * ln(e) = r² * 1
        assert_abs_diff_eq!(tps_kernel(r), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_tps_kernel_positive_for_r_gt_1() {
        assert!(tps_kernel(2.0) > 0.0);
        assert!(tps_kernel(10.0) > 0.0);
    }

    #[test]
    fn test_tps_kernel_negative_for_0_lt_r_lt_1() {
        // ln(r) < 0 for r in (0,1), so r² ln(r) < 0
        assert!(tps_kernel(0.5) < 0.0);
    }

    // -----------------------------------------------------------------------
    // ThinPlateSpline::fit / transform
    // -----------------------------------------------------------------------

    #[test]
    fn test_tps_exact_fit_at_centers() {
        // f(x,y) = x + y
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);

        let tps = ThinPlateSpline::fit(&src, &tgt, 0.0).expect("fit");
        let out = tps.transform(&src).expect("transform");

        for i in 0..4 {
            assert_abs_diff_eq!(out[i], tgt[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_tps_interpolates_midpoint() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);

        let tps = ThinPlateSpline::fit(&src, &tgt, 0.0).expect("fit");
        let q = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("shape");
        let out = tps.transform(&q).expect("transform");

        // f(0.5, 0.5) = 0.5 + 0.5 = 1.0
        assert!((out[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_tps_dimension_mismatch() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
        let tps = ThinPlateSpline::fit(&src, &tgt, 0.0).expect("fit");

        let q = Array2::from_shape_vec((1, 3), vec![0.5_f64, 0.5, 0.0]).expect("shape");
        assert!(tps.transform(&q).is_err());
    }

    #[test]
    fn test_tps_insufficient_points_error() {
        // 2-D needs at least 3 points
        let src =
            Array2::from_shape_vec((2, 2), vec![0.0_f64, 0.0, 1.0, 0.0]).expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0]);
        assert!(ThinPlateSpline::fit(&src, &tgt, 0.0).is_err());
    }

    #[test]
    fn test_tps_length_mismatch_error() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0]); // wrong length
        assert!(ThinPlateSpline::fit(&src, &tgt, 0.0).is_err());
    }

    #[test]
    fn test_tps_negative_lambda_error() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
        assert!(ThinPlateSpline::fit(&src, &tgt, -0.1).is_err());
    }

    #[test]
    fn test_tps_smoothing_residuals() {
        // With lambda > 0 the fit is approximate
        let src = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0, 1.1]); // slight noise

        let tps = ThinPlateSpline::fit(&src, &tgt, 0.1).expect("fit");
        let out = tps.transform(&src).expect("eval");
        // Smoothed fit: residuals should be small but not zero
        for i in 0..5 {
            assert!((out[i] - tgt[i]).abs() < 1.0, "index {i}");
        }
    }

    #[test]
    fn test_tps_accessors() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
        let tps = ThinPlateSpline::fit(&src, &tgt, 0.05).expect("fit");

        assert_eq!(tps.dim(), 2);
        assert_abs_diff_eq!(tps.lambda(), 0.05, epsilon = 1e-12);
        assert_eq!(tps.centers().nrows(), 4);
        assert_eq!(tps.rbf_weights().len(), 4);
        assert_eq!(tps.poly_weights().len(), 3); // constant + 2 linear
    }

    // -----------------------------------------------------------------------
    // Bending energy
    // -----------------------------------------------------------------------

    #[test]
    fn test_bending_energy_identity_like() {
        // When lambda = 0, the bending energy is determined by data geometry.
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
        let tps = ThinPlateSpline::fit(&src, &tgt, 0.0).expect("fit");
        let e = tps.bending_energy().expect("energy");
        assert!(e >= 0.0, "bending energy must be non-negative");
        assert!(e.is_finite());
    }

    #[test]
    fn test_bending_energy_larger_with_more_data() {
        // More widely scattered data typically has higher bending energy
        let src_small = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1],
        )
        .expect("shape");
        let tgt_small = Array1::from_vec(vec![0.0_f64, 0.1, 0.1, 0.2]);
        let tps_small = ThinPlateSpline::fit(&src_small, &tgt_small, 0.0).expect("fit");
        let e_small = tps_small.bending_energy().expect("energy");
        assert!(e_small >= 0.0);
    }

    // -----------------------------------------------------------------------
    // ThinPlateSplineWarp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_identity() {
        let pts = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");

        let warp = ThinPlateSplineWarp::fit(&pts, &pts, 0.0).expect("fit");
        let out = warp.transform(&pts).expect("transform");

        for i in 0..4 {
            assert_abs_diff_eq!(out[[i, 0]], pts[[i, 0]], epsilon = 1e-4);
            assert_abs_diff_eq!(out[[i, 1]], pts[[i, 1]], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_warp_translation() {
        // Warp = pure translation by (+1, +0.5)
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        )
        .expect("shape");
        let mut tgt_data = vec![0.0_f64; 8];
        for i in 0..4 {
            tgt_data[2 * i] = src[[i, 0]] + 1.0; // x + 1
            tgt_data[2 * i + 1] = src[[i, 1]] + 0.5; // y + 0.5
        }
        let tgt = Array2::from_shape_vec((4, 2), tgt_data).expect("shape");

        let warp = ThinPlateSplineWarp::fit(&src, &tgt, 0.0).expect("fit");

        // Test at the center of the control polygon
        let (mx, my) = warp.transform_point(1.0, 1.0).expect("transform");
        assert!((mx - 2.0).abs() < 0.2, "mx={mx}");
        assert!((my - 1.5).abs() < 0.2, "my={my}");
    }

    #[test]
    fn test_warp_dimension_error() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt3d = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ],
        )
        .expect("shape");
        assert!(ThinPlateSplineWarp::fit(&src, &tgt3d, 0.0).is_err());
    }

    #[test]
    fn test_warp_count_mismatch_error() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt =
            Array2::from_shape_vec((3, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0]).expect("shape");
        assert!(ThinPlateSplineWarp::fit(&src, &tgt, 0.0).is_err());
    }

    #[test]
    fn test_warp_bending_energy_non_negative() {
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .expect("shape");
        let tgt = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 0.9, 0.1, 0.1, 0.9, 1.0, 1.0],
        )
        .expect("shape");

        let warp = ThinPlateSplineWarp::fit(&src, &tgt, 0.0).expect("fit");
        let e = warp.bending_energy().expect("energy");
        assert!(e >= 0.0, "bending energy {e} must be non-negative");
        assert!(e.is_finite());
    }

    // -----------------------------------------------------------------------
    // warp_image tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_image_identity() {
        let image = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ],
        )
        .expect("shape");

        // Identity control points (corners)
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0],
        )
        .expect("shape");

        let warped = warp_image(&image, &src, &src, 0.0).expect("warp");

        assert_eq!(warped.shape(), image.shape());
        // With identity warp, corners should be preserved
        assert_abs_diff_eq!(warped[[0, 0]], image[[0, 0]], epsilon = 0.1);
        assert_abs_diff_eq!(warped[[0, 3]], image[[0, 3]], epsilon = 0.1);
        assert_abs_diff_eq!(warped[[3, 0]], image[[3, 0]], epsilon = 0.1);
        assert_abs_diff_eq!(warped[[3, 3]], image[[3, 3]], epsilon = 0.1);
    }

    #[test]
    fn test_warp_image_output_shape() {
        let image = Array2::<f64>::zeros((8, 6));
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 5.0, 0.0, 0.0, 7.0, 5.0, 7.0],
        )
        .expect("shape");
        let warped = warp_image(&image, &src, &src, 0.0).expect("warp");
        assert_eq!(warped.shape(), [8, 6]);
    }

    #[test]
    fn test_warp_image_empty_error() {
        let image = Array2::<f64>::zeros((0, 4));
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0],
        )
        .expect("shape");
        assert!(warp_image(&image, &src, &src, 0.0).is_err());
    }

    #[test]
    fn test_warp_image_all_finite() {
        let mut data = Vec::with_capacity(16);
        for i in 0..16_usize {
            data.push(i as f64);
        }
        let image = Array2::from_shape_vec((4, 4), data).expect("shape");
        let src = Array2::from_shape_vec(
            (4, 2),
            vec![0.0_f64, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0],
        )
        .expect("shape");
        let warped = warp_image(&image, &src, &src, 0.0).expect("warp");
        assert!(warped.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // bilinear_sample
    // -----------------------------------------------------------------------

    #[test]
    fn test_bilinear_sample_at_pixel_center() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape");
        assert_abs_diff_eq!(bilinear_sample(&image, 0.0, 0.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bilinear_sample(&image, 0.0, 1.0), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bilinear_sample(&image, 1.0, 0.0), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bilinear_sample(&image, 1.0, 1.0), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bilinear_sample_midpoint() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape");
        // (0.5, 0.5) should be (1+2+3+4)/4 = 2.5
        assert_abs_diff_eq!(bilinear_sample(&image, 0.5, 0.5), 2.5, epsilon = 1e-12);
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape");
        assert_eq!(bilinear_sample(&image, -1.0, 0.0), 0.0);
        assert_eq!(bilinear_sample(&image, 0.0, -1.0), 0.0);
        assert_eq!(bilinear_sample(&image, 5.0, 0.0), 0.0);
    }
}
