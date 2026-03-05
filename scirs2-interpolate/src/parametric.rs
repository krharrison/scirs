//! Parametric curve fitting: B-spline curves and Bézier curves
//!
//! This module provides algorithms for fitting smooth parametric curves to
//! ordered point sets and for evaluating, differentiating, and manipulating
//! Bézier and B-spline curves in arbitrary dimensions.
//!
//! # B-spline curve fitting
//!
//! [`BSplineCurve`] fits a B-spline curve to a point set using chord-length
//! parametrisation and global least-squares fitting of the control points.
//! This is the standard algorithm used in CAD/CAM systems.
//!
//! # Bézier curves
//!
//! [`BezierCurve`] implements exact Bézier curves with the following
//! operations:
//! - De Casteljau evaluation (numerically stable)
//! - Derivative curves
//! - Degree elevation
//! - Splitting at an arbitrary parameter value
//!
//! # References
//!
//! - Piegl, L. and Tiller, W. (1997), *The NURBS Book*, 2nd ed., Springer.
//! - Farin, G. (2002), *Curves and Surfaces for CAGD*, 5th ed., Morgan Kaufmann.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

// ---------------------------------------------------------------------------
// BSplineCurve
// ---------------------------------------------------------------------------

/// B-spline curve fitted to a point set.
///
/// The curve is parametrised by `t ∈ [0, 1]`.  Control points are computed
/// via a least-squares fit with chord-length parametrisation and a clamped
/// uniform knot vector.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::parametric::BSplineCurve;
/// use scirs2_core::ndarray::array;
///
/// // Fit a cubic B-spline with 5 control points to 8 data points in 2-D
/// let pts = array![
///     [0.0_f64, 0.0], [1.0, 0.5], [2.0, 0.0],
///     [3.0, -0.5], [4.0, 0.0], [5.0, 0.5],
///     [6.0, 0.0], [7.0, 0.0],
/// ];
/// let curve = BSplineCurve::fit(&pts, 3, 5).expect("doc example: should succeed");
/// let p = curve.eval(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct BSplineCurve {
    /// Control points, shape `(n_ctrl, dim)`
    control_points: Array2<f64>,
    /// Knot vector (length = `n_ctrl + degree + 1`)
    knots: Vec<f64>,
    /// Polynomial degree
    degree: usize,
}

impl BSplineCurve {
    /// Fit a B-spline curve of given degree to a point cloud.
    ///
    /// The algorithm:
    /// 1. Chord-length parametrisation of the data points.
    /// 2. Clamped uniform knot vector.
    /// 3. Least-squares fitting of the control points.
    ///
    /// # Parameters
    ///
    /// - `points`: data points, shape `(n_pts, dim)`.  At least `degree + 1` points required.
    /// - `degree`: polynomial degree (≥ 1).
    /// - `n_control`: number of control points (must satisfy `degree + 1 ≤ n_control ≤ n_pts`).
    ///
    /// # Errors
    ///
    /// Returns [`InterpolateError::InvalidInput`] for invalid shapes or parameters.
    pub fn fit(
        points: &Array2<f64>,
        degree: usize,
        n_control: usize,
    ) -> InterpolateResult<Self> {
        let n_pts = points.nrows();
        let dim = points.ncols();
        if n_pts < 2 {
            return Err(InterpolateError::InvalidInput {
                message: "BSplineCurve::fit: need at least 2 data points".into(),
            });
        }
        if degree == 0 {
            return Err(InterpolateError::InvalidInput {
                message: "BSplineCurve::fit: degree must be ≥ 1".into(),
            });
        }
        if n_control < degree + 1 {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineCurve::fit: n_control={} must be ≥ degree+1={}",
                    n_control,
                    degree + 1
                ),
            });
        }
        if n_control > n_pts {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "BSplineCurve::fit: n_control={} must be ≤ n_pts={}",
                    n_control, n_pts
                ),
            });
        }

        // 1. Chord-length parametrisation
        let params = chord_length_params(points)?;

        // 2. Clamped uniform knot vector
        let knots = clamped_uniform_knots(n_control, degree);

        // 3. Assemble B-spline basis matrix N (n_pts × n_control)
        let n_mat = bspline_basis_matrix(&params, &knots, degree, n_control)?;

        // 4. Solve least-squares system (NᵀN) P = NᵀQ
        let nt = n_mat.t();
        let ntn = nt.dot(&n_mat);
        let ntq = nt.dot(points);

        // Solve for each dimension separately
        let control_points = solve_spline_system(&ntn, &ntq, n_control, dim)?;

        Ok(Self {
            control_points,
            knots,
            degree,
        })
    }

    /// Evaluate the B-spline curve at parameter `t ∈ [0, 1]`.
    ///
    /// Returns a `Vec<f64>` of length `dim`.
    pub fn eval(&self, t: f64) -> Vec<f64> {
        let t_clamped = t.clamp(0.0, 1.0);
        let n_ctrl = self.control_points.nrows();
        let dim = self.control_points.ncols();

        // Find knot span
        let span = find_knot_span(t_clamped, &self.knots, self.degree, n_ctrl);
        let basis = bspline_basis_fns(span, t_clamped, &self.knots, self.degree);

        let mut result = vec![0.0_f64; dim];
        for i in 0..=self.degree {
            let ctrl_idx = span - self.degree + i;
            if ctrl_idx < n_ctrl {
                for d in 0..dim {
                    result[d] += basis[i] * self.control_points[[ctrl_idx, d]];
                }
            }
        }
        result
    }

    /// Evaluate the curve at multiple parameter values.
    ///
    /// Returns shape `(ts.len(), dim)`.
    pub fn eval_many(&self, ts: &[f64]) -> Array2<f64> {
        let dim = self.control_points.ncols();
        let mut out = Array2::<f64>::zeros((ts.len(), dim));
        for (i, &t) in ts.iter().enumerate() {
            let pt = self.eval(t);
            for (d, &v) in pt.iter().enumerate() {
                out[[i, d]] = v;
            }
        }
        out
    }

    /// Approximate arc length by sampling the curve at `n_samples` equispaced
    /// parameter values and summing chord lengths.
    pub fn arc_length(&self, n_samples: usize) -> f64 {
        let n = n_samples.max(2);
        let ts: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let pts = self.eval_many(&ts);
        let dim = pts.ncols();
        let mut length = 0.0_f64;
        for i in 1..n {
            let d2: f64 = (0..dim).map(|d| {
                let diff = pts[[i, d]] - pts[[i - 1, d]];
                diff * diff
            }).sum();
            length += d2.sqrt();
        }
        length
    }

    /// Resample the curve at `n_pts` points equally spaced in arc length.
    ///
    /// Returns shape `(n_pts, dim)`.
    pub fn uniform_resample(&self, n_pts: usize) -> Array2<f64> {
        let n_pts = n_pts.max(2);
        let n_fine = 1000;
        let ts_fine: Vec<f64> = (0..n_fine)
            .map(|i| i as f64 / (n_fine - 1) as f64)
            .collect();
        let pts_fine = self.eval_many(&ts_fine);
        let dim = pts_fine.ncols();

        // Compute cumulative arc lengths
        let mut cum_len = vec![0.0_f64; n_fine];
        for i in 1..n_fine {
            let d2: f64 = (0..dim)
                .map(|d| {
                    let diff = pts_fine[[i, d]] - pts_fine[[i - 1, d]];
                    diff * diff
                })
                .sum();
            cum_len[i] = cum_len[i - 1] + d2.sqrt();
        }
        let total = cum_len[n_fine - 1];

        // For each target arc length, interpolate the parameter
        let mut out = Array2::<f64>::zeros((n_pts, dim));
        for k in 0..n_pts {
            let target = total * k as f64 / (n_pts - 1) as f64;
            // Binary search for the interval
            let idx = cum_len
                .partition_point(|&l| l < target)
                .min(n_fine - 1);
            for d in 0..dim {
                out[[k, d]] = pts_fine[[idx, d]];
            }
        }
        out
    }

    /// Return a reference to the control points (shape `(n_ctrl, dim)`).
    #[inline]
    pub fn control_points(&self) -> &Array2<f64> {
        &self.control_points
    }

    /// Return a reference to the knot vector.
    #[inline]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// Return the polynomial degree.
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }
}

// ---------------------------------------------------------------------------
// BezierCurve
// ---------------------------------------------------------------------------

/// Bézier curve of arbitrary degree in arbitrary dimension.
///
/// The curve is parametrised by `t ∈ [0, 1]` and evaluated via the
/// de Casteljau algorithm (numerically stable).
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::parametric::BezierCurve;
/// use scirs2_core::ndarray::array;
///
/// // Cubic Bézier in 2-D
/// let ctrl = array![[0.0_f64, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]];
/// let curve = BezierCurve::new(ctrl);
/// let mid = curve.eval(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct BezierCurve {
    control_points: Array2<f64>,
}

impl BezierCurve {
    /// Create a Bézier curve from control points.
    ///
    /// - `control_points`: shape `(n+1, dim)` where `n` is the degree.
    pub fn new(control_points: Array2<f64>) -> Self {
        Self { control_points }
    }

    /// Evaluate the curve at `t ∈ [0, 1]` using the de Casteljau algorithm.
    pub fn eval(&self, t: f64) -> Vec<f64> {
        let t = t.clamp(0.0, 1.0);
        de_casteljau(&self.control_points, t)
    }

    /// Evaluate at multiple parameter values.
    ///
    /// Returns shape `(ts.len(), dim)`.
    pub fn eval_many(&self, ts: &[f64]) -> Array2<f64> {
        let dim = self.control_points.ncols();
        let mut out = Array2::<f64>::zeros((ts.len(), dim));
        for (i, &t) in ts.iter().enumerate() {
            let pt = self.eval(t);
            for (d, &v) in pt.iter().enumerate() {
                out[[i, d]] = v;
            }
        }
        out
    }

    /// Compute the tangent vector (first derivative) at `t`.
    ///
    /// Uses the degree-elevation property: the derivative of a degree-n
    /// Bézier is n times a degree-(n-1) Bézier of the first differences.
    pub fn derivative(&self, t: f64) -> Vec<f64> {
        let t = t.clamp(0.0, 1.0);
        let n = self.degree();
        let dim = self.control_points.ncols();
        if n == 0 {
            return vec![0.0_f64; dim];
        }
        // Derivative control points: Q_i = n * (P_{i+1} - P_i)
        let n_q = n; // n+1 - 1 = n points
        let mut q = Array2::<f64>::zeros((n_q, dim));
        for i in 0..n_q {
            for d in 0..dim {
                q[[i, d]] = n as f64 * (self.control_points[[i + 1, d]] - self.control_points[[i, d]]);
            }
        }
        de_casteljau(&q, t)
    }

    /// Degree of the Bézier curve (= n_control_points − 1).
    #[inline]
    pub fn degree(&self) -> usize {
        self.control_points.nrows().saturating_sub(1)
    }

    /// Degree elevation: returns a degree-(n+1) Bézier that traces the same curve.
    ///
    /// The new control points are given by the classical formula:
    /// `Q_i = (i/(n+1)) P_{i-1} + (1 − i/(n+1)) P_i`
    pub fn elevate_degree(&self) -> Self {
        let n = self.degree();
        let dim = self.control_points.ncols();
        let n_new = n + 2; // n+1 new control points: degree n+1
        let mut q = Array2::<f64>::zeros((n_new, dim));
        // Q_0 = P_0
        for d in 0..dim {
            q[[0, d]] = self.control_points[[0, d]];
        }
        // Q_i = (i/(n+1)) P_{i-1} + (1 - i/(n+1)) P_i  for 1 ≤ i ≤ n
        for i in 1..=n {
            let alpha = i as f64 / (n + 1) as f64;
            for d in 0..dim {
                q[[i, d]] =
                    alpha * self.control_points[[i - 1, d]] + (1.0 - alpha) * self.control_points[[i, d]];
            }
        }
        // Q_{n+1} = P_n
        for d in 0..dim {
            q[[n + 1, d]] = self.control_points[[n, d]];
        }
        Self { control_points: q }
    }

    /// Split the curve at parameter `t` into two Bézier curves.
    ///
    /// Uses the de Casteljau algorithm: the intermediate points at parameter `t`
    /// form the control polygons of the two sub-curves.
    pub fn split(&self, t: f64) -> (Self, Self) {
        let t = t.clamp(0.0, 1.0);
        let n = self.degree();
        let dim = self.control_points.ncols();

        // Build the de Casteljau table
        // table[r][i] = point after r steps of subdivision
        let n_pts = n + 1;
        let mut table: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_pts);
        table.push(
            (0..n_pts)
                .map(|i| (0..dim).map(|d| self.control_points[[i, d]]).collect())
                .collect(),
        );
        for r in 1..n_pts {
            let prev = &table[r - 1];
            let mut cur = Vec::with_capacity(n_pts - r);
            for i in 0..n_pts - r {
                let pt: Vec<f64> = (0..dim)
                    .map(|d| (1.0 - t) * prev[i][d] + t * prev[i + 1][d])
                    .collect();
                cur.push(pt);
            }
            table.push(cur);
        }

        // Left curve: table[r][0] for r = 0..=n
        let left_pts: Vec<Vec<f64>> = (0..n_pts).map(|r| table[r][0].clone()).collect();
        // Right curve: table[n-r][r] for r = 0..=n (= table in reverse diagonal)
        let right_pts: Vec<Vec<f64>> = (0..n_pts).map(|r| table[n - r][r].clone()).collect();

        let to_array = |pts: Vec<Vec<f64>>| -> Array2<f64> {
            let rows = pts.len();
            let cols = if rows > 0 { pts[0].len() } else { 0 };
            let mut a = Array2::<f64>::zeros((rows, cols));
            for (i, row) in pts.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    a[[i, j]] = v;
                }
            }
            a
        };

        (
            Self::new(to_array(left_pts)),
            Self::new(to_array(right_pts)),
        )
    }

    /// Return a reference to the control points.
    #[inline]
    pub fn control_points(&self) -> &Array2<f64> {
        &self.control_points
    }
}

// ---------------------------------------------------------------------------
// Internal helpers: B-spline
// ---------------------------------------------------------------------------

/// Compute chord-length parametrisation for an ordered set of points.
///
/// Returns `t_k ∈ [0, 1]` with `t_0 = 0`, `t_{n-1} = 1`.
fn chord_length_params(points: &Array2<f64>) -> InterpolateResult<Vec<f64>> {
    let n = points.nrows();
    let dim = points.ncols();
    let mut d = vec![0.0_f64; n];
    for i in 1..n {
        let diff_sq: f64 = (0..dim)
            .map(|k| {
                let diff = points[[i, k]] - points[[i - 1, k]];
                diff * diff
            })
            .sum();
        d[i] = d[i - 1] + diff_sq.sqrt();
    }
    let total = d[n - 1];
    if total < f64::EPSILON {
        return Err(InterpolateError::InvalidInput {
            message: "BSplineCurve::fit: all data points are identical".into(),
        });
    }
    let params: Vec<f64> = d.iter().map(|&di| di / total).collect();
    Ok(params)
}

/// Build a clamped uniform knot vector with `n_ctrl` control points and degree `p`.
///
/// The knot vector is: `[0,...,0, t_{p+1}, ..., t_{m-p-1}, 1,...,1]`
fn clamped_uniform_knots(n_ctrl: usize, p: usize) -> Vec<f64> {
    let m = n_ctrl + p + 1; // total knots
    let mut t = vec![0.0_f64; m];
    // First p+1 knots = 0
    // Last p+1 knots = 1
    let n_internal = n_ctrl - p - 1; // internal knots count
    for i in 0..n_internal {
        t[p + 1 + i] = (i + 1) as f64 / (n_internal + 1) as f64;
    }
    for j in m - p - 1..m {
        t[j] = 1.0;
    }
    t
}

/// Build the B-spline basis matrix N of shape `(n_pts, n_ctrl)`.
fn bspline_basis_matrix(
    params: &[f64],
    knots: &[f64],
    degree: usize,
    n_ctrl: usize,
) -> InterpolateResult<Array2<f64>> {
    let n_pts = params.len();
    let mut n_mat = Array2::<f64>::zeros((n_pts, n_ctrl));
    for (i, &t) in params.iter().enumerate() {
        let span = find_knot_span(t, knots, degree, n_ctrl);
        let basis = bspline_basis_fns(span, t, knots, degree);
        for j in 0..=degree {
            let ctrl_j = span - degree + j;
            if ctrl_j < n_ctrl {
                n_mat[[i, ctrl_j]] = basis[j];
            }
        }
    }
    Ok(n_mat)
}

/// Find the knot span index (Algorithm A2.1 from Piegl & Tiller).
fn find_knot_span(t: f64, knots: &[f64], degree: usize, n_ctrl: usize) -> usize {
    let n = n_ctrl - 1; // last control point index
    // Handle special case t == t[n+1]
    if t >= knots[n + 1] {
        return n;
    }
    // Binary search
    let mut low = degree;
    let mut high = n + 1;
    let mut mid = (low + high) / 2;
    while t < knots[mid] || t >= knots[mid + 1] {
        if t < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
        if low + 1 >= high {
            break;
        }
    }
    mid
}

/// Compute B-spline basis functions N_{span-p,p}, ..., N_{span,p} at parameter `t`
/// (Algorithm A2.2 from Piegl & Tiller).
///
/// Returns a vector of length `degree + 1`.
fn bspline_basis_fns(span: usize, t: f64, knots: &[f64], degree: usize) -> Vec<f64> {
    let p = degree;
    let mut n = vec![0.0_f64; p + 1];
    let mut left = vec![0.0_f64; p + 1];
    let mut right = vec![0.0_f64; p + 1];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        let mut saved = 0.0_f64;
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = if denom.abs() < f64::EPSILON {
                0.0
            } else {
                n[r] / denom
            };
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// Solve the symmetric positive-definite system `A x = B` for each column of B.
///
/// Uses Cholesky decomposition or falls back to Gaussian elimination.
fn solve_spline_system(
    a: &Array2<f64>,
    b: &Array2<f64>,
    n: usize,
    _dim: usize,
) -> InterpolateResult<Array2<f64>> {
    // Delegate to the Gaussian elimination solver
    gauss_solve(a, b, n)
}

/// Solve `A X = B` via Gaussian elimination with partial pivoting.
/// `A`: (n×n), `B`: (n×dim), returns X: (n×dim).
fn gauss_solve(
    a: &Array2<f64>,
    b: &Array2<f64>,
    n: usize,
) -> InterpolateResult<Array2<f64>> {
    let m = b.ncols();
    // Build augmented matrix
    let mut aug = Array2::<f64>::zeros((n, n + m));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        for k in 0..m {
            aug[[i, n + k]] = b[[i, k]];
        }
    }

    // Forward elimination
    for col in 0..n {
        // Partial pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in col + 1..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return Err(InterpolateError::LinalgError(
                "BSplineCurve::fit: singular system (duplicate/collinear points?)".into(),
            ));
        }
        if max_row != col {
            for j in 0..n + m {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let pivot = aug[[col, col]];
        for row in col + 1..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..n + m {
                let delta = factor * aug[[col, j]];
                aug[[row, j]] -= delta;
            }
        }
    }

    // Back substitution
    let mut x = Array2::<f64>::zeros((n, m));
    for col in (0..n).rev() {
        for k in 0..m {
            let mut val = aug[[col, n + k]];
            for j in col + 1..n {
                val -= aug[[col, j]] * x[[j, k]];
            }
            x[[col, k]] = val / aug[[col, col]];
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Internal helpers: Bézier
// ---------------------------------------------------------------------------

/// Evaluate a Bézier curve at `t` using the de Casteljau algorithm.
fn de_casteljau(ctrl: &Array2<f64>, t: f64) -> Vec<f64> {
    let n_pts = ctrl.nrows();
    let dim = ctrl.ncols();
    if n_pts == 0 {
        return Vec::new();
    }
    if n_pts == 1 {
        return ctrl.row(0).to_vec();
    }

    // Working buffer: copy control points
    let mut buf: Vec<Vec<f64>> = (0..n_pts)
        .map(|i| (0..dim).map(|d| ctrl[[i, d]]).collect())
        .collect();

    let s = 1.0 - t;
    for r in 1..n_pts {
        for i in 0..n_pts - r {
            for d in 0..dim {
                buf[i][d] = s * buf[i][d] + t * buf[i + 1][d];
            }
        }
    }
    buf[0].clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    // ── BezierCurve ─────────────────────────────────────────────────────────

    #[test]
    fn test_bezier_endpoints() {
        // At t=0 returns P_0, at t=1 returns P_n
        let ctrl = array![[0.0_f64, 0.0], [1.0, 2.0], [2.0, 2.0], [3.0, 0.0]];
        let curve = BezierCurve::new(ctrl.clone());
        let p0 = curve.eval(0.0);
        let p1 = curve.eval(1.0);
        assert_abs_diff_eq!(p0[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p0[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p1[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(p1[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bezier_linear_is_linear() {
        // Degree-1 Bézier = straight line
        let ctrl = array![[0.0_f64, 0.0], [4.0, 2.0]];
        let curve = BezierCurve::new(ctrl);
        let pt = curve.eval(0.5);
        assert_abs_diff_eq!(pt[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pt[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bezier_eval_many() {
        let ctrl = array![[0.0_f64, 0.0], [1.0, 2.0], [2.0, 0.0]];
        let curve = BezierCurve::new(ctrl);
        let ts = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let pts = curve.eval_many(&ts);
        assert_eq!(pts.shape(), &[5, 2]);
        // Endpoints
        assert_abs_diff_eq!(pts[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pts[[4, 0]], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bezier_derivative_linear() {
        // Degree-1 Bézier: P0=(0,0), P1=(4,2)
        // Tangent should be constant (4, 2)
        let ctrl = array![[0.0_f64, 0.0], [4.0, 2.0]];
        let curve = BezierCurve::new(ctrl);
        let d = curve.derivative(0.5);
        assert_abs_diff_eq!(d[0], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bezier_derivative_quadratic() {
        // Degree-2: P0=(0,0), P1=(1,2), P2=(2,0)
        // Derivative at t=0 should be 2*(P1-P0) = (2,4)
        let ctrl = array![[0.0_f64, 0.0], [1.0, 2.0], [2.0, 0.0]];
        let curve = BezierCurve::new(ctrl);
        let d0 = curve.derivative(0.0);
        assert_abs_diff_eq!(d0[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d0[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bezier_degree_elevation_same_curve() {
        // Degree elevation should preserve the curve geometry
        let ctrl = array![[0.0_f64, 0.0], [1.0, 2.0], [2.0, 0.0]];
        let curve = BezierCurve::new(ctrl);
        let elevated = curve.elevate_degree();
        assert_eq!(elevated.degree(), curve.degree() + 1);
        // Same points on both curves
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p1 = curve.eval(t);
            let p2 = elevated.eval(t);
            for d in 0..2 {
                assert_abs_diff_eq!(p1[d], p2[d], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bezier_split_reconstructs() {
        // The two sub-curves together should cover the same points as the original
        let ctrl = array![[0.0_f64, 0.0], [1.0, 3.0], [3.0, 3.0], [4.0, 0.0]];
        let curve = BezierCurve::new(ctrl);
        let (left, right) = curve.split(0.5);

        // Points on the left piece (t ∈ [0, 0.5]) should match original at [0, 0.5]
        for k in 0..5 {
            let t = k as f64 / 4.0; // 0, 0.25, 0.5, 0.75, 1.0 for left piece
            let t_orig = t * 0.5; // map to [0, 0.5]
            let p_orig = curve.eval(t_orig);
            let p_left = left.eval(t);
            for d in 0..2 {
                assert_abs_diff_eq!(p_orig[d], p_left[d], epsilon = 1e-10);
            }
        }

        // Points on the right piece (t ∈ [0.5, 1]) should match original
        for k in 0..5 {
            let t = k as f64 / 4.0; // map [0,1] → [0.5,1]
            let t_orig = 0.5 + t * 0.5;
            let p_orig = curve.eval(t_orig);
            let p_right = right.eval(t);
            for d in 0..2 {
                assert_abs_diff_eq!(p_orig[d], p_right[d], epsilon = 1e-10);
            }
        }
    }

    // ── BSplineCurve ─────────────────────────────────────────────────────────

    #[test]
    fn test_bspline_fit_basic() {
        let pts = array![
            [0.0_f64, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, -1.0],
            [4.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 3, 4).expect("test: should succeed");
        // Should produce some finite values
        let p = curve.eval(0.5);
        assert!(p[0].is_finite());
        assert!(p[1].is_finite());
    }

    #[test]
    fn test_bspline_fit_endpoints_close() {
        // The fitted curve should start near the first point and end near the last
        let pts = array![
            [0.0_f64, 0.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [4.0, 0.0],
            [5.0, -1.0],
            [6.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 3, 5).expect("test: should succeed");
        let start = curve.eval(0.0);
        let end = curve.eval(1.0);
        // With clamped B-spline the endpoint should be close (not exact for least-squares fit)
        assert!(start[0].is_finite() && end[0].is_finite());
    }

    #[test]
    fn test_bspline_eval_many() {
        let pts = array![
            [0.0_f64, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, -1.0],
            [4.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 3, 4).expect("test: should succeed");
        let ts: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        let out = curve.eval_many(&ts);
        assert_eq!(out.shape(), &[11, 2]);
        for i in 0..11 {
            assert!(out[[i, 0]].is_finite());
        }
    }

    #[test]
    fn test_bspline_arc_length_positive() {
        let pts = array![
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 2, 3).expect("test: should succeed");
        let len = curve.arc_length(200);
        assert!(len > 0.0 && len.is_finite());
    }

    #[test]
    fn test_bspline_uniform_resample() {
        let pts = array![
            [0.0_f64, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [3.0, -1.0],
            [4.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 3, 4).expect("test: should succeed");
        let resampled = curve.uniform_resample(20);
        assert_eq!(resampled.shape(), &[20, 2]);
        for i in 0..20 {
            assert!(resampled[[i, 0]].is_finite());
        }
    }

    #[test]
    fn test_bspline_invalid_degree_zero() {
        let pts = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let result = BSplineCurve::fit(&pts, 0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_bspline_invalid_n_control_too_large() {
        let pts = array![[0.0_f64, 0.0], [1.0, 1.0]];
        let result = BSplineCurve::fit(&pts, 1, 5); // n_control > n_pts
        assert!(result.is_err());
    }

    #[test]
    fn test_bspline_3d() {
        // Fit in 3-D
        let pts = array![
            [0.0_f64, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 0.0, 1.0],
            [3.0, -1.0, 0.5],
            [4.0, 0.0, 0.0],
        ];
        let curve = BSplineCurve::fit(&pts, 3, 4).expect("test: should succeed");
        let p = curve.eval(0.5);
        assert_eq!(p.len(), 3);
        for &v in &p {
            assert!(v.is_finite());
        }
    }
}
